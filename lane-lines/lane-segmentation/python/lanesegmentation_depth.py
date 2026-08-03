import os
import argparse
import time
import sys

# 1. Setup CUDA Context immediately
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import cv2
import numpy as np
import pyzed.sl as sl

# --- CONFIGURATION ---
INPUT_H = 512
INPUT_W = 896
INPUT_C = 3
OUTPUT_H = 512
OUTPUT_W = 896
OUTPUT_C = 4  # BG, Road, Curbs, Marks

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# --- HELPER CLASS (Unchanged) ---
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self): return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self): return self.__str__()

# --- ENGINE CLASS (Unchanged) ---
class LaneSegmentationEngine:
    def __init__(self, engine_path):
        self.cuda_ctx = pycuda.autoinit.context
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.allocations = self._allocate_buffers(self.engine)

    def _load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self, engine):
        allocations = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dims = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            vol = 1
            for d in dims: vol *= abs(d)
            host_mem = cuda.pagelocked_empty(vol, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            allocations[name] = HostDeviceMem(host_mem, device_mem)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, (1, INPUT_H, INPUT_W, INPUT_C))
        return allocations

    def infer(self, image):
        self.cuda_ctx.push()
        try:
            resized = cv2.resize(image, (INPUT_W, INPUT_H))
            input_image = np.expand_dims(resized, axis=0).astype(np.float32)
            input_image = np.ascontiguousarray(input_image)
            
            input_name, output_name = None, None
            for name, memory in self.allocations.items():
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    input_name = name
                    np.copyto(memory.host, input_image.ravel())
                    cuda.memcpy_htod_async(memory.device, memory.host, self.stream)
                else:
                    output_name = name
            
            for name, memory in self.allocations.items():
                self.context.set_tensor_address(name, int(memory.device))
            
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            output_mem = self.allocations[output_name]
            cuda.memcpy_dtoh_async(output_mem.host, output_mem.device, self.stream)
            self.stream.synchronize()
            
            return np.array(output_mem.host, dtype=np.float32).reshape((1, OUTPUT_H, OUTPUT_W, OUTPUT_C))
        finally:
            self.cuda_ctx.pop()

def apply_geometric_prior(scores, point_cloud_xyz):
    """
    Modifies scores based on geometry, but ONLY where depth data is valid.
    - If depth is valid AND on ground -> BOOST Road.
    - If depth is valid AND NOT on ground -> PENALIZE Road.
    - If depth is INVALID (too far/NaN) -> DO NOTHING (Trust Neural Net).
    """
    # 1. Resize Z-channel to match model output
    z_height = point_cloud_xyz[:, :, 2] 
    z_height_resized = cv2.resize(z_height, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_NEAREST)

    # 2. Check for Validity (Finite numbers only, no NaNs or Infs)
    # This creates a mask of where the camera can actually "see" depth
    has_depth_data = np.isfinite(z_height_resized)

    # 3. Find Ground Level (using only valid points)
    h, w = z_height_resized.shape
    sample_patch = z_height_resized[h-50:h, w//2-50:w//2+50] 
    valid_samples = sample_patch[np.isfinite(sample_patch)]
    
    if len(valid_samples) == 0:
        return scores # Camera is blind, return original scores

    ground_level = np.median(valid_samples)

    # 4. Determine Geometry
    HEIGHT_TOLERANCE = 0.20 # +/- 20cm
    
    # Calculate distance from ground (ignore NaNs for now, we filter later)
    # We use 'nan_to_num' just to suppress warnings, the mask handles the logic
    dist_from_ground = np.abs(np.nan_to_num(z_height_resized) - ground_level)
    
    # Logic:
    # It is GROUND if: (Within tolerance) AND (Has Valid Depth)
    is_confirmed_ground = (dist_from_ground < HEIGHT_TOLERANCE) & has_depth_data
    
    # It is OBSTACLE if: (Outside tolerance) AND (Has Valid Depth)
    is_confirmed_obstacle = (dist_from_ground >= HEIGHT_TOLERANCE) & has_depth_data

    # 5. Apply Logic
    ROAD_IDX = 1
    BOOST_AMOUNT = 0.1
    PENALTY_AMOUNT = 0.1

    # BOOST: Only where we are SURE it is ground
    scores[:, :, ROAD_IDX] += (is_confirmed_ground * BOOST_AMOUNT)

    # PENALTY: Only where we are SURE it is an obstacle
    # (Previously, this penalty was applied to 'Everything not ground', including NaNs)
    scores[:, :, ROAD_IDX] -= (is_confirmed_obstacle * PENALTY_AMOUNT)
    
    # Result: Pixels with NO depth data (far away) get +0 and -0, 
    # preserving the original neural network prediction.
    
    return scores   

# --- ZED SETUP (REVERTED TO ORIGINAL) ---
def setup_zed_camera(opt):
    # RESTORED: Exact parameters from your working script
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080,
                             depth_mode=sl.DEPTH_MODE.NEURAL,
                             coordinate_units=sl.UNIT.METER,
                             camera_fps=60,
                             depth_minimum_distance=1.0,
                             depth_maximum_distance=40,
                             depth_stabilization=0,
                             async_grab_camera_recovery=True,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD)

    if len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4:
            if len(ip_str.split(':')) == 2:
                init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
            else:
                init.set_from_stream(ip_str)
            print(f"[Sample] Using Stream input, IP: {ip_str}")

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to open ZED Camera: {status}")
        sys.exit(1)
    return zed

def main():
    # ... (Argparse and Init code remains the same) ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_file', type=str, default="weights/lane_segmentation.engine")
    parser.add_argument('--ip_address', type=str, default='127.0.0.1:30000')
    parser.add_argument('--debug_mode', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.engine_file):
        print(f"Error: Engine file not found at {args.engine_file}")
        sys.exit(1)

    model = LaneSegmentationEngine(args.engine_file)
    zed = setup_zed_camera(args)
    
    runtime_parameters = sl.RuntimeParameters()
    left_image = sl.Mat()
    point_cloud = sl.Mat() 

    print("Starting Inference with Geometric Priors...")
    try:
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # 1. Retrieve Data
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                
                rgb_frame = left_image.get_data()
                if rgb_frame.shape[2] == 4:
                    ip_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGRA2BGR)
                else:
                    ip_frame = rgb_frame
                
                point_cloud_data = point_cloud.get_data()

                # 2. Neural Net Inference
                output_data = model.infer(ip_frame)
                
                # output_data is (1, 512, 896, 4)
                raw_scores = output_data[0] 

                # 3. --- APPLY GEOMETRIC PRIOR (BOOST/PENALTY) ---
                # We modify the scores *before* deciding the class
                enhanced_scores = apply_geometric_prior(raw_scores, point_cloud_data)

                # 4. Argmax to get final class
                pred_indices = np.argmax(enhanced_scores, axis=2)
                
                if args.debug_mode:
                    color_map = np.array([[70, 70, 70], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
                    mask = color_map[pred_indices]
                    mask_resized = cv2.resize(mask, (ip_frame.shape[1], ip_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    overlay = cv2.addWeighted(ip_frame, 0.7, mask_resized, 0.3, 0)
                    cv2.imshow("Lane Segmentation", overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except KeyboardInterrupt:
        print("Stopping...")

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()