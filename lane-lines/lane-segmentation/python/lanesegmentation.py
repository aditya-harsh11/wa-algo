import os
import argparse
import time
import sys

import pyzed.sl as sl

# 1. Setup CUDA Context immediately
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import cv2
import numpy as np
from itertools import count

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

# --- HELPER CLASS FOR MEMORY ALLOCATION ---
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# --- MAIN ENGINE CLASS ---
class LaneSegmentationEngine:
    def __init__(self, engine_path):
        print(f"Loading Engine: {engine_path}")
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
            
            # Calculate volume
            vol = 1
            for d in dims:
                if d < 0: vol *= 1
                else: vol *= d
            
            host_mem = cuda.pagelocked_empty(vol, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            allocations[name] = HostDeviceMem(host_mem, device_mem)
            
            # FIX 1: Set shape to NHWC (1, 512, 896, 3) matching your error log
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, (1, INPUT_H, INPUT_W, INPUT_C))

        return allocations

    def infer(self, image):

        start_time = time.perf_counter()
        
        self.cuda_ctx.push()
        try:
            # FIX 2: Remove the transpose! 
            # OpenCV gives (512, 896, 3). Model wants (1, 512, 896, 3).
            resized = cv2.resize(image, (INPUT_W, INPUT_H))
            
            # Just add the batch dimension. No transpose (2,0,1).
            input_image = np.expand_dims(resized, axis=0).astype(np.float32)
            input_image = np.ascontiguousarray(input_image)

            # Transfer
            input_name = None
            output_name = None
            
            for name, memory in self.allocations.items():
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    input_name = name
                    np.copyto(memory.host, input_image.ravel())
                    cuda.memcpy_htod_async(memory.device, memory.host, self.stream)
                else:
                    output_name = name

            # Set Addresses
            for name, memory in self.allocations.items():
                self.context.set_tensor_address(name, int(memory.device))

            # Execute
            self.context.execute_async_v3(stream_handle=self.stream.handle)

            # Retrieve
            output_mem = self.allocations[output_name]
            cuda.memcpy_dtoh_async(output_mem.host, output_mem.device, self.stream)
            self.stream.synchronize()

            # Reshape Output
            # Assuming output is also NHWC based on input being NHWC
            output_data = np.array(output_mem.host, dtype=np.float32).reshape(
                (1, OUTPUT_H, OUTPUT_W, OUTPUT_C)
            )

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            print(f"Inference time: {elapsed_time}")
            return output_data, elapsed_time
            
        finally:

            self.cuda_ctx.pop()

    def postprocess_mask(self, output_data, original_shape):
        scores = output_data[0]
        class_indices = np.argmax(scores, axis=2)
        
        color_map = np.array([
            [70, 70, 70], [255, 0, 0], [0, 255, 0], [0, 0, 255]
        ], dtype=np.uint8)

        mask = color_map[class_indices]
        resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return resized_mask

# --- ZED SETUP ---
def setup_zed_camera(opt, max_retries=100, retry_delay=0.5):
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080,
                             depth_mode=sl.DEPTH_MODE.NONE,
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
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] Failed to open ZED Camera.")
        sys.exit(1)
    return zed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_file', type=str, default="weights/lane_segmentation.engine")
    parser.add_argument('--ip_address', type=str, default='127.0.0.1:30000')
    parser.add_argument('--debug_mode', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.engine_file):
        print(f"Error: Engine file not found at {args.engine_file}")
        sys.exit(1)

    model = LaneSegmentationEngine(args.engine_file)
    zed = setup_zed_camera(args)
    
    runtime_parameters = sl.RuntimeParameters()
    left_image = sl.Mat()

    window_name = 'Old'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set the window to fullscreen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Optionally resize the window to 4K resolution (3840x2160)
    cv2.resizeWindow(window_name, 1920//2, 1080//2)

    average_infer_time = 0

    print("Starting Inference...")
    try:
        for step in count():
            # ZED Grab might switch context
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                frame = left_image.get_data()
                
                if frame.shape[2] == 4:
                    ip_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    ip_frame = frame

                # Infer (automatically handles context switch now)
                output_data, infer_time = model.infer(ip_frame)
                if step != 0:
                    average_infer_time = float(average_infer_time * (step-1) + infer_time) / float(step)
                    print(f"Average Inference time: {average_infer_time}")

                if args.debug_mode:
                    mask = model.postprocess_mask(output_data, ip_frame.shape)
                    overlay = cv2.addWeighted(ip_frame, 0.7, mask, 0.3, 0)
                    cv2.imshow(window_name, overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print(f"Inference Max Score: {np.max(output_data)}")
    except KeyboardInterrupt:
        print("Stopping...")

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
