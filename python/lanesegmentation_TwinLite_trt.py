import os
import argparse
import sys
import cv2
import numpy as np

# Windows DLL path fix for ZED SDK
if sys.platform == "win32":
    zed_path = "C:/Program Files (x86)/ZED SDK/bin"
    if os.path.exists(zed_path):
        os.add_dll_directory(zed_path)

import pyzed.sl as sl
import time

# TensorRT Imports
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# --- CONFIGURATION ---
INPUT_W = 640
INPUT_H = 384
OUTPUT_W = 640
OUTPUT_H = 384
INPUT_C = 3

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# --- MEMORY HELPER ---
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

# --- MAIN ENGINE CLASS (TensorRT / TwinLiteNet+) ---
class TwinLiteTRTEngine:
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
                self.context.set_input_shape(name, (1, INPUT_C, INPUT_H, INPUT_W))
                
        return allocations

    def infer(self, image):
        self.cuda_ctx.push()
        try:
            img_rs = cv2.resize(image, (INPUT_W, INPUT_H))
            img_ts = img_rs[:, :, ::-1].transpose(2, 0, 1)
            img_ts = np.ascontiguousarray(img_ts)
            
            input_tensor = np.expand_dims(img_ts, axis=0).astype(np.float32) / 255.0
            input_tensor = np.ascontiguousarray(input_tensor)

            for name, memory in self.allocations.items():
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    np.copyto(memory.host, input_tensor.ravel())
                    cuda.memcpy_htod_async(memory.device, memory.host, self.stream)
                self.context.set_tensor_address(name, int(memory.device))

            self.context.execute_async_v3(stream_handle=self.stream.handle)

            da_mem = self.allocations['da_out']
            ll_mem = self.allocations['ll_out']
            
            cuda.memcpy_dtoh_async(da_mem.host, da_mem.device, self.stream)
            cuda.memcpy_dtoh_async(ll_mem.host, ll_mem.device, self.stream)
            self.stream.synchronize()

            da_scores = np.array(da_mem.host, dtype=np.float32).reshape((2, OUTPUT_H, OUTPUT_W)).transpose(1, 2, 0)
            ll_scores = np.array(ll_mem.host, dtype=np.float32).reshape((2, OUTPUT_H, OUTPUT_W)).transpose(1, 2, 0)

            return da_scores, ll_scores
        finally:
            self.cuda_ctx.pop()

# --- BASELINE LOGIC ---
# --- ZED SETUP ---
def setup_zed_camera(opt):
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080,
                             depth_mode=sl.DEPTH_MODE.NEURAL, 
                             coordinate_units=sl.UNIT.METER,
                             camera_fps=30,
                             depth_minimum_distance=1.0,
                             depth_maximum_distance=40,
                             depth_stabilization=1,
                             async_grab_camera_recovery=True,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD)

    if opt.svo:
        init.set_from_svo_file(opt.svo)
        init.svo_real_time_mode = True
        print(f"Reading from SVO file: {opt.svo}")
    elif len(opt.ip_address) > 0:
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

# --- PROFILER HELPER ---
class Profiler:
    def __init__(self):
        self.stats = {}

    def update(self, name, dt_ms):
        if name not in self.stats:
            self.stats[name] = {'current': 0.0, 'avg': 0.0, 'count': 0}
        
        s = self.stats[name]
        s['current'] = dt_ms
        s['count'] += 1
        # Cumulative Moving Average formula
        s['avg'] += (dt_ms - s['avg']) / s['count']

    def print_summary(self):
        # We use \r to overwrite the line in the console for a clean output without spamming
        out_str = " | ".join([f"{k}: {v['current']:.1f}ms (Avg: {v['avg']:.1f}ms)" for k, v in self.stats.items()])
        sys.stdout.write(f"\r{out_str}")
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_file', type=str, default="weights/twinlitenetplus_large.engine")
    parser.add_argument('--svo', type=str, default="", help="Path to .svo or .svo2 video file")
    parser.add_argument('--ip_address', type=str, default="")
    parser.add_argument('--debug_mode', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.engine_file):
        print(f"Error: Engine file not found at {args.engine_file}")
        sys.exit(1)

    model = TwinLiteTRTEngine(args.engine_file)
    zed = setup_zed_camera(args)
    profiler = Profiler()
    
    runtime_parameters = sl.RuntimeParameters()
    left_image = sl.Mat()
    point_cloud = sl.Mat()

    print("Starting TensorRT TwinLiteNet+ Inference...\n")
    try:
        while True:
            # Note: We do NOT time zed.grab() because it artificially blocks to match the camera FPS.
            # Measuring it would just give you 16.6ms (for 60fps) or 33.3ms (for 30fps).
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                
                # --- 1. ZED Retrieval ---
                t0 = time.perf_counter()
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                
                rgb_frame = left_image.get_data()
                if rgb_frame.shape[2] == 4:
                    ip_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGRA2BGR)
                else:
                    ip_frame = rgb_frame
                
                point_cloud_data = point_cloud.get_data()
                profiler.update("1. Retrieve", (time.perf_counter() - t0) * 1000)

                # --- 2. TRT Inference ---
                t1 = time.perf_counter()
                da_scores, ll_scores = model.infer(ip_frame)
                profiler.update("2. TensorRT", (time.perf_counter() - t1) * 1000)
                
                # --- 3. Post-processing & Vis ---
                t3 = time.perf_counter()
                da_predict = np.argmax(da_scores, axis=2).astype(np.uint8)
                ll_predict = np.argmax(ll_scores, axis=2).astype(np.uint8)
                
                # --- ROBUST CONTOUR-BASED STOP BAR DETECTION ---
                stop_bar_row = -1
                stop_bar_center_x = -1
                
                # Convert to binary mask for contour detection
                ll_mask = (ll_predict * 255).astype(np.uint8)
                da_mask = (da_predict * 255).astype(np.uint8)
                
                contours, _ = cv2.findContours(ll_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                max_y = -1
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    width_ratio = float(w) / OUTPUT_W
                    
                    # Filter for horizontal-ish blobs that are wide enough
                    if aspect_ratio > 3.0 and width_ratio > 0.15:
                        mid_x = x + w // 2
                        mid_y = y + h // 2
                        
                        # Validate if it's within the drivable area
                        if da_mask[mid_y, mid_x] > 0:
                            if (y + h) > max_y:
                                max_y = y + h
                                stop_bar_row = mid_y
                                stop_bar_center_x = mid_x
                
                if stop_bar_row != -1:
                    # Get 3D point if possible (point_cloud_data is [H, W, 4] for XYZRGBA or [H, W, 3] for XYZ)
                    # Note: Python implementation currently doesn't output 3D data to console like C++, 
                    # but we can add it for debugging.
                    sb_3d = point_cloud_data[stop_bar_row, stop_bar_center_x]
                    if np.isfinite(sb_3d[2]):
                         pass # Could print or use here
                
                if args.debug_mode:
                    img_rs = cv2.resize(ip_frame, (INPUT_W, INPUT_H))
                    
                    net_mask = np.zeros_like(img_rs)
                    net_mask[da_predict > 0] = [255, 0, 0] 
                    net_mask[ll_predict > 0] = [0, 255, 0] 
                    net_overlay = cv2.addWeighted(img_rs, 0.7, net_mask, 0.3, 0)
                    
                    if stop_bar_row != -1:
                        cv2.line(net_overlay, (0, stop_bar_row), (OUTPUT_W, stop_bar_row), (0, 0, 255), 3)
                        cv2.circle(net_overlay, (stop_bar_center_x, stop_bar_row), 5, (255, 255, 255), -1)

                    cv2.imshow("TwinLiteNet+ TensorRT", net_overlay)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                profiler.update("4. PostProc", (time.perf_counter() - t3) * 1000)

                # Print the live stats line
                profiler.print_summary()
                        
    except KeyboardInterrupt:
        print("\nStopping...")

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()