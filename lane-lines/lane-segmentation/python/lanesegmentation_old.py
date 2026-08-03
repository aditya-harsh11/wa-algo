#import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pycuda.autoinit
import argparse
import os
import cv2
import numpy as np
import tensorrt as trt
import pyzed.sl as sl
import sys
sys.path.append(os.path.dirname(os.path.abspath("/usr/src/tensorrt/samples/python/common_runtime.py")))

import common
import common_runtime
import time
from collections import deque

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# Input/Output dimensions
INPUT_H = 512
INPUT_W = 896
INPUT_C = 3
OUTPUT_H = 512
OUTPUT_W = 896
OUTPUT_C = 4  # Assuming 4 segmentation classes (BG, Road, Curbs, Marks)

# Normalization Parameters
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def normalize(im):
    # im = im.astype(np.float32) / 255.0
    im = im.transpose(2, 0, 1)  # Convert HWC to CHW
    im -= np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    im /= np.array(std, dtype=np.float32).reshape(3, 1, 1)
    return np.expand_dims(im, axis=0)  # Add batch dimension

def get_engine(engine_path):
    print(f"Reading engine from file: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

class LaneSegmentationEngine:
    def __init__(self, engine_path):
        self.engine = get_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Allocate memory buffers for input and output
        #self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

        # 1. Unpack only the 3 values the function actually returns
        self.inputs, self.outputs, self.bindings = common.allocate_buffers(self.engine)

        # 2. Manually create the CUDA stream (since common.py didn't give us one)
        import pycuda.driver as cuda
        self.stream = cuda.Stream()

    def infer(self, image):
        input_image = cv2.resize(image, (INPUT_W, INPUT_H))
        # input_image = normalize(input_image)

        # Copy input data to GPU
        np.copyto(self.inputs[0].host, input_image.ravel())

        # Perform inference
        inference_start = time.time()
        common_runtime.do_inference(self.context, engine=self.engine, bindings=self.bindings,
                                    inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        inference_end = time.time()
        print(f"Inference Time: {inference_end - inference_start:.4f} sec")

        # Retrieve output
        output_data = np.array(self.outputs[0].host, dtype=np.float32).reshape(
            (1, OUTPUT_H, OUTPUT_W, OUTPUT_C)
        )
        return output_data

    def postprocess_mask(self, output_data, original_shape):
        print("Postprocessing Segmentation Mask...")
        scores = output_data[0]  # Shape: (512, 896, 4)

        # Assign class with highest score for each pixel
        class_indices = np.argmax(scores, axis=2)

        # Define color map (in BGR)
        color_map = np.array([
            [70, 70, 70],   # Class 0: Background (Gray)
            [255, 0, 0],    # Class 1: Road (Blue)
            [0, 255, 0],    # Class 2: Curbs (Green)
            [0, 0, 255]     # Class 3: Lane Marks (Red)
        ], dtype=np.uint8)

        mask = color_map[class_indices]  # Shape: (512, 896, 3)

        # Resize mask to match original frame dimensions
        resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return resized_mask

def setup_zed_camera(opt, max_retries=100, retry_delay=0.5):
    init = sl.InitParameters(camera_resolution = sl.RESOLUTION.HD1080,
                             depth_mode=sl.DEPTH_MODE.NEURAL,
                             coordinate_units=sl.UNIT.METER,
                             camera_fps = 60.0, # switch to 60 for live
                             depth_minimum_distance = 1.0, # in coordinate_units selected
                             depth_maximum_distance = 40, # in coordinate_units selected
                             depth_stabilization = 0,
                             async_grab_camera_recovery = True,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD)

    if len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")

    zed = sl.Camera()
    retry_count = 0

    # while True:
    while retry_count < max_retries:
        if zed.open(init) == sl.ERROR_CODE.SUCCESS:
            zed.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, 1)
            print("[INFO] ZED Camera successfully opened!")
            return zed
        else:
            print(f"[WARNING] Failed to open ZED Camera. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_count += 1
    print("[ERROR] Unable to establish connection with ZED Camera after multiple attempts.")
    zed.close()
    sys.exit(1)
    return zed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_file', type=str, help="Path to the TensorRT .engine file",  default="weights/lane_segmentation.engine")
    parser.add_argument('--ip_address',  type=str, help="IP Adress, in format a.b.c.d:port", default = '127.0.0.1:30000')
    parser.add_argument('--debug_mode',  type=str, help="Set to 1 to turn on visualization", default = "0")
    args = parser.parse_args()

    # Load TensorRT model
    model = LaneSegmentationEngine(args.engine_file)

    # Initialize ZED camera from SVO
    zed = setup_zed_camera(args)
    runtime_parameters = sl.RuntimeParameters()
    left_image = sl.Mat()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            frame = left_image.get_data()

            # Convert to RGB
            ip_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Run inference
            output_data = model.infer(ip_frame)

            if args.debug_mode:

                # Post-process segmentation mask
                mask = model.postprocess_mask(output_data, ip_frame.shape)

                # Blend the original frame and the segmentation mask
                overlay = cv2.addWeighted(ip_frame, 0.7, mask, 0.3, 0)

                # Display the overlay frame
                cv2.imshow("Lane Segmentation Overlay", overlay)
                cv2.waitKey(1)
            
            else :
                print(output_data)


    # Release resources
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
