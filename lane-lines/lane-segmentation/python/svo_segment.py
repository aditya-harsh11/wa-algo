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
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

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

def setup_zed_camera(svo_file):
    """ Initialize ZED camera with the SVO file """
    init = sl.InitParameters()
    init.set_from_svo_file(svo_file)
    init.svo_real_time_mode = True
    init.camera_resolution = sl.RESOLUTION.HD1080

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED Camera")
        zed.close()
        exit(1)
    return zed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    parser.add_argument('--engine_file', type=str, default="weights/lane_segmentation.engine",
                        help='Path to the TensorRT engine file')
    args = parser.parse_args()

    # Load TensorRT model
    model = LaneSegmentationEngine(args.engine_file)

    # Initialize ZED camera from SVO
    zed = setup_zed_camera(args.input_svo_file)
    runtime_parameters = sl.RuntimeParameters()
    left_image = sl.Mat()

    nb_frames = zed.get_svo_number_of_frames()

    print("Press 'q' to exit the video display.")

    for frame_id in range(nb_frames):
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            frame = left_image.get_data()

            # Convert to RGB
            ip_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Run inference
            output_data = model.infer(ip_frame)

            # Post-process segmentation mask
            mask = model.postprocess_mask(output_data, ip_frame.shape)

            # Blend the original frame and the segmentation mask
            overlay = cv2.addWeighted(ip_frame, 0.7, mask, 0.3, 0)

            # Display the overlay frame
            cv2.imshow("Lane Segmentation Overlay", overlay)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed.")
                break

    # Release resources
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
