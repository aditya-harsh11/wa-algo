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
import math

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# Mean and std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Model configurations
MODEL_CONFIG = {
    "tusimple": {
        "input": (800, 288),
        "row_anchor": [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132,
                       136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196,
                       200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260,
                       264, 268, 272, 276, 280, 284],
        "griding_num": 100,
        "num_per_lane": 56,
        "output": (101, 56, 4)
    },
    "culane": {
        "input": (800, 288),
        "row_anchor": [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287],
        "griding_num": 200,
        "num_per_lane": 18,
        "output": (201, 18, 4)
    },
}


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def normalize(im):
    im = np.asarray(im, dtype="float32")
    im = (im / 255.0 - mean) / std
    im = im.transpose(2, 0, 1)
    im = np.expand_dims(im, axis=0)
    return im.astype("float32")

def fast_softmax(x, axis=0):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of TensorRT model.", required=True)
    parser.add_argument("--model_config", type=str, default="culane",
                        help='The name of the model. Either "tusimple" or "culane".')
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    parser.add_argument('--resolution', type=str, help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA, VGA', default='HD720')
    return parser.parse_args()


def setup_zed_camera(opt):
    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                             coordinate_units=sl.UNIT.METER,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    # Set input SVO file and resolution
    init.set_from_svo_file(opt.input_svo_file)
    print(f"[INFO] Using SVO file input: {opt.input_svo_file}")

    resolution_dict = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA
    }
    init.camera_resolution = resolution_dict.get(opt.resolution.upper(), sl.RESOLUTION.HD720)
    print(f"[INFO] Camera resolution set to {opt.resolution.upper()}")

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED Camera")
        zed.close()
        sys.exit(1)
    return zed


def main():
    args = parse_args()

    # Load TensorRT model
    engine = get_engine(args.model)
    context = engine.create_execution_context()
    config = MODEL_CONFIG[args.model_config]

    # Initialize ZED camera from SVO
    zed = setup_zed_camera(args)
    runtime_parameters = sl.RuntimeParameters()
    depth_map = sl.Mat()
    left_image = sl.Mat()

    nb_frames = zed.get_svo_number_of_frames()

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    for svo_position in range(nb_frames):
        # Retrieve left image from SVO
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            frame = left_image.get_data()

            # Prepare and normalize image for inference
            im = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            h,w,_ = im.shape
            resized_im = cv2.resize(im, config["input"])
            normalized_im = normalize(resized_im)

            # Inference
            # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            np.copyto(inputs[0].host, normalized_im.ravel())

            common_runtime.do_inference(
                context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
            )

            output_data = np.array(outputs[0].host).reshape(config["output"])

            # Post-process lane detection results
            output = output_data.reshape(config["output"])[:, ::-1, :]
            prob = fast_softmax(output[:-1, :, :], axis=0)
            idx = np.arange(config["griding_num"]) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            output = np.argmax(output, axis=0)
            loc[output == config["griding_num"]] = 0
            output = loc

            col_sample = np.linspace(0, 800 - 1, config["griding_num"])
            col_sample_w = col_sample[1] - col_sample[0]

            max_lanes = output.shape[1]
            for lane_num in range(max_lanes):
                lane_points_with_depth = []
                for point_num in range(output.shape[0]):
                    if output[point_num, lane_num] > 0:
                        x = int(output[point_num, lane_num] * col_sample_w * w / config["input"][0]) - 1
                        y = int(h * (config["row_anchor"][config["num_per_lane"] - 1 - point_num] / config["input"][1])) - 1

                        # Retrieve depth from ZED depth map
                        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                        depth_value = depth_map.get_value(x, y)[1]

                        lane_points_with_depth.append(((x, y), depth_value))

                # Print lane points with depth for each lane ID
                if lane_points_with_depth:
                    print(f"Lane ID {lane_num}: {lane_points_with_depth}")

        print("********NEED TO DEVELOP CPP CODE FOR FASTER INFERENCE********")

        if zed.get_svo_position() >= nb_frames - 1:
            print("\nEnd of SVO file reached. Exiting.")
            break

    # Release resources
    zed.close()


if __name__ == "__main__":
    main()
