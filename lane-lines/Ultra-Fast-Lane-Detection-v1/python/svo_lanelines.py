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
import json
import time
from collections import deque

# TensorRT Logger
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

SOLID = 0
DASHED = 1
WHITE = 0
YELLOW = 1

# Model configurations
MODEL_CONFIG = {
    "culane": {
        "input": (800, 288),
        "row_anchor": [
            121, 131, 141, 150, 160, 170, 180, 189, 199,
            209, 219, 228, 238, 248, 258, 267, 277, 287
        ],
        "griding_num": 200,
        "num_per_laneline": 18,
        "output": (201, 18, 4)
    },
}

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

#Best way to normalize???? nope!
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(im):
    im = im.astype(np.float32) / 255.0 
    im = im.transpose(2, 0, 1)  
    im -= np.array(mean, dtype=np.float32).reshape(3, 1, 1) 
    im /= np.array(std, dtype=np.float32).reshape(3, 1, 1)
    return np.expand_dims(im, axis=0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    return parser.parse_args()


def setup_zed_camera(opt):
    """ Some of these are overwritten when using svo2 files, 
        eg. setting resolution to 1080 if you recorded at 1200 will not crop the input """
    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                             coordinate_units=sl.UNIT.METER,
                             camera_fps = 60, # switch to 60 for live
                             depth_minimum_distance = 1.0, # in coordinate_units selected
                             depth_maximum_distance = 40, # in coordinate_units selected
                             depth_stabilization = 0,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD)

    # Set input SVO file and resolution
    init.set_from_svo_file(opt.input_svo_file)
    init.svo_real_time_mode = True # Set to True try to emulate realtime camera framerate during recording from svo2 timestamps
    init.camera_resolution = sl.RESOLUTION.HD1080

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED Camera")
        zed.close()
        sys.exit(1)
    return zed

def fast_softmax(x, axis=0):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def get_line_type(line_type, line_color):
    LINE_TYPE_MAPPING = {
        (SOLID,   WHITE): 0,  # SOLID WHITE
        (DASHED,  WHITE): 1,  # DASHED WHITE
        (SOLID,  YELLOW): 2,  # SOLID YELLOW
        (DASHED, YELLOW): 3,  # DASHED YELLOW
    }
    return LINE_TYPE_MAPPING.get((line_type, line_color), None)  # Default to None if invalid

def main():
    args = parse_args()

    cap_frame_height, cap_frame_width = (1080,1920)
    rect_size = [200, 1000] # height, width of ROI
    v_offset = 300 # Offset from the center of the frame to bottom
    """ Change h_offset if you move ZED X CENTER """
    h_offset = 50 # Offset from the center of the frame to left 
    roi_x1, roi_y1 = (int((cap_frame_width - rect_size[1]) / 2)-h_offset, int((cap_frame_height - rect_size[0]) / 2)+v_offset)
    roi_x2, roi_y2 = (int((cap_frame_width + rect_size[1]) / 2)-h_offset, int((cap_frame_height + rect_size[0]) / 2)+v_offset)
    half_patch = 24  # Half patch size for centering i.e. 48x48 patch
    
    SOLID_DASHED_THRES = 195
    YELLOW_HUE_RANGE = [20, 40]
    laneline_1_type_sum = 0
    laneline_2_type_sum = 0
    laneline_1_color_sum = 0
    laneline_2_color_sum = 0
    rolling_frame_count = 0

    # Load TensorRT model
    engine = get_engine("weights/culane_res18.engine")
    context = engine.create_execution_context()
    config = MODEL_CONFIG["culane"]

    # Initialize ZED camera from SVO
    zed = setup_zed_camera(args)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, 1)
    runtime_parameters = sl.RuntimeParameters()
    depth_map = sl.Mat()
    left_image = sl.Mat()

    nb_frames = zed.get_svo_number_of_frames()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    fps_history = deque(maxlen=300)

    lanelines_data = {
        f"{laneline_id}": {
            "line_points": [],        # List of X, Y, Z coordinates and confidence
            "line_type": 9,   # SOLID WHITE "0"/ DASHED  WHITE "1"/ SOLID YELLOW "2" / DASHED YELLOW "3" / None "9"
        }
        for laneline_id in (range(4))  # 4 lanelines (0,1,2,3)
    }

    for svo_position in range(nb_frames):
        start_time = time.time()
        
        # Clear laneline data before processing a new frame
        for laneline_id in range(4): 
            lanelines_data[str(laneline_id)]["line_points"].clear()

        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            image_start_time = time.time()
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            image_end_time = time.time()
            depth_start_time = time.time()
            zed.retrieve_measure(depth_map, sl.MEASURE.XYZ)
            depth_end_time = time.time()
            frame = left_image.get_data()

            cv_start_time = time.time()
            rolling_frame_count += 1
            im = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            image_crop_height = 1080 #Fix to 1080 for now as depth map will be computed at 1080
            if cap_frame_height > image_crop_height:
                resized_im = im[-image_crop_height:, :, :]
                resized_im = cv2.resize(resized_im, config["input"])
            else: 
                resized_im = cv2.resize(im, config["input"])
                image_crop_height = cap_frame_height
            normalized_im = normalize(resized_im)

            # Inference
            inference_start = time.time()
            np.copyto(inputs[0].host, normalized_im.ravel())
            common_runtime.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            output_data = np.array(outputs[0].host).reshape(config["output"])
            inference_end = time.time()

            # Post-process laneline detection results
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
            laneline_confidences = np.max(prob, axis=0)

            laneline1_hue_values = []
            # laneline2_hue_values = []
            laneline1_val_values = []
            laneline2_val_values = []

            #debug_lanelines_data = []

            max_lanelines = output.shape[1]
            for laneline_num in range(max_lanelines):
                #debug_laneline_points_with_depth = []
                for point_num in range(output.shape[0]):
                    if output[point_num, laneline_num] > 0:
                        x = int(output[point_num, laneline_num] * col_sample_w * cap_frame_width / config["input"][0]) - 1
                        y = int((image_crop_height * (config["row_anchor"][config["num_per_laneline"] - 1 - point_num]) / config["input"][1])) - 1 + (cap_frame_height - image_crop_height)
                        confidence = laneline_confidences[point_num, laneline_num]
                        status, (X, Y, Z, _) = depth_map.get_value(x, y)  
                        if status == sl.ERROR_CODE.SUCCESS and math.isfinite(X) and math.isfinite(Y) and math.isfinite(Z):
                            if laneline_num in {3, 4} or (laneline_num in {1, 2} and confidence > 0.1):
                                lanelines_data[str(laneline_num)]["line_points"].append([float(X), float(Y), float(Z), float(confidence)])
                                #debug_laneline_points_with_depth.append(((x, y), confidence, (X, Y, Z)))
                            # Check if the point is inside the rectangle ROI
                        if laneline_num in {1, 2} and roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2 and confidence > 0.3: # emperical
                            # Define the 13x13 region around the laneline point
                            x_start, x_end = max(x - half_patch, 0), min(x + half_patch, im.shape[1])
                            y_start, y_end = max(y - half_patch, 0), min(y + half_patch, im.shape[0])
                            patch = im[y_start:y_end, x_start:x_end]
                            b, g, r = cv2.split(patch)
                            equalized_patch = cv2.merge([
                                cv2.equalizeHist(b),
                                cv2.equalizeHist(g),
                                cv2.equalizeHist(r)
                            ])
                            # avg_color = equalized_patch[half_patch, half_patch]
                            half_patch_size = 3
                            avg_color = equalized_patch[
                                half_patch-half_patch_size : half_patch+half_patch_size+1,
                                half_patch-half_patch_size : half_patch+half_patch_size+1
                            ].mean(axis=(0, 1))
                            eq_pt_hue, _, eq_pt_value = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0, 0]

                            if laneline_num == 1:
                                laneline1_hue_values.append(eq_pt_hue)
                                laneline1_val_values.append(eq_pt_value)
                            elif laneline_num == 2:
                                # laneline2_hue_values.append(eq_pt_hue)
                                laneline2_val_values.append(eq_pt_value)
                
                #debug_lanelines_data.append(#debug_laneline_points_with_depth)
                            
            avg_hue_laneline1 = np.mean(laneline1_hue_values) if laneline1_hue_values else WHITE
            # avg_hue_laneline2 = np.mean(laneline2_hue_values) if laneline2_hue_values else WHITE
            avg_val_laneline1 = np.mean(laneline1_val_values) if laneline1_val_values else 0
            avg_val_laneline2 = np.mean(laneline2_val_values) if laneline2_val_values else 0

            laneline_1_color = YELLOW if YELLOW_HUE_RANGE[0] < avg_hue_laneline1 < YELLOW_HUE_RANGE[1] else WHITE
            # laneline_2_color = YELLOW if YELLOW_HUE_RANGE[0] < avg_hue_laneline2 < YELLOW_HUE_RANGE[1] else WHITE
            laneline_1_type = DASHED if avg_val_laneline1 < SOLID_DASHED_THRES else SOLID
            laneline_2_type = DASHED if avg_val_laneline2 < SOLID_DASHED_THRES else SOLID

            laneline_1_type_sum += laneline_1_type
            laneline_2_type_sum += laneline_2_type
            laneline_1_color_sum += laneline_1_color
            # laneline_2_color_sum += laneline_2_color

            if rolling_frame_count >= 30:
                rolling_frame_count = 0
                laneline_1_type = SOLID if int(laneline_1_type_sum / 10) == SOLID else DASHED
                laneline_2_type = SOLID if int(laneline_2_type_sum / 10) == SOLID else DASHED
                laneline_1_color = WHITE if int(laneline_1_color_sum / 3) == WHITE else YELLOW
                # laneline_2_color = WHITE if int(laneline_2_color_sum / 3) == WHITE else YELLOW
                laneline_2_color = WHITE #Only left lanelines are yellow in the US

                lanelines_data["1"]["line_type"] = get_line_type(laneline_1_type, laneline_1_color)
                lanelines_data["2"]["line_type"] = get_line_type(laneline_2_type, laneline_2_color)
                
                laneline_1_type_sum = 0
                laneline_2_type_sum = 0
                laneline_1_color_sum = 0
                laneline_2_color_sum = 0
            
            if lanelines_data:
                print(f'LANE_DATA: {json.dumps(lanelines_data)}')
                #debug_frame_rectangle = cv2.rectangle(im, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)
                #debug_frame_with_lanelines = draw_laneline_points(im, #debug_lanelines_data)
            post_end = time.time()
            
            #debug_print_lines = print_proctimes(fps_history,svo_position,nb_frames,inference_start,inference_end,post_end,depth_start_time,depth_end_time,image_start_time,image_end_time,start_time,cv_start_time)
            
        if zed.get_svo_position() >= nb_frames - 1:
            print("\nEnd of SVO file reached. Exiting.")
            break

    # Release resources
    zed.close()

def print_proctimes(fps_history, svo_position, nb_frames,
                          inference_start, inference_end, post_end,
                          depth_start_time, depth_end_time, image_start_time,
                          image_end_time, start_time, cv_start_time):
    
    total_time = post_end - start_time
    fps = 1.0 / total_time
    fps_history.append(fps)  # Add to rolling window
    rolling_fps = sum(fps_history) / len(fps_history)

    inference_time = inference_end - inference_start
    post_time = post_end - inference_end
    depth_processing_time = depth_end_time - depth_start_time
    img_processing_time = image_end_time - image_start_time
    img_grab_time = image_start_time - start_time
    cv_proc_time = inference_start - cv_start_time 

    print(f"Frame {svo_position+1}/{nb_frames} | Rolling FPS: {rolling_fps:.2f} | Current FPS: {fps:.2f}")
    print(f" - Image-Grab Time: {img_grab_time:.4f}s")
    print(f" - Image-Retrieve Time: {img_processing_time:.4f}s")
    print(f" - Depth-processing Time: {depth_processing_time:.4f}s")
    print(f" - CV Processing Time: {cv_proc_time:.4f}s")
    print(f" - Inference Time: {inference_time:.4f}s")
    print(f" - Post-processing Time: {post_time:.4f}s")
    print(f" - Total Time: {total_time:.4f}s")

def draw_laneline_points(frame, lanelines_data):
    """
    Draws laneline detection points on the image.
    
    :param frame: The original image (NumPy array).
    :param lanelines_data: List of detected laneline points with depth information.
    """

    for laneline_num, laneline_points in enumerate(lanelines_data):
        for point in laneline_points:
            (x, y), confidence, depth = point  # Extract x and y coordinates
            color = (0, 255, 0)  # Green color for laneline points
            cv2.circle(frame, (x, y), radius=3, color=color, thickness=-1)

    # Display the image
    cv2.imshow("Lane Detection", frame)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
