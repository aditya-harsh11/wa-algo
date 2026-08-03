import argparse
import os
import time

import cv2
import numpy as np
import tensorrt as trt

import sys
sys.path.append(os.path.dirname(os.path.abspath("/usr/src/tensorrt/samples/python/common_runtime.py")))

import common
import common_runtime

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT Ultra-Fast-Lane-Detection example."

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

MODEL_CONFIG = {
    "tusimple": {
        "input": (800, 288),
        "row_anchor": [
            64,  68,  72,  76,  80,  84,  88,  92,  96,
            100, 104, 108, 112, 116, 120, 124, 128, 132,
            136, 140, 144, 148, 152, 156, 160, 164, 168,
            172, 176, 180, 184, 188, 192, 196, 200, 204,
            208, 212, 216, 220, 224, 228, 232, 236, 240,
            244, 248, 252, 256, 260, 264, 268, 272, 276,
            280, 284
        ],
        "griding_num": 100,
        "num_per_lane": 56,
        "output": (101, 56, 4)
    },
    "culane": {
        "input": (800, 288),
        "row_anchor": [
            121, 131, 141, 150, 160, 170, 180, 189, 199,
            209, 219, 228, 238, 248, 258, 267, 277, 287
        ],
        "griding_num": 200,
        "num_per_lane": 18,
        "output": (201, 18, 4)
    },
}

def draw_circle(image, point, color, confidence):
    confidence = max(0, min(confidence, 1))
    if confidence < 0.2:
        scaled_color = (255, 0, 0)
    else:
        scaled_color = (int(color[0] * confidence), int(color[1] * confidence), int(color[2] * confidence))
    cv2.circle(image, point, 7, scaled_color, -1)


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

def draw_caption(image, box, caption, thick_mul=1):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2*thick_mul)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1*thick_mul)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of trt model.", required=True)
    parser.add_argument(
        "--model_config",
        type=str,
        default="culane",
        help='The name of the model. Either "tusimple" or "culane".',
    )
    parser.add_argument("--videopath", help="File path of input video file.", default=None, type=str)
    parser.add_argument("--output", help="File path of output image.", type=str)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Video capture.
    if args.videopath == "":
        print("open camera.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        print("open video file", args.videopath)
        cap = cv2.VideoCapture(args.videopath)


    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Input Video (height, width, fps): ", h, w, fps)

    # Load model.
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    engine = get_engine(args.model)
    context = engine.create_execution_context()
    config = MODEL_CONFIG[args.model_config]

    # Output Video file
    # Define the codec and create VideoWriter object
    video_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    elapsed_list = []
    
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)


    # rect_size = [200, 1000]  # height, width of ROI
    rect_size = [200, 1000] # height, width of ROI
    v_offset = 300 # Offset from the center of the frame to bottom
    h_offset = 50 # Offset from the center of the frame to left
    
    rolling_frame_count = 0
    SOLID = 0
    DASHED = 1
    WHITE = 0
    YELLOW = 1
    solid_dashed_threshold = 195
    yellow_lane_hue_range = [20, 40]
    lane_1_type_sum = 0
    lane_2_type_sum = 0
    lane_1_color_sum = 0
    lane_2_color_sum = 0

    lane1_str = ""
    lane2_str = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        rolling_frame_count += 1
        top_left     = (int((frame.shape[1] - rect_size[1]) / 2)-h_offset, int((frame.shape[0] - rect_size[0]) / 2)+v_offset)
        bottom_right = (int((frame.shape[1] + rect_size[1]) / 2)-h_offset, int((frame.shape[0] + rect_size[0]) / 2)+v_offset)

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resized_im = cv2.resize(im, config["input"])
        height, width, _ = im.shape
        image_crop_height = 1200 #lower to study the affect of distortion due to resizing
        if height > image_crop_height:
            resized_im = im[-image_crop_height:, :, :]
            resized_im = cv2.resize(resized_im, config["input"])
        else: 
            resized_im = cv2.resize(im, config["input"])
            image_crop_height = height
        normalized_im = normalize(resized_im)
        cv2.imshow("temp", resized_im)

        # inference.
        start = time.perf_counter()
        
        np.copyto(inputs[0].host, normalized_im.ravel())
        common_runtime.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        output_data = np.array(outputs[0].host).reshape(config["output"])

        inference_time = (time.perf_counter() - start) * 1000

        # post process.
        output = output_data
        output = output.reshape(config["output"])
        output = output[:, ::-1, :]
        prob = fast_softmax(output[:-1, :, :], axis=0)
        idx = np.arange(config["griding_num"]) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        output = np.argmax(output, axis=0)
        loc[output == config["griding_num"]] = 0
        output = loc

        col_sample = np.linspace(0, 800 - 1, config["griding_num"])
        col_sample_w = col_sample[1] - col_sample[0]
        lane_confidences = np.max(prob, axis=0)

        lanes_points = []
        lanes_confidence_scores = []
        lanes_detected = []

        max_lanes = output.shape[1]
        for lane_num in range(max_lanes):
            lane_points = []
            lane_confidences_list = []

            # Process each of the points for each lane.
            for point_num in range(output.shape[0]):
                if output[point_num, lane_num] > 0:

                    x = int(output[point_num, lane_num] * col_sample_w * w / config["input"][0]) - 1
                    y = int((image_crop_height * (config["row_anchor"][config["num_per_lane"] - 1 - point_num]) / config["input"][1])) - 1 + (h - image_crop_height)
                    confidence = lane_confidences[point_num, lane_num]
                    lane_point = [x, y, confidence]
                    lane_points.append(lane_point)
                    lane_confidences_list.append(confidence)

            lanes_points.append(lane_points)

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)
        
        # Draw white rectangle
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 1)

        x1, y1 = top_left
        x2, y2 = bottom_right
    
        patch_size = 48 
        half_patch = patch_size // 2  # Half patch size for centering

        lane1_hue_values = []
        # lane2_hue_values = []
        lane1_val_values = []
        lane2_val_values = []

        for lane_num in [1, 2]:  # Only process lanes 1 and 2
            for lane_point in lanes_points[lane_num]:
                px, py, confidence = lane_point  # Get point coordinates and confidence

                # Check if the point is inside the white rectangle ROI
                if x1 <= px <= x2 and y1 <= py <= y2 and confidence > 0.3:
                    # Define the 13x13 region around the lane point
                    x_start, x_end = max(px - half_patch, 0), min(px + half_patch, frame.shape[1])
                    y_start, y_end = max(py - half_patch, 0), min(py + half_patch, frame.shape[0])

                    # Extract the patch
                    patch = frame[y_start:y_end, x_start:x_end]

                    # Split into R, G, B channels
                    b, g, r = cv2.split(patch)
                    b_eq = cv2.equalizeHist(b)
                    g_eq = cv2.equalizeHist(g)
                    r_eq = cv2.equalizeHist(r)
                    equalized_patch = cv2.merge([b_eq, g_eq, r_eq])
                    eq_pt_color = equalized_patch[half_patch, half_patch]
                    frame[y_start:y_end, x_start:x_end] = eq_pt_color
                    eq_pt_color_hsv = cv2.cvtColor(np.uint8([[eq_pt_color]]), cv2.COLOR_BGR2HSV)

                    eq_pt_hue = eq_pt_color_hsv[0, 0, 0]  # Extract hue value
                    eq_pt_value = eq_pt_color_hsv[0, 0, 2]  # Extract Value value

                    if lane_num == 1:
                        lane1_hue_values.append(eq_pt_hue)
                        lane1_val_values.append(eq_pt_value)
                    elif lane_num == 2:
                        # lane2_hue_values.append(eq_pt_hue)
                        lane2_val_values.append(eq_pt_value)
                        
        avg_hue_lane1 = np.mean(lane1_hue_values) if lane1_hue_values else 0
        # avg_hue_lane2 = np.mean(lane2_hue_values) if lane2_hue_values else 0
        avg_val_lane1 = np.mean(lane1_val_values) if lane1_val_values else 0
        avg_val_lane2 = np.mean(lane2_val_values) if lane2_val_values else 0

        if avg_hue_lane1 < yellow_lane_hue_range[1] and avg_hue_lane1 > yellow_lane_hue_range[0]:
            lane_1_color = YELLOW
        else:
            lane_1_color = WHITE
        
        # if avg_hue_lane2 < yellow_lane_hue_range[1] and avg_hue_lane2 > yellow_lane_hue_range[0]:
        #     lane_2_color = YELLOW
        # else:
        #     lane_2_color = WHITE

        if avg_val_lane1 < solid_dashed_threshold:
            lane_1_type = DASHED
        else:
            lane_1_type = SOLID
        
        if avg_val_lane2 < solid_dashed_threshold:
            lane_2_type = DASHED
        else:
            lane_2_type = SOLID

        lane_1_type_sum += lane_1_type
        lane_2_type_sum += lane_2_type
        lane_1_color_sum += lane_1_color
        # lane_2_color_sum += lane_2_color

        if rolling_frame_count == 30:
            rolling_frame_count = 0
            lane_1_type = int(lane_1_type_sum / 10)
            lane_2_type = int(lane_2_type_sum / 10)
            lane_1_color = int(lane_1_color_sum / 3) #lane_1_color 
            # lane_2_color = int(lane_2_color_sum / 10)

            if lane_1_type == SOLID:
                lane_1_type = "Solid"
            else:
                lane_1_type = "Dashed"

            if lane_2_type == SOLID:
                lane_2_type = "Solid"
            else:   
                lane_2_type = "Dashed"

            if lane_1_color == WHITE:
                lane_1_color = "White"
            else:
                lane_1_color = "Yellow"
            
            # if lane_2_color == WHITE:
            #     lane_2_color = "White"
            # else:
            #     lane_2_color = "Yellow"
            
            lane_2_color = "White" #Only left lanes can be yellow in the US

            lane1_type_str = f" L1 : {lane_1_type}"
            lane2_type_str = f" L2 : {lane_2_type}"
            lane1_color_str = f" {lane_1_color}"
            lane2_color_str = f" {lane_2_color}"

            lane1_str = lane1_type_str + lane1_color_str
            lane2_str = lane2_type_str + lane2_color_str
            
            lane_1_type_sum = 0
            lane_2_type_sum = 0
            lane_1_color_sum = 0        

        # Draw lanes
        for lane_num, lane_points in enumerate(lanes_points):
            if lane_num == 1 or lane_num == 2:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            for lane_point in lane_points:
                draw_circle(frame, (lane_point[0], lane_point[1]), color, lane_point[2]) #lane_point[2] is confidence score

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + " " + fps_text + avg_text #+ lane1_str + lane2_str
        lane_type_text = lane1_str + "       " + lane2_str
        draw_caption(frame, (10, 30), display_text)
        draw_caption(frame, (int((frame.shape[1]) / 2)-350, int((frame.shape[0]) / 2)+v_offset), lane_type_text, 2)


        # Output video file
        if video_writer is not None:
            video_writer.write(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()