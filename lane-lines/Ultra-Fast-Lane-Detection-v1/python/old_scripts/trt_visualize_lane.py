"""
    TensorRT Ultra-Fast-Lane-Detection example.
    Author: Keshav Sharan
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

###### REVERT TO ORIGINAL FROM MAIN BRANCH BEFORE MERGE ######

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


# def draw_circle(image, point):
#     cv2.circle(image, point, 10, (246, 250, 250), -1)
#     cv2.circle(image, point, 4, (255, 209, 0), 2)

def draw_circle(image, point, color, confidence):
    confidence = max(0, min(confidence, 1))
    if confidence < 0.2:
        scaled_color = (255, 0, 0)
    else:
        scaled_color = (int(color[0] * confidence), int(color[1] * confidence), int(color[2] * confidence))
    cv2.circle(image, point, 7, scaled_color, -1)

def fast_softmax(x, axis=0):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

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


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of trt model.", required=True)
    parser.add_argument(
        "--model_config",
        type=str,
        default="tusimple",
        help='The name of the model. Either "tusimple" or "culane".',
    )
    parser.add_argument(
        "--videopath", help="File path of input video file.", default=None, type=str
    )
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_im = cv2.resize(im, config["input"])
        # resized_im = frame[:-288, :800, :]
        normalized_im = normalize(resized_im)

        # inference.
        start = time.perf_counter()
        # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        np.copyto(inputs[0].host, normalized_im.ravel())
        
        # outputs = common_runtime.do_inference(
        #     context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        # )
        common_runtime.do_inference(
            context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        output_data = np.array(outputs[0].host).reshape(config["output"])

        inference_time = (time.perf_counter() - start) * 1000

        # post process.
        # output = outputs[0]
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

            # Check if there are any points detected in the lane.
            if np.sum(output[:, lane_num] != 0) > 2:
                lanes_detected.append(True)

                # Process each of the points for each lane.
                for point_num in range(output.shape[0]):
                    if output[point_num, lane_num] > 0:
                        x = int(output[point_num, lane_num] * col_sample_w * w / config["input"][0]) - 1
                        y = int(h * (config["row_anchor"][config["num_per_lane"] - 1 - point_num] / config["input"][1])) - 1
                        confidence = lane_confidences[point_num, lane_num]
                        lane_point = [x, y, confidence]
                        lane_points.append(lane_point)
                        lane_confidences_list.append(confidence)
            else:
                lanes_detected.append(False)

            lanes_points.append(lane_points)
            lanes_confidence_scores.append(lane_confidences_list)

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)
        
        rect_size = [1900, 200]  # Width, Height of ROI
        v_offset = 300 # Offset from the center of the frame to bottom
        h_offset = 50 # Offset from the center of the frame to left
        # Compute top-left and bottom-right corners
        top_left = (int((1920 - rect_size[0]) / 2)-h_offset, int((1200 - rect_size[1]) / 2)+v_offset)
        bottom_right = (int((1920 + rect_size[0]) / 2)-h_offset, int((1200 + rect_size[1]) / 2)+v_offset)
        
        # Draw white rectangle
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)

        x1, y1 = top_left
        x2, y2 = bottom_right


        """
        roi1 = frame[y1:y2, x1:x1 + small_roi_width]
        roi2 = frame[y1:y2, x2 - small_roi_width:x2]

        gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized_roi1 = cv2.equalizeHist(gray_roi1)
        equalized_roi2 = cv2.equalizeHist(gray_roi2)        

        # Convert equalized grayscale ROI back to BGR (to match original frame format)
        equalized_roi1_bgr = cv2.cvtColor(equalized_roi1, cv2.COLOR_GRAY2BGR)
        equalized_roi2_bgr = cv2.cvtColor(equalized_roi2, cv2.COLOR_GRAY2BGR)
        """

        # small_roi_width = 250
        # roi1 = frame[y1:y2, x1:x1 + small_roi_width]
        # roi2 = frame[y1:y2, x2 - small_roi_width:x2]
        # # Split into B, G, R channels
        # b1, g1, r1 = cv2.split(roi1)
        # b2, g2, r2 = cv2.split(roi2)

        # # Apply histogram equalization to each channel separately
        # b1_eq = cv2.equalizeHist(b1)
        # g1_eq = cv2.equalizeHist(g1)
        # r1_eq = cv2.equalizeHist(r1)

        # b2_eq = cv2.equalizeHist(b2)
        # g2_eq = cv2.equalizeHist(g2)
        # r2_eq = cv2.equalizeHist(r2)

        # # Merge back equalized channels
        # equalized_roi1_bgr = cv2.merge([b1_eq, g1_eq, r1_eq])
        # equalized_roi2_bgr = cv2.merge([b2_eq, g2_eq, r2_eq])

        # # Replace the processed ROI back into the frame
        # frame[y1:y2, x1:x1 + small_roi_width] = equalized_roi1_bgr
        # frame[y1:y2, x2 - small_roi_width:x2] = equalized_roi2_bgr
        

        patch_size = 49  # 13x13 patch
        half_patch = patch_size // 2  # Half patch size for centering

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

                    # Check if the patch size is valid (should be 13x13 or smaller if near edges)
                    if patch.shape[0] > 1 and patch.shape[1] > 1:
                        # Split into R, G, B channels
                        b, g, r = cv2.split(patch)
                        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                        # Apply histogram equalization to each channel separately
                        b_eq = cv2.equalizeHist(b)
                        g_eq = cv2.equalizeHist(g)
                        r_eq = cv2.equalizeHist(r)
                        gray_eq = cv2.equalizeHist(gray)

                        # Merge back equalized channels
                        # equalized_patch = cv2.merge([b_eq, g_eq, r_eq])
                        equalized_patch = cv2.merge([gray_eq, gray_eq, gray_eq])
                        eq_pt_color = equalized_patch[half_patch, half_patch]
                        eq_pt_color[0] = int(eq_pt_color[0]/150) * 255
                        eq_pt_color = [eq_pt_color[0],eq_pt_color[0],eq_pt_color[0]]
                        frame[y_start:y_end, x_start:x_end] = eq_pt_color

                        # Replace the processed patch back into the frame
                        # frame[y_start:y_end, x_start:x_end] = equalized_patch


        # Draw lanes
        for lane_num, lane_points in enumerate(lanes_points):
            if lane_num == 1 or lane_num == 2:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            for lane_point in lane_points:
                # draw_alpha_circle(frame, (lane_point[0], lane_point[1]), color, lane_point[2])
                draw_circle(frame, (lane_point[0], lane_point[1]), color, lane_point[2]) #lane_point[2] is confidence score

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + " " + fps_text + avg_text
        draw_caption(frame, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()