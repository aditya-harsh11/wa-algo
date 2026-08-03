import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import os
import sys
import time
from scipy.special import softmax
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config


class UFLDv2:
    def __init__(self, engine_path, config_path, ori_size):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []

        num_tensors = self.engine.num_io_tensors

        for i in range(num_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            dtype = self.engine.get_tensor_dtype(tensor_name)
            shape = tuple(self.engine.get_tensor_shape(tensor_name))  

            if is_input:
                self.batch_size = shape[0]

            size = int(np.prod(shape)) * np.dtype(trt.nptype(dtype)).itemsize  
            allocation = cuda.mem_alloc(size)

            binding = {
                'index': i,
                'name': tensor_name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': shape,  
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        # Load Config
        cfg = Config.fromfile(config_path)
        self.ori_img_w, self.ori_img_h = ori_size
        self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        self.input_width = cfg.train_width
        self.input_height = cfg.train_height
        self.num_row = cfg.num_row
        self.num_col = cfg.num_col
        self.row_anchor = np.linspace(0.42, 1, self.num_row)
        self.col_anchor = np.linspace(0, 1, self.num_col)

    def forward(self, img):
        """Runs inference and visualizes lane detection."""
        im0 = img.copy()

        img = img[self.cut_height:, :, :]
        # print(im0.shape)
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)

        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        img = np.ascontiguousarray(img)  

        cuda.memcpy_htod(self.inputs[0]['allocation'], img)

        start_time = time.perf_counter()
        self.context.execute_v2(self.allocations)
        inference_time = (time.perf_counter() - start_time) * 1000  

        preds = {out['name']: np.zeros(out['shape'], out['dtype']) for out in self.outputs}
        for out in self.outputs:
            cuda.memcpy_dtoh(preds[out['name']], out['allocation'])

        coords = self.pred2coords(preds)

        for lane in coords:
            for coord in lane:
                cv2.circle(im0, coord, 2, (0, 255, 0), -1)

        cv2.putText(im0, f"Inference: {inference_time:.2f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("result", im0)

    def pred2coords(self, pred):
        """Convert model predictions to lane coordinates."""
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1)
        valid_row = pred['exist_row'].argmax(1)

        max_indices_col = pred['loc_col'].argmax(1)
        valid_col = pred['exist_col'].argmax(1)

        coords = []
        for i in range(num_lane_row):
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = np.arange(max(0, max_indices_row[0, k, i] - self.input_width),
                                            min(num_grid_row - 1, max_indices_row[0, k, i] + self.input_width) + 1)
                        out_tmp = (softmax(pred['loc_row'][0, all_ind, k, i]) * all_ind.astype(np.float32)).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w
                        tmp.append((int(out_tmp), int(self.row_anchor[k] * self.ori_img_h)))
                coords.append(tmp)
        return coords


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/curvelanes_res18.py', type=str, help='Path to config file')
    parser.add_argument('--engine_path', default='weights/curvelanes_res18.engine', type=str, help='Path to engine file')
    parser.add_argument('--video_path', default='example.mp4', type=str, help='Path to video file')
    parser.add_argument('--ori_size', default="(1600, 800)", type=str, help='Size of original frame')
    return parser.parse_args()


def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    if h > 800 and w > 1600:
        startx = w // 2 - (cropx // 2)
        starty = h // 2 - (cropy // 2)
        img = img[starty:starty + cropy, startx:startx + cropx]
    return img


if __name__ == "__main__":
    args = get_args()
    cap = cv2.VideoCapture(args.video_path)
    ori_size = eval(args.ori_size)
    isnet = UFLDv2(args.engine_path, args.config_path, ori_size)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = crop_center(img, 1600, 800)
        isnet.forward(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
