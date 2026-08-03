import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# Input/Output dimensions
INPUT_H = 512
INPUT_W = 896
INPUT_C = 3
OUTPUT_H = 512
OUTPUT_W = 896
OUTPUT_C = 4  # Assuming 4 segmentation classes (BG, Road, Curbs, Marks)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

def get_engine(engine_path):
    print(f"Reading engine from file: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

class LaneSegmentationEngine:
    def __init__(self, engine_path):
        self.engine = get_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate memory
        self.inputs, self.outputs, self.bindings = self.allocate_buffers()

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = np.zeros(size, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            is_input = self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
            if is_input:
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))
        return inputs, outputs, bindings

    def preprocess_image(self, image):
        # Resize image to network input size and add batch dimension
        image = cv2.resize(image, (INPUT_W, INPUT_H))
        image = np.expand_dims(image, axis=0)  # (1, 512, 896, 3)
        return np.ascontiguousarray(image)

    def infer(self, image):
        input_image = self.preprocess_image(image)
        np.copyto(self.inputs[0][0], input_image.ravel())

        # Copy input to GPU
        cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)

        # Set tensor addresses properly
        input_tensor_name = self.engine.get_tensor_name(0)
        output_tensor_name = self.engine.get_tensor_name(1)
        self.context.set_tensor_address(input_tensor_name, self.inputs[0][1])
        self.context.set_tensor_address(output_tensor_name, self.outputs[0][1])

        # Run inference
        inference_start = time.time()
        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()
        inference_end = time.time()
        print(f"Inference Time: {inference_end - inference_start:.4f} sec")

        # Retrieve output from GPU
        cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
        self.stream.synchronize()

        # Reshape output to (1, 512, 896, 4)
        output_data = np.array(self.outputs[0][0], dtype=np.float32, copy=True).reshape(
            (1, OUTPUT_H, OUTPUT_W, OUTPUT_C)
        )
        return output_data

    def postprocess_mask(self, output_data, original_shape):
        print("Postprocessing Segmentation Mask...")
        scores = output_data[0]  # Shape: (512, 896, 4)

        # For each pixel, choose the class with the highest score
        class_indices = np.argmax(scores, axis=2)

        # Define a color map (in BGR)
        color_map = np.array([
            [70, 70, 70],   # Class 0: Gray (Background)
            [255, 0, 0],    # Class 1: Blue (Road)
            [0, 255, 0],    # Class 2: Green (Curbs)
            [0, 0, 255]     # Class 3: Red (Marks)
        ], dtype=np.uint8)

        mask = color_map[class_indices]  # (512, 896, 3)

        # Resize mask to match the original frame dimensions
        resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return resized_mask

if __name__ == "__main__":
    engine_path = "weights/lane_segmentation.engine"
    input_video_path = "/home/auc7us/Videos/espringsr.avi"

    # Initialize segmentation model
    model = LaneSegmentationEngine(engine_path)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        exit(1)

    print("Press 'q' to exit the video display.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error encountered.")
            break

        # Run inference on the frame
        output_data = model.infer(frame)
        mask = model.postprocess_mask(output_data, frame.shape)

        # Blend the original frame and the segmentation mask
        overlay = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

        # Display the overlay frame
        cv2.imshow("Lane Segmentation Overlay", overlay)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
