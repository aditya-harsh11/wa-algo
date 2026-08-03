import cv2
import numpy as np
import onnxruntime as ort
import time

# Model input/output dimensions
INPUT_H = 512
INPUT_W = 896
INPUT_C = 3
OUTPUT_H = 512
OUTPUT_W = 896
OUTPUT_C = 4  # Assuming 4 segmentation classes (BG, Road, Curbs, Marks)

def preprocess_image(image):
    image = cv2.resize(image, (INPUT_W, INPUT_H))  # Resize to (512, 896)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension, output shape (1, 512, 896, 3)
    return np.ascontiguousarray(image)

def postprocess_mask(output_data, original_shape):
    print("Postprocessing Segmentation Mask...")
    
    # output_data shape: (1, 4, 512, 896)
    scores = output_data[0]
    
    # Create an empty output mask
    H, W, _ = scores.shape
    blended_mask = np.zeros((H, W, 3), dtype=np.float32)
    
    # Compute the mask
    blended_mask[..., 0] = scores[..., 0] * 70 + scores[..., 1] * 255  # Blue channel
    blended_mask[..., 1] = scores[..., 0] * 70 + scores[..., 2] * 255  # Green channel
    blended_mask[..., 2] = scores[..., 0] * 70 + scores[..., 3] * 255  # Red channel

    # Clip values and convert to uint8
    blended_mask = np.clip(blended_mask, 0, 255).astype(np.uint8)
    cv2.imwrite("raw_mask.jpg", blended_mask)
    print("Raw mask saved as raw_mask.jpg")
    
    # Resize to match original image dimensions
    resized_mask = cv2.resize(blended_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized_mask

if __name__ == "__main__":
    model_path = "weights/road-segmentation-adas-0001.onnx"
    image_path = "dash.jpg"
    output_path = "onnx_output.jpg"
    overlay_path = "onnx_overlay_output.jpg"

    # Load ONNX model
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        exit(1)
    
    input_image = preprocess_image(image)
    
    # Run inference
    inference_start = time.time()
    output_data = session.run([output_name], {input_name: input_image})[0]
    print(f"Raw model output shape: {output_data.shape}")  # Debugging

    inference_end = time.time()
    print(f"Inference Time: {inference_end - inference_start:.4f} sec")
    
    # Postprocess the output
    output_mask = postprocess_mask(output_data, image.shape)
    cv2.imwrite(output_path, output_mask)
    print(f"Segmentation saved to: {output_path}")
    
    # Blend segmentation mask with input image
    blended_image = cv2.addWeighted(image, 0.7, output_mask, 0.3, 0)
    cv2.imwrite(overlay_path, blended_image)
    print(f"Overlayed segmentation saved to: {overlay_path}")
