import sys
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

def main():
    # --- 1. CONFIGURATION ---
    # We use the 'seg' model. 
    # To use TensorRT, first run: 'yolo export model=yolov8s-seg.pt format=engine device=0'
    # Then point this path to the .engine file.
    model_path = 'yolov8s-seg.engine' 
    # If you haven't exported yet, you can use .pt, but it will be slower:
    # model_path = 'yolov8s-seg.pt'

    # --- 2. INITIALIZE ZED CAMERA ---
    print("Initializing ZED Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 or HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use ULTRA for higher precision
    init_params.coordinate_units = sl.UNIT.METER       # Measurements in meters

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open failed: {status}")
        sys.exit(1)

    # --- 3. LOAD YOLO MODEL ---
    print(f"Loading Model: {model_path}...")
    # The 'task' argument ensures we are in segmentation mode
    model = YOLO(model_path, task='segment')

    # --- 4. PREPARE BUFFERS ---
    image_zed = sl.Mat()
    depth_map = sl.Mat()
    point_cloud = sl.Mat()
    
    runtime_parameters = sl.RuntimeParameters()

    print("Starting Stream. Press 'q' to exit.")

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A. Retrieve data from ZED
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            
            # Convert ZED Mat to Numpy (RGBA -> RGB)
            frame_np = image_zed.get_data()
            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2RGB)

            # B. Run Inference (TensorRT happens here if .engine is used)
            # classes=[0, 1] filters for specific classes if trained on COCO (person, bicycle etc.)
            # You will likely need to train on a custom dataset for "road/curb", 
            # but for now we run generic inference.
            results = model(frame_rgb, verbose=False, imgsz=640)

            # C. Process Results
            annotated_frame = frame_rgb.copy()
            
            if results[0].masks is not None:
                # Get masks and boxes
                masks = results[0].masks.data.cpu().numpy() # Raw masks usually 160x160 or 640x640
                boxes = results[0].boxes.data.cpu().numpy()
                
                # Resize masks to original image size for overlay
                # Note: Ultralytics usually handles plotting, but for custom logic we do it manually
                for i, mask in enumerate(masks):
                    # Resize mask to frame dimensions
                    mask_resized = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]))
                    
                    # Create a binary threshold
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    if np.sum(mask_binary) > 0:
                        # --- ZED INTEGRATION START ---
                        # Find the centroid of the segmented area (e.g., the road)
                        M = cv2.moments(mask_binary)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            # Get the depth at the centroid of the detected object
                            err, depth_value = depth_map.get_value(cX, cY)
                            
                            if err == sl.ERROR_CODE.SUCCESS and not np.isnan(depth_value) and not np.isinf(depth_value):
                                label = f"Dist: {depth_value:.2f}m"
                                cv2.putText(annotated_frame, label, (cX, cY), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                cv2.circle(annotated_frame, (cX, cY), 5, (0, 0, 255), -1)
                        # --- ZED INTEGRATION END ---

                        # Draw the mask overlay
                        color = (0, 255, 0) # Green for driveable surface
                        colored_mask = np.zeros_like(frame_rgb, dtype=np.uint8)
                        colored_mask[mask_binary == 1] = color
                        
                        # Blend image with mask
                        annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.4, 0)

            # D. Display
            cv2.imshow("ZED Lane Segmentation", annotated_frame)

            key = cv2.waitKey(1)
            if key == 113 or key == 27: # 'q' or 'esc'
                break
    
    # Cleanup
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()