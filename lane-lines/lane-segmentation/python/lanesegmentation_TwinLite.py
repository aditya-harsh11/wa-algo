import os
import argparse
import sys
import cv2
import numpy as np
import pyzed.sl as sl
import torch
import time
from itertools import count

# Import TwinLiteNet from the local model/ folder
from TwinLiteNet.model import TwinLite as net

# --- CONFIGURATION ---
INPUT_W = 640
INPUT_H = 360
OUTPUT_W = 640
OUTPUT_H = 360

# --- MAIN ENGINE CLASS (PyTorch / TwinLiteNet) ---
class TwinLiteEngine:
    def __init__(self, weight_path):
        print(f"Loading TwinLiteNet+ Weights: {weight_path}")
        self.model = net.TwinLiteNet()
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()

    def infer(self, image):
        start_time = time.perf_counter()

        # 1. Resize and format for PyTorch
        img_rs = cv2.resize(image, (INPUT_W, INPUT_H))
        
        # BGR to RGB, then HWC to CHW format
        img_ts = img_rs[:, :, ::-1].transpose(2, 0, 1)
        img_ts = np.ascontiguousarray(img_ts)
        
        # Convert to Tensor, Add Batch Dim, Normalize to 0-1
        img_tensor = torch.from_numpy(img_ts)
        img_tensor = torch.unsqueeze(img_tensor, 0) 
        img_tensor = img_tensor.cuda().float() / 255.0

        # 2. Inference
        with torch.no_grad():
            img_out = self.model(img_tensor)
        
        # 3. Process Outputs
        # x0 = Drivable Area (Shape: [1, 2, 360, 640])
        # x1 = Lane Lines (Shape: [1, 2, 360, 640])
        x0 = img_out[0]
        x1 = img_out[1]

        # Convert back to Numpy (H, W, C) so we can apply our geometric prior easily
        da_scores = x0.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ll_scores = x1.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f"Inference time: {elapsed_time}")

        return da_scores, ll_scores, elapsed_time

# --- GEOMETRIC PRIOR WITH DEPTH LIMIT & EDGE MASKING ---
def apply_geometric_prior(da_scores, point_cloud_xyz):
    """
    Modifies DA scores based on geometry.
    Uses bottom-center sampling and ignores points that are too far away or on the edges.
    """
    # 1. Extract Forward Distance (X) and Height (Z)
    forward_dist = point_cloud_xyz[:, :, 0] 
    z_height = point_cloud_xyz[:, :, 2] 
    
    # 2. Resize to match neural network output
    z_height_resized = cv2.resize(z_height, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_NEAREST)
    forward_dist_resized = cv2.resize(forward_dist, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_NEAREST)

    # 3. Create Edge Mask
    EDGE_PERCENT = 0.08
    edge_pixels = int(OUTPUT_W * EDGE_PERCENT)
    valid_fov = np.zeros((OUTPUT_H, OUTPUT_W), dtype=bool)
    valid_fov[:, edge_pixels:-edge_pixels] = True 

    # --- NEW: Create Distance Mask ---
    MAX_DEPTH_LIMIT = 15.0  # Limit in meters (e.g., trust geometry up to 15m)
    
    # Suppress NaN warnings for the distance check
    with np.errstate(invalid='ignore'):
        is_within_range = forward_dist_resized < MAX_DEPTH_LIMIT

    # 4. Final Validity Mask
    # Valid IF: Real number AND inside edge mask AND closer than MAX_DEPTH_LIMIT
    has_depth_data = np.isfinite(z_height_resized) & is_within_range  & valid_fov 

    # 5. Find Ground Level (using only the center patch)
    h, w = z_height_resized.shape
    sample_patch = z_height_resized[h-50:h, w//2-50:w//2+50] 
    valid_samples = sample_patch[np.isfinite(sample_patch)]
    
    if len(valid_samples) == 0:
        blank_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=bool)
        return da_scores, blank_mask, blank_mask 

    ground_level = np.median(valid_samples)

    # 6. Determine Geometry
    HEIGHT_TOLERANCE = 0.20 # +/- 20cm
    dist_from_ground = np.abs(np.nan_to_num(z_height_resized) - ground_level)
    
    is_confirmed_ground = (dist_from_ground < HEIGHT_TOLERANCE) & has_depth_data
    is_confirmed_obstacle = (dist_from_ground >= HEIGHT_TOLERANCE) & has_depth_data

    # 7. Apply Boost/Penalty
    ROAD_IDX = 1
    BOOST_AMOUNT = 2.0  
    PENALTY_AMOUNT = 1.0 

    da_scores[:, :, ROAD_IDX] += (is_confirmed_ground * BOOST_AMOUNT)
    da_scores[:, :, ROAD_IDX] -= (is_confirmed_obstacle * PENALTY_AMOUNT)
    
    return da_scores, is_confirmed_ground, is_confirmed_obstacle


# --- ZED SETUP ---
def setup_zed_camera(opt):
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080,
                             depth_mode=sl.DEPTH_MODE.NEURAL, # Or NEURAL if you fixed your TensorRT cache
                             coordinate_units=sl.UNIT.METER,
                             camera_fps=60,
                             depth_minimum_distance=1.0,
                             depth_maximum_distance=40,
                             depth_stabilization=0,
                             async_grab_camera_recovery=True,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD)

    if len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4:
            if len(ip_str.split(':')) == 2:
                init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
            else:
                init.set_from_stream(ip_str)
            print(f"[Sample] Using Stream input, IP: {ip_str}")

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to open ZED Camera: {status}")
        sys.exit(1)

    # (Removed the positional tracking block from here)
    return zed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="TwinLiteNet/pretrained/best.pth")
    parser.add_argument('--ip_address', type=str, default='127.0.0.1:30000')
    parser.add_argument('--debug_mode', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        sys.exit(1)

    # Initialize PyTorch Model and ZED
    model = TwinLiteEngine(args.weights)
    zed = setup_zed_camera(args)
    
    runtime_parameters = sl.RuntimeParameters()
    
    left_image = sl.Mat()
    point_cloud = sl.Mat()

    average_infer_time = 0

    print("Starting TwinLiteNet+ Inference with Geometric Priors...")
    try:
        for step in count():
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                
                # 1. Retrieve Camera Data
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                
                rgb_frame = left_image.get_data()
                if rgb_frame.shape[2] == 4:
                    ip_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGRA2BGR)
                else:
                    ip_frame = rgb_frame
                
                point_cloud_data = point_cloud.get_data()

                # 2. Neural Net Inference
                da_scores, ll_scores, infer_time = model.infer(ip_frame)
                if step != 0:
                    average_infer_time = float(average_infer_time * (step-1) + infer_time) / float(step)
                    print(f"Average Inference time: {average_infer_time}")
                
                # 3. Apply Geometric Prior to Drivable Area
                # Unpack the two new masks here:
                #enhanced_da_scores, ground_mask, obstacle_mask = apply_geometric_prior(da_scores, point_cloud_data)
                enhanced_da_scores = da_scores

                # 4. Final Classification (Argmax)
                da_predict = np.argmax(enhanced_da_scores, axis=2)
                ll_predict = np.argmax(ll_scores, axis=2)
                
                # 5. Visualization
                if args.debug_mode:
                    img_rs = cv2.resize(ip_frame, (INPUT_W, INPUT_H))
                    
                    # --- WINDOW 1: Neural Net Segmentation ---
                    net_mask = np.zeros_like(img_rs)
                    net_mask[da_predict > 0] = [255, 0, 0] # Blue: Drivable Area
                    net_mask[ll_predict > 0] = [0, 255, 0] # Green: Lane Lines
                    
                    net_overlay = cv2.addWeighted(img_rs, 0.7, net_mask, 0.3, 0)
                    cv2.imshow("TwinLiteNet+ Segmentation", net_overlay)

                    # --- WINDOW 2: Depth Geometry Debug ---
                    #geo_mask = np.zeros_like(img_rs)
                    #geo_mask[ground_mask] = [255, 255, 0]  # Cyan (BGR): Confirmed Ground
                    #geo_mask[obstacle_mask] = [0, 0, 255]  # Red (BGR): Confirmed Obstacle
                    
                    #geo_overlay = cv2.addWeighted(img_rs, 0.7, geo_mask, 0.3, 0)
                    #cv2.imshow("ZED Depth Geometry", geo_overlay)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    except KeyboardInterrupt:
        print("Stopping...")

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()