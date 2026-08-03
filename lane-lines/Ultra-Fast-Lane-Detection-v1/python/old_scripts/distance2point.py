########################################################################
#
# Combined ZED SDK Script for Depth Measurement at Image Center
# Takes either a camera feed or .svo file as input.
#
########################################################################

import sys
import math
import argparse
import pyzed.sl as sl

def parse_args(init, opt):
    # Check and set input source based on input_svo_file or live camera
    if opt.input_svo_file and opt.input_svo_file.endswith(".svo2"):
        init.set_from_svo_file(opt.input_svo_file)
        print(f"[INFO] Using SVO file input: {opt.input_svo_file}")
    else:
        print("[INFO] Using live camera input")

    # Set resolution based on user input
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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it', default='')
    parser.add_argument('--resolution', type=str, help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA, VGA', default='HD720')
    opt = parser.parse_args()

    # Initialize ZED camera parameters
    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                             coordinate_units=sl.UNIT.METER,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    # Set up input parameters based on arguments
    parse_args(init, opt)

    # Initialize and open the camera
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Error opening the ZED Camera: " + repr(status))
        zed.close()
        exit()

    runtime_parameters = sl.RuntimeParameters()
    depth_map = sl.Mat()

    # Get image dimensions based on the input source
    if opt.input_svo_file and opt.input_svo_file.endswith(".svo2"):
        # Hardcoded values if using SVO file
        width, height = 1920, 1200
        print("[INFO] Using hardcoded resolution values for SVO input")
    else:
        # Dynamic values if camera is connected
        width = zed.get_camera_information().camera_resolution.width
        height = zed.get_camera_information().camera_resolution.height
        print(f"[INFO] Using camera resolution: {width}x{height}")

    # Set the picked coordinates for depth measurement
    center_x, center_y = width // 5, 3 * height // 4

    print("Press Ctrl+C to stop the program.\n")

    try:
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve depth data at the specified center point
                zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                depth_value = depth_map.get_value(center_x, center_y)[1]

                if math.isfinite(depth_value):
                    print(f"Depth at specified point ({center_x}, {center_y}): {depth_value} mm")
                else:
                    print("Depth at specified point could not be computed.")
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        zed.close()
        print("ZED camera closed.")

if __name__ == "__main__":
    main()
