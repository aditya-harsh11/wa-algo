import torch
import sys
import os
import argparse

# Add the directory containing 'model' to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "TwinLiteNetPlus"))

from TwinLiteNetPlus.model.model import TwinLiteNetPlus

def main(size):
    print("Loading PyTorch model...")

    # 1. FIX: Mock the args object required by TwinLiteNetPlus
    # Assuming the config key for large.pth is 'large' based on standard naming
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=size, type=str, help='Model configuration')
    args = parser.parse_args([]) 
    
    # Instantiate the model with the configuration arguments
    model = TwinLiteNetPlus(args)
    
    # 2. FIX: Safely load weights regardless of DataParallel training
    state_dict = torch.load(f'weights/{size}.pth', map_location='cpu')
    
    # Clean up the state_dict keys to remove 'module.' prefix if the model was 
    # saved using nn.DataParallel, avoiding the need to wrap our eval model.
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    # Load the cleaned weights into the model
    model.load_state_dict(clean_state_dict, strict=True)
    
    # Move the model to the GPU and set to evaluation mode
    model = model.cuda()
    model.eval()

    # Create the dummy input on the GPU (ensure resolution matches your deployment targets)
    dummy_input = torch.randn(1, 3, 384, 640).cuda()

    print("Exporting to ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        f"weights/twinlitenetplus_{size}.onnx", # Renamed to avoid mixing with standard TwinLiteNet
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['da_out', 'll_out']
    )
                      
    print("Exported to twinlitenetplus_large.onnx successfully!")

if __name__ == "__main__":
    main("large")
    main("medium")
    main("nano")
    main("small")