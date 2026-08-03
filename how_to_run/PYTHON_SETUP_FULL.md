# 🛠️ Python Full Setup Documentation (For Everyone)

This guide takes a fresh Windows machine to a fully functional Python Lane Segmentation environment.

### 1. Prerequisites (External Installs)
1. **Miniconda/Anaconda**: Install and ensure you can open the Anaconda Prompt.
2. **CUDA Toolkit 12.8.1**: Crucial for TensorRT and PyTorch GPU acceleration.

### 2. Environment Initialization
Open **Anaconda Prompt** and run:
```cmd
cd "c:\Projects\Wisconsin Autonomous\lane-lines\lane-segmentation\python"
conda env create -f conda_base.yml
conda activate ll
pip install -r piprequirements.txt
```

### 3. Local Dependencies (Wheels)
Install the Windows-specific binaries in the `python` folder:
```cmd
pip install PyOpenGL-3.1.6-cp310-cp310-win_amd64.whl
pip install PyOpenGL_accelerate-3.1.6-cp310-cp310-win_amd64.whl
pip install pyzed-5.2-cp310-cp310-win_amd64.whl
```

### 4. Neural Network & Weights
1. **Clone Network Code**:
   `git clone https://github.com/chequanghuy/TwinLiteNetPlus.git`
2. **Download Weights**: Get `.pth` files (large, medium, etc) from [Google Drive](https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz).
3. **Move to folder**: Put weights into `python/weights/`.

### 5. Engine Generation
Pre-compile the model for your specific GPU:
```cmd
python pytorch_to_onnx.py
python convert_to_engine.py
```

### 6. Inference

1. **On a live Camera**:
```cmd
python lanesegmentation_TwinLite_trt.py
```

2. **On an SVO Video File**:
```cmd
python lanesegmentation_TwinLite_trt.py --svo "../videos/Columbus/columbus.svo2"
```
*(Use `--debug_mode 1` to see the live visualization windows)*

### What you are seeing:
1. **TwinLiteNet+ TensorRT**: The AI "Brain" (Green = Lanes, Blue = Drivable Area).
