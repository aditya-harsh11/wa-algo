# 🏎️ Python Quick Run (Aditya)

Use this when your environment is already setup and you just want to run the code.

### 1. Launch Environment
1. Open **"Anaconda Prompt (Miniconda3)"**.
2. Activate your AI environment: `conda activate ll`
3. Enter the work directory: `cd "c:\Projects\Wisconsin Autonomous\lane-lines\lane-segmentation\python"`

### 2. Run Inference
*   **On a live Camera**:
    1. `python lanesegmentation_TwinLite_trt.py`

*   **On an SVO Video File**:
    1. `python lanesegmentation_TwinLite_trt.py --svo "../videos/Columbus/columbus.svo2"`
*(Use `--debug_mode 1` to see the live visualization windows)*

### What you are seeing:
1. **TwinLiteNet+ TensorRT**: The AI "Brain" (Green = Lanes, Blue = Drivable Area).
