# 🏎️ C++ Quick Run (Aditya)

Use this when your environment is already setup and you just want to run the code.

### 1. Re-Compile (Only if you changed code)
1. Open **"x64 Native Tools Command Prompt for VS"**.
2. Go to build folder: `cd "C:\Projects\Wisconsin Autonomous\lane-lines\lane-segmentation\cpp\build"`
3. Configure CMake: `"C:\Program Files\CMake\bin\cmake.exe" .. -A x64 -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib" -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"`
4. Build: `"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release`

### 2. Run the Program
*   **On a live Camera**:
    1. Enter directory: `cd Release`
    2. Run: `TwinLiteNet_ZED_CPP.exe`

*   **On an SVO Video File**:
    1. Enter directory: `cd Release`
    2. Run: `TwinLiteNet_ZED_CPP.exe large true "../../../videos/Columbus/columbus.svo2"`
