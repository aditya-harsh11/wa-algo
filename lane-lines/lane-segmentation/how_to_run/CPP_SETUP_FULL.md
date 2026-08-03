# 🛠️ C++ Full Setup Documentation (For Everyone)

This guide takes a fresh Windows machine to a fully functional C++ Lane Segmentation node.

### 1. Prerequisites (External Installs)
1. **Visual Studio 2022/2026**: Install "Desktop development with C++".
2. **CUDA Toolkit 12.8.1**: Download from [NVIDIA Archive](https://developer.nvidia.com/cuda-toolkit-archive).
3. **ZED SDK**: Install the version matching your CUDA Toolkit.
4. **CMake**: Install the latest Windows installer.
5. **TensorRT**: Download 10.x and extract to `C:/TensorRT`.

### 2. Configuration
The `CMakeLists.txt` is configured to look for TensorRT in `C:/TensorRT`. If you move it, update the HINTS in that file.

### 3. Compilation
1. Open the **"x64 Native Tools Command Prompt for VS"** (Crucial: 32-bit prompt will fail).
2. Execute these steps:
   1. Go to the C++ directory: `cd "C:\Projects\Wisconsin Autonomous\lane-lines\lane-segmentation\cpp"`
   2. Create a build folder: `mkdir build`
   3. Enter the folder: `cd build`
   4. Configure CMake: `"C:\Program Files\CMake\bin\cmake.exe" .. -A x64 -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib" -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"`
   5. Build the project: `"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release`

### 4. Running
The executable will be located in `build/Release/TwinLiteNet_ZED_CPP.exe`.
