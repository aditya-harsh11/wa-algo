cmake_minimum_required(VERSION 3.10)
project(TwinLiteNet_ZED_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ZED SDK
find_package(ZED 4 REQUIRED) # Change to 3 if using an older SDK

# Find TensorRT (Adjust paths if TensorRT is not in standard system paths)
find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/local/cuda/include /usr/include C:/TensorRT/include
)
find_library(TRT_LIBRARY NAMES nvinfer nvinfer_10
    HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib C:/TensorRT/lib
)
find_library(TRT_PLUGIN_LIBRARY NAMES nvinfer_plugin nvinfer_plugin_10
    HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib C:/TensorRT/lib
)
find_library(TRT_ONNX_LIBRARY NAMES nvonnxparser nvonnxparser_10
    HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib C:/TensorRT/lib
)

if (NOT TRT_LIBRARY OR NOT TRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT not found. Please specify TRT_INCLUDE_DIR and TRT_LIBRARY.")
endif()

# --- Build Executable ---
add_executable(${PROJECT_NAME} main_TwinLite.cpp)

# Includes
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)

# Linking
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    ${TRT_LIBRARY}
    ${TRT_PLUGIN_LIBRARY}
    ${TRT_ONNX_LIBRARY}
    ${CUDA_LIBRARIES}
)