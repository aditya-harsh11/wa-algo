cmake_minimum_required(VERSION 3.18)
project(lane-seg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)


add_executable(lane-seg
    src/main.cpp
    src/utils.cpp
    src/segmentation.cpp
    src/model_engine.cpp
    src/kernel.cu
    include/utils.hpp
    include/segmentation.hpp
    include/model_engine.hpp
    include/kernel.cuh
    include/argparse.hpp
)


find_package(OpenCV REQUIRED)
if(OpenCV_CUDA_VERSION)
    message(STATUS "Found OpenCV+CUDA install w/ CUDA version ${OpenCV_CUDA_VERSION}")
else()
    message(STATUS "Found OpenCV install WITHOUT CUDA")
endif()
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")


find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")

find_package(ZED REQUIRED)
message(STATUS "ZED_INCLUDE_DIRS: ${ZED_INCLUDE_DIRS}")
message(STATUS "ZED_LIBRARIES: ${ZED_LIBRARIES}")


if(WIN32)
    message(STATUS "Detected platform is Windows, so Nvidia VPI is unavailable")
else()
    message(STATUS "WILL search for Nvidia VPI (!!)")
    find_package(vpi QUIET)
    if(vpi_FOUND)
        message(STATUS "Found Nvidia VPI install automatically")
        target_link_libraries(lane-seg PRIVATE vpi)
        target_compile_definitions(lane-seg PUBLIC _OBJDET_VPI_BACKEND)
    else()
        if(OpenCV_CUDA_VERSION)
            message(WARNING "Nvidia VPI install NOT found on a non-Windows platform, defaulting to OpenCV+CUDA for parts of GPU backend")
        else()
            message(FATAL_ERROR "Found neither Nvidia VPI nor OpenCV+CUDA, required for parts of GPU backend")
        endif()
    endif()
endif()

find_package(TensorRT QUIET)
if(TensorRT_FOUND)
    message(STATUS "Found TensorRT install automatically")
    target_link_libraries(lane-seg PRIVATE nvinfer nvonnxparser)
else()
    message(STATUS "Could not find TensorRT install automatically, defaulting to platform-specific preset(s)")
    if(WIN32)
        set(TENSORRT_ROOT "C:/Users/djb11/source/libs/TensorRT-10.13.3.9")  # change me!!
        message(STATUS "[WIN32] Setting TensorRT root as: ${TENSORRT_ROOT}")
        message(STATUS "TENSORRT_ROOT: ${TENSORRT_ROOT}")
        message(STATUS "TENSORRT_LIB_DIR: ${TENSORRT_ROOT}/lib")
        target_include_directories(lane-seg PRIVATE ${TENSORRT_ROOT}/include)
        target_link_directories(lane-seg PRIVATE ${TENSORRT_ROOT}/lib)
        target_link_libraries(lane-seg PRIVATE
            nvinfer_10          # on windows I guess they're versioned with _10 as a suffix
            nvonnxparser_10     # ^^
        )
    else()
        set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
        set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
        message(STATUS "[JETSON] Setting TensorRT include dir as: ${TENSORRT_INCLUDE_DIR} and lib dir as: ${TENSORRT_LIB_DIR}")
        target_include_directories(lane-seg PRIVATE ${TENSORRT_INCLUDE_DIR})
        target_link_directories(lane-seg PRIVATE ${TENSORRT_LIB_DIR})
        target_link_libraries(lane-seg PRIVATE
            nvinfer
            nvonnxparser
        )
    endif()
endif()

if(WIN32)
    target_compile_definitions(lane-seg PUBLIC _OBJDET_CVCUDA_VIDEO_FIX)
else()
    target_include_directories(lane-seg PRIVATE /usr/include)  # specific fix for vpi on the jetson
endif()

target_include_directories(lane-seg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
)

target_link_libraries(lane-seg PRIVATE
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    cuda
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
endif()

get_property(INCLUDED_DIRS TARGET lane-seg PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "Final list of included directories: ${INCLUDED_DIRS}")

get_property(LINKED_LIBS TARGET lane-seg PROPERTY LINK_LIBRARIES)
message(STATUS "Final list of linked libraries: ${LINKED_LIBS}")