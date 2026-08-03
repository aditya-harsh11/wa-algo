#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#define INPUT_H 512
#define INPUT_W 896
#define INPUT_C 3
#define OUTPUT_H 512
#define OUTPUT_W 896
#define OUTPUT_C 4  // Assuming 4 segmentation classes (BG, Road, Curbs, Marks)

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

Logger gLogger;


class LaneSegmentationEngine {
public:
    LaneSegmentationEngine(const std::string& engine_path);
    ~LaneSegmentationEngine();
    bool infer(const cv::Mat& input_image, cv::Mat& output_mask);

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    void* buffers[2]; // Input and output buffers
    cudaStream_t stream;

    bool loadEngine(const std::string& engine_path);
    cv::Vec3b getColorForSegmentation(int class_id);
};

LaneSegmentationEngine::LaneSegmentationEngine(const std::string& engine_path) {
    if (!loadEngine(engine_path)) {
        std::cerr << "Failed to load TensorRT engine!" << std::endl;
    }
    cudaStreamCreate(&stream);
}

LaneSegmentationEngine::~LaneSegmentationEngine() {
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
}

bool LaneSegmentationEngine::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), size));
    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT engine!" << std::endl;
        return false;
    }

    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create execution context!" << std::endl;
        return false;
    }

    // Allocate GPU memory for input/output
    cudaMalloc(&buffers[0], INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[1], OUTPUT_C * OUTPUT_H * OUTPUT_W * sizeof(float));

    // Print all tensor names in the model
    std::cout << "Model I/O Tensors:" << std::endl;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        std::cout << "  Tensor " << i << ": " << engine->getIOTensorName(i) << std::endl;
    }


    return true;
}

cv::Vec3b LaneSegmentationEngine::getColorForSegmentation(int class_id) {
    switch (class_id) {
        case 0: return {70, 70, 70};  // Background
        case 1: return {255, 0, 0};   // Road
        case 2: return {0, 255, 0};   // Curbs
        case 3: return {0, 0, 255};   // Lane Marks
        default: return {0, 0, 0};    // Unknown
    }
}

bool LaneSegmentationEngine::infer(const cv::Mat& input_image, cv::Mat& output_mask) {
    if (input_image.empty()) return false;

    cv::Size original_size = input_image.size();

    // Preprocess: Resize without color conversion
    cv::Mat resized;
    cv::resize(input_image, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat input_blob;
    resized.convertTo(input_blob, CV_32FC3); 

    cudaMemcpyAsync(buffers[0], input_blob.data, INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Set tensor addresses
    std::string input_tensor_name = engine->getIOTensorName(0);
    std::string output_tensor_name = engine->getIOTensorName(1);
    context->setTensorAddress(input_tensor_name.c_str(), buffers[0]);
    context->setTensorAddress(output_tensor_name.c_str(), buffers[1]);

    // Run inference
    if (!context->enqueueV3(stream)) {
        std::cerr << "TensorRT Inference Failed!" << std::endl;
        return false;
    }

    // Retrieve output
    std::vector<float> output_data(OUTPUT_C * OUTPUT_H * OUTPUT_W);
    cudaError_t err = cudaMemcpyAsync(output_data.data(), buffers[1], OUTPUT_C * OUTPUT_H * OUTPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Memcpy Error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    output_mask = cv::Mat(OUTPUT_H, OUTPUT_W, CV_8UC3);
#pragma omp parallel for
    for (int y = 0; y < OUTPUT_H; y++) {
        for (int x = 0; x < OUTPUT_W; x++) {
            float* scores = &output_data[y * (OUTPUT_W * OUTPUT_C) + x * OUTPUT_C];
            int max_idx = std::max_element(scores, scores + OUTPUT_C) - scores;
            output_mask.at<cv::Vec3b>(cv::Point(x, y)) = getColorForSegmentation(max_idx);
        }
    }

    cv::resize(output_mask, output_mask, original_size, 0, 0, cv::INTER_LINEAR);

    return true;
}


int main(int argc, char* argv[]) {
    std::string engine_path = "/home/wautodrive/VisionCore/lane-lines/lane-segmentation/weights/lane_segmentation.engine";
    std::string input_image_path = (argc > 1) ? argv[1] : "/home/wautodrive/VisionCore/lane-lines/lane-segmentation/test.jpg";

    LaneSegmentationEngine model(engine_path);

    cv::Mat input_image = cv::imread(input_image_path);
    if (input_image.empty()) {
        std::cerr << "Failed to load image: " << input_image_path << std::endl;
        return -1;
    }

    cv::Mat output_mask;
    if (model.infer(input_image, output_mask)) {
        std::string output_path = "/home/wautodrive/VisionCore/lane-lines/lane-segmentation/output.jpg";
        cv::imwrite(output_path, output_mask);
        std::cout << "Segmentation saved to: " << output_path << std::endl;
    } else {
        std::cerr << "Inference failed!" << std::endl;
    }

    cv::Mat mask_float;
    output_mask.convertTo(mask_float, CV_32FC3, 1.0 / 255.0);

    // Blend the mask with the original image
    cv::Mat blended_image;
    cv::addWeighted(input_image, 0.7, output_mask, 0.3, 0, blended_image);

    // Save the blended image
    std::string overlay_output_path = "/home/wautodrive/VisionCore/lane-lines/lane-segmentation/overlay_output.jpg";
    cv::imwrite(overlay_output_path, blended_image);
    std::cout << "Overlayed segmentation saved to: " << overlay_output_path << std::endl;


    return 0;
}
