#include "model_engine.hpp"

lane_seg::TwinLite::TwinLite(const std::filesystem::path& modelPath, const Configuration& config) :
    path(modelPath),
    config(config),
    inputIndex(-1),
    llIndex(-1),
    daIndex(-1)
{
    if (!loadEngine(modelPath)) {
        globalLogger.error("Failed to load TensorRT engine model(s)!");
        return;
    }

    if (!allocate()) {
        globalLogger.error("Failed to allocate data for TensorRT engine model(s)!");
        return;
    }
}

lane_seg::TwinLite::~TwinLite() {
    cudaStreamDestroy(stream);
    cudaStreamDestroy(altStream);

    for (auto& tensorMem : engineAllocs) {
        switch (tensorMem.location) {
        case TensorLocation::kDEVICE:
            cudaFree(tensorMem.data);
            break;
        case TensorLocation::kHOST:
            cudaFreeHost(tensorMem.data);
            break;
        default:
            globalLogger.log(Severity::kWARNING, "Attempted & FAILED to deallocate memory at unknown tensor location #" + std::to_string((int)tensorMem.location));
            break;
        }
    }
}

bool lane_seg::TwinLite::predict(
    const sl::Mat& frameBGR, const sl::Mat& depthXYZ, 
    std::vector<float>& llScores, std::vector<float>& daScores, 
    bool computeMasks) {

    if (frameBGR.getWidth() == 0 || frameBGR.getHeight() == 0) {
        globalLogger.error("Input to predict is empty");
        return false;
    }

    if (depthXYZ.getWidth() == 0 || depthXYZ.getHeight() == 0) {
        globalLogger.error("Depth input is empty");
        return false;
    }

    if (!preprocess(frameBGR)) {
        globalLogger.error("Lane seg preprocessing failed!");
        return false;
    }

    if (!runEngine()) {
        globalLogger.error("Lane seg inference failed!");
        return false;
    }

    if (!postprocess(frameBGR, depthXYZ, llScores, daScores, computeMasks)) {
        globalLogger.error("Lane seg postprocessing failed!");
        return false;
    }

    return true;
}

bool lane_seg::TwinLite::isLoaded() const {
    return engine != nullptr;
}

constexpr const std::filesystem::path& lane_seg::TwinLite::getFilePath() const {
    return path;
}

constexpr const lane_seg::TwinLite::Configuration& lane_seg::TwinLite::getConfig() const {
    return config;
}

constexpr const cv::Mat& lane_seg::TwinLite::getGroundMask() const {
    return groundMask;
}

constexpr const cv::Mat& lane_seg::TwinLite::getObstacleMask() const {
    return obstacleMask;
}

// if this fails, the problem might be a mismatch between enterprise and community versions of TensorRT - learned the hard way
// if so, build with the trtexec.exe that comes with the enterprise version
bool lane_seg::TwinLite::loadEngine(const std::filesystem::path& modelPath) {
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file) {
        globalLogger.error("Failed to open engine file: \"" + modelPath.string() + "\"");
        return false;
    }

    std::size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(globalLogger));
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), size));
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    
    if (!runtime) {
        globalLogger.error("Failed to create TensorRT inference runtime");
        return false;
    }
    
    if (!engine) {
        globalLogger.error("Failed to deserialize TensorRT seg engine");
        return false;
    }

    if (!context) {
        globalLogger.error("Failed to create TensorRT execution context for seg engine");
        return false;
    }

    globalLogger.log(Severity::kINFO, "Loaded engine \"" + std::string(engine->getName()) + "\" with the following params: ");
    globalLogger.log(Severity::kINFO, "Engine capability: " + std::to_string((int)engine->getEngineCapability()) + "  (nvinfer1::EngineCapability)");
    globalLogger.log(Severity::kINFO, "# I/O tensors: " + std::to_string(engine->getNbIOTensors()));
    for (std::int32_t index = 0; index < engine->getNbIOTensors(); index++) {
        const std::string name(engine->getIOTensorName(index));

        auto format = engine->getTensorFormatDesc(name.c_str());
        auto dataType = engine->getTensorDataType(name.c_str());
        auto ioMode = engine->getTensorIOMode(name.c_str());
        auto location = engine->getTensorLocation(name.c_str());
        auto shape = engine->getTensorShape(name.c_str());

        std::string output = "\t[" + std::to_string(index) + "] \"" + name + "\": ";
        output += std::string(format) + " (";
        for (std::int32_t iter = 0; iter < shape.nbDims; iter++) {
            output += std::to_string(shape.d[iter]);
            if (iter != shape.nbDims - 1) {
                output += " x ";
            }
        }
        output += std::string("), ") + (location == nvinfer1::TensorLocation::kDEVICE ? "DEVICE" : "HOST");
        globalLogger.log(Severity::kINFO, output);

        if (name == "images") {
            inputIndex = index;
        } else if (name == "ll_out") {
            llIndex = index;
        } else if (name == "da_out") {
            daIndex = index;
        }
    }

    return true;
}

bool lane_seg::TwinLite::allocate() {
    cudaError_t msg = cudaStreamCreate(&stream);
    if (msg != cudaSuccess) {
        globalLogger.error("CUDA stream create error: \"" + std::string(cudaGetErrorString(msg)) + "\"");
        return false;
    }
    msg = cudaStreamCreate(&altStream);
    if (msg != cudaSuccess) {
        globalLogger.error("CUDA stream create error: \"" + std::string(cudaGetErrorString(msg)) + "\"");
        return false;
    }

    engineAllocs.resize(engine->getNbIOTensors());
    for (std::int32_t index = 0; index < engine->getNbIOTensors(); index++) {
        const char* name = engine->getIOTensorName(index);

        auto shape = engine->getTensorShape(name);
        auto type = engine->getTensorDataType(name);
        auto location = engine->getTensorLocation(name);

        std::size_t size = 1;
        for (std::int32_t iter = 0; iter < shape.nbDims; iter++) {
            size *= shape.d[iter];
        }

        using nvinfer1::DataType;

        switch (type) {
        case DataType::kINT64:
            size *= 8;
            break;

        case DataType::kFLOAT:
        case DataType::kINT32:
            size *= 4;
            break;

        case DataType::kHALF:
        case DataType::kBF16:
            size *= 2;
            break;

        case DataType::kINT8:
        case DataType::kUINT8:
        case DataType::kFP8:
        case DataType::kBOOL:
        case DataType::kE8M0:
            size *= 1;
            break;

        case DataType::kINT4:
        case DataType::kFP4:
            if (size % 2 != 0) {
                globalLogger.error("Tensor \"" + std::string(name) + "\" (" + std::to_string(index) + ") specifies a half-byte data type with odd total dims");
                return false;
            }
            size /= 2;
            break;

        default:
            globalLogger.error("Tensor \"" + std::string(name) + "\" (" + std::to_string(index) + ") specifies unsupported data type #" + std::to_string((int)type));
            return false;
        }


        switch (location) {
        case TensorLocation::kDEVICE:
            msg = cudaMalloc(&engineAllocs[index].data, size);
            if (msg != cudaSuccess) {
                globalLogger.error("CUDA device malloc error: \"" + std::string(cudaGetErrorString(msg)) + "\"");
                return false;
            }
            break;
        case TensorLocation::kHOST:
            msg = cudaMallocHost(&engineAllocs[index].data, size);
            if (msg != cudaSuccess) {
                globalLogger.error("CUDA host malloc error: \"" + std::string(cudaGetErrorString(msg)) + "\"");
                return false;
            }
            break;
        default:
            globalLogger.error("Tensor \"" + std::string(name) + "\" (" + std::to_string(index) + ") specifies unsupported location #" + std::to_string((int)location));

        }
        
        engineAllocs[index].location = location;
        engineAllocs[index].size = size;
    }

    msg = cudaEventCreateWithFlags(&preprocessEnd, cudaEventDisableTiming | cudaEventBlockingSync);
    if (msg != cudaSuccess) {
        globalLogger.error("CUDA preprocessEnd event creation error:\"" + std::string(cudaGetErrorString(msg)) + "\"");
        return false;
    }

    msg = cudaEventCreateWithFlags(&inferenceEnd, cudaEventDisableTiming | cudaEventBlockingSync);
    if (msg != cudaSuccess) {
        globalLogger.error("CUDA inferenceEnd event creation error:\"" + std::string(cudaGetErrorString(msg)) + "\"");
        return false;
    }

    return true;
}

bool lane_seg::TwinLite::preprocess(const sl::Mat& leftBGR) {
    if (!device::queuePreprocess(leftBGR, (float*)engineAllocs[inputIndex].data, config.inputSize, config.inputChannels, stream)) {
        globalLogger.error("Failed to queue preprocessing");
        return false;
    }
    
    return true;
}

bool lane_seg::TwinLite::runEngine()  {
    if (!context->enqueueV3(stream)) {
        globalLogger.error("TensorRT lane seg inference failed");
        return false;
    }

    return true;
}

bool lane_seg::TwinLite::postprocess(
    const sl::Mat& frameBGR, const sl::Mat& depthXYZ,
    std::vector<float>& llScores, std::vector<float>& daScores,
    bool computeMasks) {

    device::queuePostprocess(
        (float*)engineAllocs[llIndex].data, (float*)engineAllocs[daIndex].data,
        (float*)llConvertBuffer.data, (float*)daConvertBuffer.data,
        config.outputSize, stream
    );

    llScores.resize(llConvertBuffer.size / sizeof(float));
    daScores.resize(daConvertBuffer.size / sizeof(float));

    cudaMemcpyAsync(&llScores[0], llConvertBuffer.data, llConvertBuffer.size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&daScores[0], daConvertBuffer.data, daConvertBuffer.size, cudaMemcpyDeviceToHost, stream);

    cudaEventSynchronize(inferenceEnd);
    cudaStreamSynchronize(stream);

    // TODO: optimize me into a bunch of kernels, etc. etc.
    if (computeMasks) {
        applyGeometricPrior(daScores, depthXYZ);
    }

    //cv::parallel_for_(cv::Range(0, detections.size()), [&](const cv::Range& range) {
    //    for (int index = range.start; index < range.end; index++) {
    //        detections[index].classId = modelIdMap.at((int)detections[index].classId);
    //    }
    //});

    return true;
}

void lane_seg::TwinLite::applyGeometricPrior(std::vector<float> daScores, const sl::Mat& depthXYZ) {
    // Convert SL PointCloud to separate X and Z cv::Mats
    cv::Mat x_mat(depthXYZ.getHeight(), depthXYZ.getWidth(), CV_32FC1);
    cv::Mat z_mat(depthXYZ.getHeight(), depthXYZ.getWidth(), CV_32FC1);

    sl::float4* pc_ptr = depthXYZ.getPtr<sl::float4>();
    float* x_ptr = (float*)x_mat.data;
    float* z_ptr = (float*)z_mat.data;
    int total_pixels = depthXYZ.getWidth() * depthXYZ.getHeight();

    for (int i = 0; i < total_pixels; ++i) {
        x_ptr[i] = pc_ptr[i].x;
        z_ptr[i] = pc_ptr[i].z;
    }

    cv::Mat x_resized, z_resized;
    cv::resize(x_mat, x_resized, cv::Size(config.outputSize.width, config.outputSize.height), 0, 0, cv::INTER_NEAREST);
    cv::resize(z_mat, z_resized, cv::Size(config.outputSize.width, config.outputSize.height), 0, 0, cv::INTER_NEAREST);

    // Dynamic Slope Fitting (Polyfit)
    int strip_h_start = config.outputSize.height * 0.50;
    int strip_w_start = config.outputSize.width * 0.40;
    int strip_w_end = config.outputSize.width * 0.60;

    double sum_x = 0, sum_z = 0, sum_xx = 0, sum_xz = 0;
    int valid_samples = 0;

    for (int r = strip_h_start; r < config.outputSize.height; ++r) {
        for (int c = strip_w_start; c < strip_w_end; ++c) {
            float x = x_resized.at<float>(r, c);
            float z = z_resized.at<float>(r, c);
            if (std::isfinite(x) && std::isfinite(z)) {
                sum_x += x; sum_z += z;
                sum_xx += x * x; sum_xz += x * z;
                valid_samples++;
            }
        }
    }

    float m = 0.0f, c_val = 0.0f;
    if (valid_samples > 100) {
        float denom = valid_samples * sum_xx - sum_x * sum_x;
        if (std::abs(denom) > 1e-6) {
            m = (valid_samples * sum_xz - sum_x * sum_z) / denom;
            c_val = (sum_z - m * sum_x) / valid_samples;
            m = std::clamp(m, -0.35f, 0.35f);
        }
    }

    // Evaluate Geometry and Apply Boost/Penalty
    int edge_pixels = config.outputSize.width * 0.08;
    float max_depth = 15.0f;
    float height_tol = 0.20f;
    float boost_amount = 0.0f;
    float penalty_amount = 0.0f;

    groundMask = cv::Mat::zeros(config.outputSize.height, config.outputSize.width, CV_8UC1);
    obstacleMask = cv::Mat::zeros(config.outputSize.height, config.outputSize.width, CV_8UC1);

    float* da_road_channel = &daScores[0] + (1 * config.outputSize.height * config.outputSize.width); // Access Road IDX (1)

    for (int r = 0; r < config.outputSize.height; ++r) {
        for (int col = 0; col < config.outputSize.width; ++col) {
            float x = x_resized.at<float>(r, col);
            float z = z_resized.at<float>(r, col);

            bool valid_fov = (col >= edge_pixels && col < config.outputSize.width - edge_pixels);
            bool is_within_range = (x < max_depth);
            bool has_depth = std::isfinite(z) && valid_fov && is_within_range;

            if (has_depth) {
                float expected_z = (m * x) + c_val;
                float dist = std::abs(z - expected_z);

                int idx = r * config.outputSize.width + col;

                if (dist < height_tol) {
                    da_road_channel[idx] += boost_amount;
                    groundMask.at<uchar>(r, col) = 255;
                } else {
                    da_road_channel[idx] -= penalty_amount;
                    obstacleMask.at<uchar>(r, col) = 255;
                }
            }
        }
    }
}