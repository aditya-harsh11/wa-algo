#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

using namespace nvinfer1;

// --- CONFIGURATION ---
const int INPUT_W = 640;
const int INPUT_H = 360;
const int OUTPUT_W = 640;
const int OUTPUT_H = 360;
const int INPUT_C = 3;

// --- TRT LOGGER ---
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

// --- MEMORY HELPER ---
struct HostDeviceMem {
    void* host = nullptr;
    void* device = nullptr;
    size_t size = 0;

    void allocate(size_t bytes) {
        size = bytes;
        cudaMallocHost(&host, size);
        cudaMalloc(&device, size);
    }
    void free() {
        if (host) cudaFreeHost(host);
        if (device) cudaFree(device);
    }
};

// --- PROFILER HELPER ---
struct ProfilerStat { float current = 0.0f; float avg = 0.0f; int count = 0; };
class Profiler {
    std::map<std::string, ProfilerStat> stats;
public:
    void update(const std::string& name, float dt_ms) {
        auto& s = stats[name];
        s.current = dt_ms;
        s.count++;
        s.avg += (dt_ms - s.avg) / s.count;
    }
    void print_summary() {
        std::string out;
        for (const auto& kv : stats) {
            out += kv.first + ": " + std::to_string((int)kv.second.current) + "ms (Avg: " + std::to_string((int)kv.second.avg) + "ms) | ";
        }
        std::cout << "\r" << out << std::flush;
    }
};

// --- MAIN ENGINE CLASS ---
class TwinLiteTRTEngine {
private:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream;

public:
    std::map<std::string, HostDeviceMem> allocations;

    TwinLiteTRTEngine(const std::string& engine_path) {
        initLibNvInferPlugins(&gLogger, "");
        cudaStreamCreate(&stream);

        // Load Engine
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.good()) throw std::runtime_error("Error reading engine file.");
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> trtModelStream(size);
        file.read(trtModelStream.data(), size);

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
        context = engine->createExecutionContext();

        // Allocate Buffers
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            Dims dims = engine->getTensorShape(name);
            
            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d) vol *= std::abs(dims.d[d]);
            
            // Assuming float32 for simplicity based on Python script
            size_t bytes = vol * sizeof(float);
            allocations[name].allocate(bytes);

            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                Dims4 inputDims(1, INPUT_C, INPUT_H, INPUT_W);
                context->setInputShape(name, inputDims);
            }
        }
    }

    ~TwinLiteTRTEngine() {
        for (auto& kv : allocations) kv.second.free();
        cudaStreamDestroy(stream);
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }

    void infer(const cv::Mat& image) {
        // 1. Preprocess: Resize -> BGR2RGB -> Normalize -> NCHW
        cv::Mat resized, rgb, float_img;
        cv::resize(image, resized, cv::Size(INPUT_W, INPUT_H));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

        // Split channels for NCHW format
        std::vector<cv::Mat> channels;
        cv::split(float_img, channels);

        // Copy to input buffer
        float* host_input = nullptr;
        for (auto& kv : allocations) {
            if (engine->getTensorIOMode(kv.first.c_str()) == TensorIOMode::kINPUT) {
                host_input = static_cast<float*>(kv.second.host);
                context->setTensorAddress(kv.first.c_str(), kv.second.device);
                break;
            }
        }

        int channel_size = INPUT_H * INPUT_W;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(host_input + c * channel_size, channels[c].data, channel_size * sizeof(float));
        }

        // 2. Transfer to Device
        for (auto& kv : allocations) {
            if (engine->getTensorIOMode(kv.first.c_str()) == TensorIOMode::kINPUT) {
                cudaMemcpyAsync(kv.second.device, kv.second.host, kv.second.size, cudaMemcpyHostToDevice, stream);
            } else {
                context->setTensorAddress(kv.first.c_str(), kv.second.device);
            }
        }

        // 3. Execute Async
        context->enqueueV3(stream);

        // 4. Transfer outputs back to host
        for (auto& kv : allocations) {
            if (engine->getTensorIOMode(kv.first.c_str()) == TensorIOMode::kOUTPUT) {
                cudaMemcpyAsync(kv.second.host, kv.second.device, kv.second.size, cudaMemcpyDeviceToHost, stream);
            }
        }
        cudaStreamSynchronize(stream);
    }
};

// --- GEOMETRIC PRIOR WITH DYNAMIC SLOPE FITTING ---
void apply_geometric_prior(float* da_scores, const sl::Mat& point_cloud, cv::Mat& ground_mask, cv::Mat& obstacle_mask) {
    // Convert SL PointCloud to separate X and Z cv::Mats
    cv::Mat x_mat(point_cloud.getHeight(), point_cloud.getWidth(), CV_32FC1);
    cv::Mat z_mat(point_cloud.getHeight(), point_cloud.getWidth(), CV_32FC1);

    sl::float4* pc_ptr = point_cloud.getPtr<sl::float4>();
    float* x_ptr = (float*)x_mat.data;
    float* z_ptr = (float*)z_mat.data;
    int total_pixels = point_cloud.getWidth() * point_cloud.getHeight();

    for (int i = 0; i < total_pixels; ++i) {
        x_ptr[i] = pc_ptr[i].x;
        z_ptr[i] = pc_ptr[i].z;
    }

    cv::Mat x_resized, z_resized;
    cv::resize(x_mat, x_resized, cv::Size(OUTPUT_W, OUTPUT_H), 0, 0, cv::INTER_NEAREST);
    cv::resize(z_mat, z_resized, cv::Size(OUTPUT_W, OUTPUT_H), 0, 0, cv::INTER_NEAREST);

    // Dynamic Slope Fitting (Polyfit)
    int strip_h_start = OUTPUT_H * 0.50;
    int strip_w_start = OUTPUT_W * 0.40;
    int strip_w_end = OUTPUT_W * 0.60;

    double sum_x = 0, sum_z = 0, sum_xx = 0, sum_xz = 0;
    int valid_samples = 0;

    for (int r = strip_h_start; r < OUTPUT_H; ++r) {
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
    int edge_pixels = OUTPUT_W * 0.08;
    float max_depth = 15.0f;
    float height_tol = 0.20f;
    float boost_amount = 0.0f;
    float penalty_amount = 0.0f;

    ground_mask = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_8UC1);
    obstacle_mask = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_8UC1);

    float* da_road_channel = da_scores + (1 * OUTPUT_H * OUTPUT_W); // Access Road IDX (1)

    for (int r = 0; r < OUTPUT_H; ++r) {
        for (int col = 0; col < OUTPUT_W; ++col) {
            float x = x_resized.at<float>(r, col);
            float z = z_resized.at<float>(r, col);
            
            bool valid_fov = (col >= edge_pixels && col < OUTPUT_W - edge_pixels);
            bool is_within_range = (x < max_depth);
            bool has_depth = std::isfinite(z) && valid_fov && is_within_range;

            if (has_depth) {
                float expected_z = (m * x) + c_val;
                float dist = std::abs(z - expected_z);

                int idx = r * OUTPUT_W + col;

                if (dist < height_tol) {
                    da_road_channel[idx] += boost_amount;
                    ground_mask.at<uchar>(r, col) = 255;
                } else {
                    da_road_channel[idx] -= penalty_amount;
                    obstacle_mask.at<uchar>(r, col) = 255;
                }
            }
        }
    }
}

// --- ZED SETUP ---
void setup_zed_camera(sl::Camera& zed, const std::string& ip_address) {
    sl::InitParameters init;
    init.camera_resolution = sl::RESOLUTION::HD1080;
    init.depth_mode = sl::DEPTH_MODE::NEURAL;
    init.coordinate_units = sl::UNIT::METER;
    init.camera_fps = 60;
    init.depth_minimum_distance = 1.0f;
    init.depth_maximum_distance = 40.0f;
    init.depth_stabilization = 0;
    init.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;

    if (!ip_address.empty()) {
        std::string ip = ip_address.substr(0, ip_address.find(':'));
        int port = std::stoi(ip_address.substr(ip_address.find(':') + 1));
        init.input.setFromStream(sl::String(ip.c_str()), port);
    }

    // Notice we removed 'sl::Camera zed;' from here. We just use the one passed in.
    sl::ERROR_CODE status = zed.open(init);
    if (status != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to open ZED: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
    // No need to return anything anymore
}

int main(int argc, char** argv) {
    std::string engine_file = "../../python/weights/twinlitenet.engine";
    std::string ip_address = "127.0.0.1:30000";
    bool debug_mode = true;

    // A lightweight command line check for basic usage
    if (argc > 1) engine_file = argv[1];

    TwinLiteTRTEngine model(engine_file);
    
    sl::Camera zed; // Create the empty camera object first
    setup_zed_camera(zed, ip_address); // Pass it by reference to initialize it
    
    Profiler profiler;

    sl::RuntimeParameters runtime_parameters;
    sl::Mat left_image, point_cloud;

    std::cout << "Starting C++ TensorRT Inference...\n";

    while (true) {
        if (zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS) {
            
            // 1. ZED Retrieval
            auto t0 = std::chrono::high_resolution_clock::now();
            zed.retrieveImage(left_image, sl::VIEW::LEFT);
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZ);
            
            cv::Mat ip_frame(left_image.getHeight(), left_image.getWidth(), CV_8UC4, left_image.getPtr<sl::uchar1>(sl::MEM::CPU));
            cv::cvtColor(ip_frame, ip_frame, cv::COLOR_BGRA2BGR); // ZED gives BGRA

            auto t1 = std::chrono::high_resolution_clock::now();
            profiler.update("1. Retrieve", std::chrono::duration<float, std::milli>(t1 - t0).count());

            // 2. TRT Inference
            model.infer(ip_frame);
            float* da_scores = static_cast<float*>(model.allocations["da_out"].host);
            float* ll_scores = static_cast<float*>(model.allocations["ll_out"].host);

            auto t2 = std::chrono::high_resolution_clock::now();
            profiler.update("2. TensorRT", std::chrono::duration<float, std::milli>(t2 - t1).count());

            // 3. Geometric Prior
            cv::Mat ground_mask, obstacle_mask;
            apply_geometric_prior(da_scores, point_cloud, ground_mask, obstacle_mask);

            auto t3 = std::chrono::high_resolution_clock::now();
            profiler.update("3. GeoPrior", std::chrono::duration<float, std::milli>(t3 - t2).count());

            // 4. Post-processing & Visualization
            if (debug_mode) {
                cv::Mat img_rs;
                cv::resize(ip_frame, img_rs, cv::Size(INPUT_W, INPUT_H));

                cv::Mat net_overlay = img_rs.clone();
                cv::Mat geo_overlay = img_rs.clone();

                // Argmax for Net (Simplified C++ equivalent mapping CHW flat arrays back to HWC space)
                int plane_size = OUTPUT_H * OUTPUT_W;
                for (int i = 0; i < plane_size; ++i) {
                    if (da_scores[1 * plane_size + i] > da_scores[0 * plane_size + i]) {
                        net_overlay.at<cv::Vec3b>(i / OUTPUT_W, i % OUTPUT_W) = cv::Vec3b(255, 0, 0); // Blue (Drivable)
                    }
                    if (ll_scores[1 * plane_size + i] > ll_scores[0 * plane_size + i]) {
                        net_overlay.at<cv::Vec3b>(i / OUTPUT_W, i % OUTPUT_W) = cv::Vec3b(0, 255, 0); // Green (Lines)
                    }

                    // Geo Overlay mapping
                    int r = i / OUTPUT_W, c = i % OUTPUT_W;
                    if (ground_mask.at<uchar>(r, c)) geo_overlay.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 255); // Yellow
                    if (obstacle_mask.at<uchar>(r, c)) geo_overlay.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255); // Red
                }

                cv::addWeighted(img_rs, 0.5, net_overlay, 0.5, 0, net_overlay);
                cv::addWeighted(img_rs, 0.5, geo_overlay, 0.5, 0, geo_overlay);

                //cv::imshow("TwinLiteNet+ TensorRT", net_overlay);
		//cv::imshow("ZED Depth Geometry", geo_overlay);
                
                if (cv::waitKey(1) == 'q') break;
            }

            auto t4 = std::chrono::high_resolution_clock::now();
            profiler.update("4. PostProc", std::chrono::duration<float, std::milli>(t4 - t3).count());

            profiler.print_summary();
        }
    }

    zed.close();
    cv::destroyAllWindows();
    return 0;
}
