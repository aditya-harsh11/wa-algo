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
#include <NvOnnxParser.h>

using namespace nvinfer1;

// --- MEMORY HELPER ---

// --- CONFIGURATION ---
const int INPUT_W = 640;
const int INPUT_H = 384;
const int OUTPUT_W = 640;
const int OUTPUT_H = 384;
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

void build_engine_from_onnx(const std::string& engine_path) {
    std::string onnx_path = "../../../python/weights/twinlitenetplus_large.onnx"; 

    auto builder = createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file: " << onnx_path << std::endl;
        exit(1);
    }

    auto config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); 
    
    if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        std::cout << "Building with FP16 precision..." << std::endl;
    }

    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    std::ofstream p(engine_path, std::ios::binary);
    p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    p.close();

    delete serializedModel;
    delete config;
    delete parser;
    delete network;
    delete builder;
    std::cout << "Engine build complete: " << engine_path << std::endl;
}

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

        // check if engine exists
        std::ifstream f(engine_path.c_str());
        if (!f.good()) {
            std::cout << "Engine not found at " << engine_path << ". Building it now (this takes a few mins)..." << std::endl;
            build_engine_from_onnx(engine_path);
        }

        // Load Engine
        std::cout << "Attempting to load engine: " << engine_path << std::endl;
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.good()) {
            std::cerr << "FATAL ERROR: Could not find engine file at: " << engine_path << std::endl;
            std::cerr << "Please ensure you have run the engine generation in Python first!" << std::endl;
            exit(EXIT_FAILURE);
        }
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
        // 1. Preprocess: Resize -> Pad -> BGR2RGB -> Normalize -> NCHW
        cv::Mat resized, padded, rgb, float_img;
        
        // Resize to a perfect 16:9 ratio (640x360)
        cv::resize(image, resized, cv::Size(640, 360)); 
        
        // Pad 12 pixels top and bottom to hit the 384 network requirement
        cv::copyMakeBorder(resized, padded, 12, 12, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        
        // Proceed with standard conversions on the padded 640x384 image
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
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

// --- ZED SETUP (RGB ONLY) ---
void setup_zed_camera(sl::Camera& zed, const std::string& ip_address, const std::string& svo_path) {
    sl::InitParameters init;
    init.camera_resolution = sl::RESOLUTION::HD1080;
    init.depth_mode = sl::DEPTH_MODE::NEURAL; 
    init.coordinate_units = sl::UNIT::METER; 
    init.camera_fps = 30;

    if (!svo_path.empty()) {
        init.input.setFromSVOFile(sl::String(svo_path.c_str()));
        init.svo_real_time_mode = true;
        std::cout << "ZED: Opening SVO File -> " << svo_path << std::endl;
    } else if (!ip_address.empty()) {
        std::cout << "ZED: Connecting to IP -> " << ip_address << std::endl;
        std::string ip = ip_address.substr(0, ip_address.find(':'));
        int port = std::stoi(ip_address.substr(ip_address.find(':') + 1));
        init.input.setFromStream(sl::String(ip.c_str()), port);
    }

    sl::ERROR_CODE status = zed.open(init);
    if (status != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to open ZED: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    std::string size = "large";
    if(argc > 1) {
        size = std::string(argv[1]);   
    }
    std::string engine_file = "../../../python/weights/twinlitenetplus_"+size+"_cpp.engine";
    std::string ip_address = "";
    std::string svo_path = "";
    bool debug_mode = true;
    
    if(argc > 2) {
        std::string arg2(argv[2]);
        if(arg2 == "false") debug_mode = false;
        else if(arg2 == "true") debug_mode = true;
        else if(arg2.find(".svo") != std::string::npos) svo_path = arg2;
        else ip_address = arg2;
    }

    if(argc > 3) {
        std::string arg3(argv[3]);
        if(arg3.find(".svo") != std::string::npos) svo_path = arg3;
        else ip_address = arg3;
    }

    TwinLiteTRTEngine model(engine_file);
    
    sl::Camera zed;
    setup_zed_camera(zed, ip_address, svo_path);
    
    Profiler profiler;
    sl::RuntimeParameters runtime_parameters;
    sl::Mat left_image;
    sl::Mat point_cloud;

    std::cout << "Starting C++ TensorRT Inference (RGB Only)...\n";

    uint frame_count = 0;
    while (true) {
        if (zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS) {
            
            // 1. ZED Retrieval
            auto t0 = std::chrono::high_resolution_clock::now();
            zed.retrieveImage(left_image, sl::VIEW::LEFT, sl::MEM::CPU, sl::Resolution(640, 360));
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU, sl::Resolution(640, 360));
            
            cv::Mat ip_frame(left_image.getHeight(), left_image.getWidth(), CV_8UC4, left_image.getPtr<sl::uchar1>(sl::MEM::CPU));
            cv::cvtColor(ip_frame, ip_frame, cv::COLOR_BGRA2BGR);

            auto t1 = std::chrono::high_resolution_clock::now();
            profiler.update("1. Retrieve", std::chrono::duration<float, std::milli>(t1 - t0).count());

            // 2. TRT Inference
            model.infer(ip_frame);
            float* da_scores = static_cast<float*>(model.allocations["da_out"].host);
            float* ll_scores = static_cast<float*>(model.allocations["ll_out"].host);

            auto t2 = std::chrono::high_resolution_clock::now();
            profiler.update("2. TensorRT", std::chrono::duration<float, std::milli>(t2 - t1).count());

            int plane_size = OUTPUT_H * OUTPUT_W;
            int PADDING = 12; 
            int search_bottom_y = OUTPUT_H - 1 - PADDING;
            int search_top_y = 100 + PADDING; 
            int NUM_POINTS_PER_SIDE = 10; 
            int y_sample_interval = (search_bottom_y - search_top_y) / NUM_POINTS_PER_SIDE;

            std::vector<cv::Point> left_bounds;
            std::vector<cv::Point> right_bounds;

            for (int y = search_bottom_y; y > search_top_y; --y) {
                int left_x = 0;
                int right_x = OUTPUT_W - 1;
                while (left_x < OUTPUT_W && da_scores[1 * plane_size + (y * OUTPUT_W + left_x)] <= da_scores[0 * plane_size + (y * OUTPUT_W + left_x)]) { left_x++; }
                while (right_x > 0 && da_scores[1 * plane_size + (y * OUTPUT_W + right_x)] <= da_scores[0 * plane_size + (y * OUTPUT_W + right_x)]) { right_x--; }

                if ((search_bottom_y - y) % y_sample_interval == 0 && left_bounds.size() < NUM_POINTS_PER_SIDE) {
                    int display_y = y - PADDING;
                    left_bounds.push_back(cv::Point(left_x, display_y));
                    right_bounds.push_back(cv::Point(right_x, display_y));
                }
            }

            cv::Mat ll_mask = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_8UC1);
            cv::Mat da_mask = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_8UC1);

            for (int y = search_top_y; y <= search_bottom_y; ++y) {
                for (int x = 0; x < OUTPUT_W; ++x) {
                    int idx = y * OUTPUT_W + x;
                    if (ll_scores[1 * plane_size + idx] > ll_scores[0 * plane_size + idx]) {
                        ll_mask.at<uchar>(y, x) = 255;
                    }
                    if (da_scores[1 * plane_size + idx] > da_scores[0 * plane_size + idx]) {
                        da_mask.at<uchar>(y, x) = 255;
                    }
                }
            }

            // --- ROBUST CONTOUR-BASED STOP BAR DETECTION ---
            int stop_bar_row = -1;
            int stop_bar_center_x = -1;
            
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(ll_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            double max_y = -1;
            for (const auto& contour : contours) {
                cv::Rect rect = cv::boundingRect(contour);
                double aspect_ratio = (double)rect.width / rect.height;
                double width_ratio = (double)rect.width / OUTPUT_W;
                
                // Filter for horizontal-ish blobs that are wide enough
                if (aspect_ratio > 3.0 && width_ratio > 0.15) {
                    // Validate if it's within the drivable area
                    // Check middle point of the bounding box
                    int mid_x = rect.x + rect.width / 2;
                    int mid_y = rect.y + rect.height / 2;
                    
                    if (da_mask.at<uchar>(mid_y, mid_x) > 0) {
                        // Pick the one closest to the vehicle (highest y)
                        if (rect.y + rect.height > max_y) {
                            max_y = rect.y + rect.height;
                            stop_bar_row = mid_y;
                            stop_bar_center_x = mid_x;
                        }
                    }
                }
            }

            std::vector<cv::Point> ego_polygon = left_bounds;
            for (int i = right_bounds.size() - 1; i >= 0; --i) {
                ego_polygon.push_back(right_bounds[i]);
            }

            // --- 3D DATA EXTRACTION ---
            sl::float4 stopbar_point3D; 
            stopbar_point3D.z = INFINITY; 
            
            if (stop_bar_row != -1) {
                int display_y = stop_bar_row - PADDING;
                if (display_y >= 0 && display_y < 360) {
                    point_cloud.getValue(stop_bar_center_x, display_y, &stopbar_point3D);
                }
            }

            std::vector<sl::float3> polygon_3d;
            for (const auto& pt2D : ego_polygon) {
                sl::float4 point3D;
                point_cloud.getValue(pt2D.x, pt2D.y, &point3D);
                if (std::isfinite(point3D.z)) {
                    polygon_3d.push_back(sl::float3(point3D.x, point3D.y, point3D.z));
                }
            }

            // --- OUTPUT DATA FOR ROS NODE TO PARSE ---
            std::stringstream data_out;
            data_out << "TWINLITE_DATA: STOPBAR ";
            if (std::isfinite(stopbar_point3D.z)) {
                data_out << stopbar_point3D.x << " " << stopbar_point3D.y << " " << stopbar_point3D.z << " ";
            } else {
                data_out << "NaN NaN NaN ";
            }
            data_out << "POLYGON ";
            for (const auto& pt3D : polygon_3d) {
                data_out << pt3D.x << " " << pt3D.y << " " << pt3D.z << " ";
            }
            // Print and flush so the ROS node receives it instantly
            std::cout << data_out.str() << std::endl;

            // --- Visualization (Only when debug_mode is true) ---
            if (debug_mode) {
                cv::Mat img_rs;
                cv::resize(ip_frame, img_rs, cv::Size(640, 360)); 
                cv::Mat net_overlay = img_rs.clone();
                
                for (int y = 0; y < 360; ++y) {
                    for (int x = 0; x < 640; ++x) {
                        int trt_idx = (y + 12) * OUTPUT_W + x; 
                        if (da_scores[1 * plane_size + trt_idx] > da_scores[0 * plane_size + trt_idx]) {
                            net_overlay.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0); 
                        }
                        if (ll_scores[1 * plane_size + trt_idx] > ll_scores[0 * plane_size + trt_idx]) {
                            net_overlay.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); 
                        }
                    }
                }

                if (ego_polygon.size() > 0) {
                    std::vector<std::vector<cv::Point>> polys = {ego_polygon};
                    cv::polylines(net_overlay, polys, true, cv::Scalar(0, 255, 255), 2); 
                }

                if (stop_bar_row != -1) {
                    int display_y = stop_bar_row - PADDING;
                    if (display_y >= 0 && display_y < 360) {
                        cv::line(net_overlay, cv::Point(0, display_y), cv::Point(640, display_y), cv::Scalar(0, 0, 255), 3);                                     
                    }
                }

                cv::addWeighted(img_rs, 0.5, net_overlay, 0.5, 0, net_overlay);
                cv::imshow("TwinLiteNet+ TensorRT", net_overlay);

                cv::Mat bev = cv::Mat::zeros(500, 500, CV_8UC3);
                float scale = 15.0f; 
                int origin_x = 250; 
                int origin_y = 450; 

                cv::circle(bev, cv::Point(origin_x, origin_y), 5, cv::Scalar(255, 255, 255), -1);

                std::vector<cv::Point> bev_poly;
                for (const auto& pt3D : polygon_3d) {
                    int bev_x = origin_x + (int)(pt3D.x * scale);
                    int bev_y = origin_y - (int)(pt3D.z * scale);
                    if (bev_y > 0 && bev_y < 500 && bev_x > 0 && bev_x < 500) {
                        bev_poly.push_back(cv::Point(bev_x, bev_y));
                    }
                }

                if (!bev_poly.empty()) {
                    std::vector<std::vector<cv::Point>> polys = {bev_poly};
                    cv::polylines(bev, polys, true, cv::Scalar(0, 255, 255), 2); 
                }

                if (stop_bar_row != -1 && std::isfinite(stopbar_point3D.z)) {
                    int sb_x = origin_x + (int)(stopbar_point3D.x * scale);
                    int sb_y = origin_y - (int)(stopbar_point3D.z * scale);
                    
                    if (sb_y > 0 && sb_y < 500 && sb_x > 0 && sb_x < 500) {
                        cv::line(bev, cv::Point(sb_x - 30, sb_y), cv::Point(sb_x + 30, sb_y), cv::Scalar(0, 0, 255), 3);
                    }
                }
                cv::imshow("BEV Radar", bev);
                frame_count++;
                if (cv::waitKey(1) == 'q') break;
            }

            auto t3 = std::chrono::high_resolution_clock::now();
            profiler.update("3. PostProc", std::chrono::duration<float, std::milli>(t3 - t2).count());
            if(debug_mode) profiler.print_summary();
        }
    }

    zed.close();
    cv::destroyAllWindows();
    return 0;
}
