#pragma once

#include <map>
#include <cmath>
#include <stack>
#include <future>
#include <chrono>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <limits>
#include <fstream>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <string_view>
#include <unordered_map>

#include <cuda.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_launch_parameters.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include "NvInfer.h"



using Severity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger {
public:

    void setSeverityThreshold(Severity severity) noexcept;           

    void log(Severity severity, const char* msg) noexcept override;   
    void log(Severity severity, const std::string& msg) noexcept; 

    void error(const std::string& msg) noexcept;      // error wrapper for clarity

private:

    Severity severityThreshold = Severity::kINFO;

};

extern Logger globalLogger;     // Global logger instance


class Timer {
public:

    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::micro>;

    Timer(const std::string& label);
    ~Timer();

    double stop();

    static int lastTimeUS;

#ifdef _DEBUG
    static double getAverageUS(const std::string& label);
#endif

private:
#ifdef _DEBUG
    static std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> runningAvg;
#endif

    std::string label;
    TimePoint start;

    Duration elapsed;
    bool stopped;

};