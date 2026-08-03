#include "utils.hpp"

void Logger::setSeverityThreshold(Severity severity) noexcept {
    severityThreshold = severity;
}

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity > severityThreshold) return;

    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::fprintf(stderr, "\033[1;31m[lane-seg] [INTERNAL ERROR] %s\033[0m\n", msg);
            break;
        case Severity::kERROR:
            std::fprintf(stderr, "\033[1;31m[lane-seg] [ERROR] %s\033[0m\n", msg);
            break;
        case Severity::kWARNING:
            std::printf("\033[1;33m[lane-seg] [WARNING] %s\033[0m\n", msg);
            break;
        case Severity::kINFO:
            std::printf("[lane-seg] [INFO] %s\n", msg);
            break;
        case Severity::kVERBOSE:
            std::printf("[lane-seg] [VERBOSE] %s\n", msg);
            break;
        default:
            std::printf("[lane-seg] [UNKNOWN] %s\n", msg);
            break;
    }
}

void Logger::log(Severity severity, const std::string& msg) noexcept {
    log(severity, msg.c_str());
}

void Logger::error(const std::string& msg) noexcept {
    log(Severity::kERROR, msg.c_str());
}

Logger globalLogger; // Global logger instance for TensorRT


Timer::Timer(const std::string& label) : label(label), stopped(false), elapsed(0) {
    start = Clock::now();
}

Timer::~Timer() {
    if (stopped) return;

    stop();
    lastTimeUS = (int)elapsed.count();
#ifdef _DEBUG
    if (runningAvg.find(label) == runningAvg.end()) {
        runningAvg[label] = std::make_pair((std::size_t)elapsed.count(), 1);
    }
    runningAvg[label].first += (std::size_t)elapsed.count();
    runningAvg[label].second++;
#endif
}

double Timer::stop() {
    TimePoint end = Clock::now();
    if (stopped) {
        return elapsed.count();
    }

    stopped = true;
    elapsed = std::chrono::duration_cast<Duration>(end - start);

    globalLogger.log(Severity::kINFO, "[" + label + "]: completed in " + std::to_string(elapsed.count()) + "us");
    return elapsed.count();
}

#ifdef _DEBUG
double Timer::getAverageUS(const std::string& label) {
    try {
        auto avg = runningAvg.at(label);
        return (double)(avg.first) / (double)(avg.second);
    } catch (const std::out_of_range& e) {
        return -1.0;
    }
}

std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> Timer::runningAvg = {};
#endif


int Timer::lastTimeUS = 0;