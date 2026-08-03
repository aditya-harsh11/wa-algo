#include <thread>
#include <chrono>
#include <string>
#include <utility>
#include <filesystem>
#include <unordered_map>

#include "utils.hpp"
#include "segmentation.hpp"
#include "model_engine.hpp"

#include <sl/Camera.hpp>
#include <argparse.hpp>


bool startup(int argc, char* argv[]) {
    std::string ipAddress, svoPath;
    std::string modelPath;
    int debugMode;
    
    argparse::ArgumentParser program("lane-seg", __DATE__);
    program.add_argument("--ip_address")
        .help("IP Address for ZED input, in format \"a.b.c.d:port\"")
        .default_value("127.0.0.1:30000")
        .store_into(ipAddress);
    program.add_argument("--model")
        .help("File path of first (main) TensorRT model")
        .default_value("weights/twinlitenet.engine")
        .store_into(modelPath);
    program.add_argument("--debug_mode")
        .help("NONE (0 - default), VISUALIZE (1 - requires display), or VIDEO_OUT (2)")
        .default_value(0)
        .scan<'i', int>()
        .store_into(debugMode);
    program.add_argument("--svo_path")
        .help("Path to SVO file to OVERRIDE live camera (for DEBUG/TESTING)")
        .default_value("")
		.store_into(svoPath);

    // FOR DEBUGGING, REMOVE LATER
#ifdef WIN32
    debugMode  = 1;
	modelPath = "C:/Users/djb11/source/repos/wa-lane-seg/weights/twinlitenet.engine";
	svoPath    = "C:/Users/djb11/Downloads/regent1_x.svo2";
    ipAddress  = "";
#endif

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& e) {
        globalLogger.error("Lane segmentation failed to parse args! " + std::string(e.what()));
        return false;
    }

    globalLogger.log(Severity::kINFO, "Initializing TwinLite-based lane segmentation...");   
    globalLogger.log(Severity::kINFO, "Built on " + std::string(__DATE__));

    switch (debugMode) {
        case 0:
            globalLogger.setSeverityThreshold(Severity::kINFO);
            globalLogger.log(Severity::kINFO, "Debug mode set to NONE");
            break;
        case 1:
			globalLogger.setSeverityThreshold(Severity::kVERBOSE);
            globalLogger.log(Severity::kINFO, "Debug mode set to VISUALIZE");
            break;
        case 2:
            globalLogger.setSeverityThreshold(Severity::kVERBOSE);
            globalLogger.log(Severity::kINFO, "Debug mode set to VIDEO_OUT");
            break;
        default:
            globalLogger.setSeverityThreshold(Severity::kINFO);
            globalLogger.log(Severity::kWARNING, "Invalid debug mode specified, defaulting to NONE");
            debugMode = 0;
            break;
	}

    auto model = std::make_shared<lane_seg::TwinLite>(modelPath);
    if(!model->isLoaded()) {
        globalLogger.error("Model(s) failed to load!");
        return -1;
	}

    lane_seg::Segmenter segmenter(model, ipAddress, svoPath, (lane_seg::DebugMode)debugMode);

    bool success = segmenter.runLoop();
    if (!success) {
        globalLogger.error("Segmentation loop exited with failure!");
        return false;
    }

#ifdef _DEBUG
    globalLogger.log(Severity::kVERBOSE, "Frame time avg.: " + std::to_string(Timer::getAverageUS("Frame Time")) + "us");
#endif

    return true;
}

int main(int argc, char* argv[]) {
    if (!startup(argc, argv)) {
        globalLogger.error("Lane segmentation exiting WITH failure");
        return -1;
    }

    globalLogger.log(Severity::kINFO, "Lane segmentation exiting WITHOUT failure");
    return 1;
}