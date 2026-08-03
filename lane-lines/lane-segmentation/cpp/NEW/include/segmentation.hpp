#pragma once

#include "utils.hpp"
#include "model_engine.hpp"


namespace lane_seg {

    enum class DebugMode {
        NONE = 0,
        VISUALIZE = 1,
        VIDEO_OUT = 2
    };


    struct LaneSegment {
        // placeholder ig
    };

    struct LaneLine {
        // placeholder ig
    };



    class Segmenter {
    public:


        Segmenter() = delete;

        Segmenter(std::shared_ptr<TwinLite> model,
            std::string ipAddress, std::string svoPath,
            DebugMode debugMode);

        ~Segmenter() = default;

        Segmenter(const Segmenter&) = delete;
        Segmenter& operator=(const Segmenter&) = delete;

        Segmenter(Segmenter&&) = delete;
        Segmenter& operator=(Segmenter&&) = delete;


        bool runLoop();

        bool isValid() const;

    private:

        std::unique_ptr<sl::Camera> zed;
        sl::Resolution resolution;

        std::shared_ptr<TwinLite> model;

        bool valid;

        const std::string ipAddress, optSvoPath;

        const DebugMode debugMode;

        bool setupZED(std::size_t maxRetries = 5, double retryDelay = 0.5);
   
        void serializeSegments(const std::vector<LaneSegment>& segments, std::string& json);
        //static void inferenceThread(std::shared_ptr<YoloV11> model, const device::Mat& frame, const sl::Mat& depthMap, std::vector<Detection>& detections);
        //inline void createTracks(const std::vector<Detection>& dets, cv::Mat& frame, std::vector<TrackedObject>& tracks);

    };


    // static literals preinstantiated here for fast json serialization
    namespace json_literals {
        static const std::string S_QUOTE = "\"";
        static const std::string S_COLON = ": {";
        static const std::string S_COORDS = "\"obj_cartesian_coords\": [";
        static const std::string S_COMMA = ", ";
        static const std::string S_BRACKET_C = "]";
        static const std::string S_CLASS = ", \"obj_class\": ";
        static const std::string S_TYPE = ", \"obj_type\": ";
        static const std::string S_WIDTH = ", \"obj_width\": ";
        static const std::string S_HEIGHT = ", \"obj_height\": ";
        static const std::string S_BRACE_C = "}";
        static const std::string S_BRACE_O = "{";
        static const std::string S_255 = "255";
    }

}