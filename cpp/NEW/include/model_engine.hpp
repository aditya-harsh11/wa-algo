#pragma once

#include "utils.hpp"
#include "kernel.cuh"




namespace lane_seg {

    class TwinLite {
    public:

        struct Configuration {
            cv::Size inputSize = cv::Size(640, 360);   // as (width, height)
            cv::Size outputSize = cv::Size(640, 360);

            std::uint32_t inputChannels = 3;
        };


        TwinLite(const std::filesystem::path& modelPath, const Configuration& config);
        ~TwinLite();

        TwinLite(const TwinLite&) = delete;
        TwinLite& operator=(const TwinLite&) = delete;

        TwinLite(TwinLite&&) = delete;
        TwinLite& operator=(TwinLite&&) = delete;

        bool predict(
            const sl::Mat& frameBGR, const sl::Mat& depthXYZ,
            std::vector<float>& llScores, std::vector<float>& daScores,
            bool computeMasks
        );

        bool isLoaded() const;

        constexpr const std::filesystem::path& getFilePath() const;
        constexpr const Configuration& getConfig() const;

        constexpr const cv::Mat& getGroundMask() const;
        constexpr const cv::Mat& getObstacleMask() const;

    private:

        const std::filesystem::path path;
        const Configuration config;

        std::unique_ptr<nvinfer1::IRuntime> runtime;   
        std::unique_ptr<nvinfer1::ICudaEngine> engine;  
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        cudaStream_t stream, altStream;       
        cudaEvent_t preprocessEnd, inferenceEnd;             // cuda events for internal sync

        using TensorLocation = nvinfer1::TensorLocation;
        struct TensorMem {
            void* data;
            std::size_t size;
            
            TensorLocation location;
        
        };

        std::vector<TensorMem> engineAllocs;
        std::int32_t inputIndex, llIndex, daIndex; // "da" = drivable area, "ll" = lane lines

        TensorMem llConvertBuffer, daConvertBuffer;
        cv::Mat groundMask, obstacleMask;

        bool loadEngine(const std::filesystem::path& modelPath);
        bool allocate();

        bool preprocess(const sl::Mat& leftBGR);
        bool runEngine();                    
        bool postprocess(
            const sl::Mat& frameBGR, const sl::Mat& depthXYZ,
            std::vector<float>& llScores, std::vector<float>& daScores,
            bool computeMasks
        );


        void applyGeometricPrior(std::vector<float> daScores, const sl::Mat& depthXYZ);

    };

}