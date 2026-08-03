#pragma once

#include "utils.hpp"


namespace device {

	bool queuePreprocess(const sl::Mat& leftBGR, float* output, cv::Size outputSize, std::uint32_t numChannels, cudaStream_t stream);
	bool queuePostprocess(
		const float* __restrict__ llInput, const float* __restrict__ daInput,
		float* __restrict__ llOutput, float* __restrict__ daOutput,
		cv::Size size, cudaStream_t stream
	);


	namespace _kernels {

		__global__ void preprocessKernel(
			const uchar* __restrict__ inputBGR, std::uint32_t inputW, std::uint32_t inputH, std::uint32_t inputStep,
			float* __restrict__ output, std::uint32_t outputW, std::uint32_t outputH
		);


		// Postprocessing

		__global__ void channelConvertKernel(
			const float* __restrict__ llInput, const float* __restrict__ daInput,
			float* __restrict__ llOutput, float* __restrict__ daOutput,
			std::uint32_t width, std::uint32_t height
		);

		__global__ void extractScaledXZKernel(
			const float* __restrict__ depthXYZ,
			float* __restrict__ scaledX, float* __restrict__ scaledZ,
			std::uint32_t inputW, std::uint32_t intputH, std::uint32_t outputW, std::uint32_t outputH
		);

		__global__ void computeDepthMaskKernel(
			bool* __restrict__ depthMask, 
			float* __restrict__ scaledX, float* __restrict__ scaledZ,
			std::int32_t width, std::int32_t height,
			float edgePercent, float maxDepthCutoff
		);

	}



}