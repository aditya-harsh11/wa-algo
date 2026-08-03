#include "kernel.cuh"

bool device::queuePreprocess(const sl::Mat& leftBGR, float* output, cv::Size outputSize, std::uint32_t numChannels, cudaStream_t stream) {
    if (outputSize.width <= 0 || outputSize.height <= 0) {
        globalLogger.error("Cannot queue preprocessing with non-positive dimensions");
        return false;
    }

    constexpr dim3 blockSize(16, 16);
    const dim3 gridSize(
        (outputSize.width + blockSize.x - 1) / blockSize.x, 
        (outputSize.height + blockSize.y - 1) / blockSize.y
    );

    _kernels::preprocessKernel<<<gridSize, blockSize, 0, stream>>>(
        (uchar*)leftBGR.getPtr<sl::uchar1>(sl::MEM::GPU),
        leftBGR.getStepBytes(sl::MEM::GPU),
        numChannels,
        output,
        outputSize.width, outputSize.height
    );

    return true;
}

bool device::queuePostprocess(
    const float* __restrict__ llInput, const float* __restrict__ daInput, 
    float* __restrict__ llOutput, float* __restrict__ daOutput, 
    cv::Size size, cudaStream_t stream) {

    constexpr dim3 blockSize(16, 16);
    const dim3 gridSize(
        (size.width + blockSize.x - 1) / blockSize.x,
        (size.height + blockSize.y - 1) / blockSize.y
    );

    _kernels::channelConvertKernel<<<gridSize, blockSize, 0, stream>>>(
        llInput, daInput,
        llOutput, daOutput,
        size.width, size.height
    );

    // TODO: more detailed filtering / postprocessing, but good enough for now

    return true;
}


__global__ void device::_kernels::preprocessKernel(
    const uchar* __restrict__ inputBGR, std::uint32_t inputW, std::uint32_t inputH, std::uint32_t inputStep,
    float* __restrict__ output, std::uint32_t outputW, std::uint32_t outputH) {

    const std::uint32_t outputX = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t outputY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outputX >= outputW || outputY >= outputH) return;

    const float scaleX = __fdividef((float)inputW, (float)outputW);
    const float scaleY = __fdividef((float)inputW, (float)outputH);

    float srcX = ((float)outputX + 0.5f) * scaleX - 0.5f;
    float srcY = ((float)outputY + 0.5f) * scaleY - 0.5f;

    srcX = (srcX < 0.0f ? 0.0f : (srcX > (float)(inputW - 1) ? (float)(inputW - 1) : srcX));  //clamp
    srcY = (srcY < 0.0f ? 0.0f : (srcY > (float)(inputH - 1) ? (float)(inputH - 1) : srcY));  //clamp

    const std::int32_t x0 = (std::int32_t)floorf(srcX);
    const std::int32_t y0 = (std::int32_t)floorf(srcY);
    const std::int32_t x1 = min(x0 + 1, inputW - 1);
    const std::int32_t y1 = min(y0 + 1, inputH - 1);

    const float weightX = srcX - (float)x0;
    const float weightY = srcY - (float)y0;

    // TODO: determine if it's feasible / worthwhile here to implement shared memory tiling / loading here for speedup
    // each thread should 

    const uchar3 p00 = *(uchar3*)(inputBGR + y0 * inputStep + x0 * 3);
    const uchar3 p10 = *(uchar3*)(inputBGR + y0 * inputStep + x1 * 3);
    const uchar3 p01 = *(uchar3*)(inputBGR + y1 * inputStep + x0 * 3);
    const uchar3 p11 = *(uchar3*)(inputBGR + y1 * inputStep + x1 * 3);

    const std::uint32_t outIndex = outputY * outputW + outputX;
    const std::uint32_t planeStep = outputW * outputH;

    float b = (1.0f - weightX) * (1.0f - weightY) * (float)p00.x
        + weightX * (1.0f - weightY) * (float)p10.x
        + (1.0f - weightX) * weightY * (float)p01.x
        + weightX * weightY * (float)p11.x;
    
    float g = (1.0f - weightX) * (1.0f - weightY) * (float)p00.y
        + weightX * (1.0f - weightY) * (float)p10.y
        + (1.0f - weightX) * weightY * (float)p01.y
        + weightX * weightY * (float)p11.y;

    float r = (1.0f - weightX) * (1.0f - weightY) * (float)p00.z
        + weightX * (1.0f - weightY) * (float)p10.z
        + (1.0f - weightX) * weightY * (float)p01.z
        + weightX * weightY * (float)p11.z;
    
    output[0 * planeStep + outIndex] = r / 255.0f;
    output[1 * planeStep + outIndex] = g / 255.0f;
    output[2 * planeStep + outIndex] = b / 255.0f;
}

__global__ void device::_kernels::channelConvertKernel(
    const float* __restrict__ llInput, const float* __restrict__ daInput,
    float* __restrict__ llOutput, float* __restrict__ daOutput,
    std::uint32_t width, std::uint32_t height) {

    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    
    const std::uint32_t inputBaseIndex = y * width + x;
    const std::uint32_t outputBaseIndex = 2 * (y * width + x);
    const std::uint32_t planeStep = width * height;

    //CHW -> HWC w/ C=2

    llOutput[outputBaseIndex + 0] = llInput[0 * planeStep + inputBaseIndex];
    llOutput[outputBaseIndex + 1] = llInput[1 * planeStep + inputBaseIndex];

    daOutput[outputBaseIndex + 0] = daInput[0 * planeStep + inputBaseIndex];
    daOutput[outputBaseIndex + 1] = daInput[1 * planeStep + inputBaseIndex];
}


__global__ void device::_kernels::extractScaledXZKernel(
    const float* __restrict__ depthXYZ,
    float* __restrict__ scaledX, float* __restrict__ scaledZ,
    std::uint32_t inputW, std::uint32_t intputH, std::uint32_t outputW, std::uint32_t outputH) {

    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outputW || y >= outputH) return;

    std::uint32_t inX = min((std::int32_t)(x * (float)inputW / outputW), inputW - 1);
    std::uint32_t inY = min((std::int32_t)(y * (float)intputH / outputH), intputH - 1);

    std::uint32_t inIndex = (inY * inputW + inX) * 3;
    std::uint32_t outIndex = y * outputW + x;

    scaledX[outIndex] = depthXYZ[inIndex + 0];
    scaledZ[outIndex] = depthXYZ[inIndex + 2];
}

__global__ void device::_kernels::computeDepthMaskKernel(
    bool* __restrict__ depthMask, 
    float* __restrict__ scaledX, float* __restrict__ scaledZ, 
    std::int32_t width, std::int32_t height, 
    float edgePercent, float maxDepthCutoff) {

    std::int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    std::int32_t edgeCutoffX = (std::int32_t)(width * 0.08f);
    bool validFOV = (x >= edgeCutoffX) && (x < (width - edgeCutoffX));

    std::int32_t index = y * width + x;
    float valX = scaledX[index];
    float valZ = scaledZ[index];

    depthMask[index] = isfinite(valZ) && validFOV && (valX < maxDepthCutoff);
}

// MORE...


