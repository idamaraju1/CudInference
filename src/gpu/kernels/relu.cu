#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace onnx_runner {
namespace kernels {

// ReLU activation: y = max(0, x)
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Vectorized ReLU using float4 for better memory bandwidth
__global__ void relu_vectorized_kernel(const float* input, float* output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float4 val = *reinterpret_cast<const float4*>(&input[idx]);
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        *reinterpret_cast<float4*>(&output[idx]) = val;
    } else if (idx < size) {
        // Handle remaining elements
        for (int i = idx; i < size; ++i) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }
}

void launchReLU(const float* input, float* output, int size, cudaStream_t stream) {
    // Choose kernel based on size and alignment
    if (size >= 1024 && (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(output) % 16 == 0)) {
        // Use vectorized kernel for large, aligned arrays
        int blockSize = 256;
        int gridSize = (size + blockSize * 4 - 1) / (blockSize * 4);
        relu_vectorized_kernel<<<gridSize, blockSize, 0, stream>>>(input, output, size);
    } else {
        // Use simple kernel for small or unaligned arrays
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        relu_kernel<<<gridSize, blockSize, 0, stream>>>(input, output, size);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ReLU kernel launch failed: ") +
                               cudaGetErrorString(error));
    }
}

// In-place version
void launchReLUInPlace(float* data, int size, cudaStream_t stream) {
    launchReLU(data, data, size, stream);
}

// CPU fallback
void reluCPU(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// Multi-threaded CPU implementation using OpenMP
void reluCPUMultiThreaded(const float* input, float* output, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

} // namespace kernels
} // namespace onnx_runner
