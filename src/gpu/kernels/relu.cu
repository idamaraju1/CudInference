#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace onnx_runner {
namespace kernels {

// ReLU activation: y = max(0, x)
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Vectorized ReLU using float4 for better memory bandwidth
__global__ void relu_vectorized_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int size) {
    constexpr int VEC = 4;
    int idx_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int n_vec   = size / VEC;           // number of full float4s

    if (idx_vec < n_vec) {
        const float4* in4  = reinterpret_cast<const float4*>(input);
        float4*       out4 = reinterpret_cast<float4*>(output);

        float4 val = in4[idx_vec];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        out4[idx_vec] = val;
    }

    // Handle remaining (size % 4) elements with a single thread
    if (idx_vec == 0) {
        int tail_start = n_vec * VEC;
        for (int i = tail_start; i < size; ++i) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }
}
void launchReLU(const float* input, float* output, int size, cudaStream_t stream) {
    // return early on empty.
    if (size <= 0) return;
    // small helper for alignment checks.
    auto is_aligned_16 = [](const void* p) {
        return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
    };
    // Choose kernel based on size and alignment
    if (size >= 1024 && (is_aligned_16(input) % 16 == 0) &&
        (is_aligned_16(output) % 16 == 0)) {
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
