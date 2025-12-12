#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>

namespace onnx_runner {
namespace kernels {

// CUDA kernel for Sigmoid activation: f(x) = 1 / (1 + exp(-x))
__global__ void sigmoidKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

// Vectorized version using float4 for better memory throughput
__global__ void sigmoidKernelVectorized(const float* input, float* output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float4 val = reinterpret_cast<const float4*>(input)[idx / 4];
        val.x = 1.0f / (1.0f + expf(-val.x));
        val.y = 1.0f / (1.0f + expf(-val.y));
        val.z = 1.0f / (1.0f + expf(-val.z));
        val.w = 1.0f / (1.0f + expf(-val.w));
        reinterpret_cast<float4*>(output)[idx / 4] = val;
    } else if (idx < size) {
        // Handle remaining elements
        for (int i = idx; i < size; ++i) {
            float x = input[i];
            output[i] = 1.0f / (1.0f + expf(-x));
        }
    }
}

// CPU implementation (single-threaded)


// Launcher function
void launchSigmoid(const float* input, float* output, int size, cudaStream_t stream) {
    // Use vectorized kernel if size is divisible by 4, otherwise use regular kernel
    if (size % 4 == 0 && reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(output) % 16 == 0) {
        int block_size = 256;
        int grid_size = ((size / 4) + block_size - 1) / block_size;
        sigmoidKernelVectorized<<<grid_size, block_size, 0, stream>>>(input, output, size);
    } else {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        sigmoidKernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
    }
}

} // namespace kernels
} // namespace onnx_runner
