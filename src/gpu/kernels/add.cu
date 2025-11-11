#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace onnx_runner {
namespace kernels {

// Element-wise addition: C = A + B
// Supports broadcasting for simple cases
__global__ void add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// Vectorized addition using float4
__global__ void add_vectorized_kernel(const float* A, const float* B, float* C, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx]);
        float4 b = *reinterpret_cast<const float4*>(&B[idx]);
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        *reinterpret_cast<float4*>(&C[idx]) = c;
    } else if (idx < size) {
        // Handle remaining elements
        for (int i = idx; i < size; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

// Add with scalar broadcasting: C = A + scalar
__global__ void add_scalar_kernel(const float* A, float scalar, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] + scalar;
    }
}

void launchAdd(const float* A, const float* B, float* C, int size, cudaStream_t stream) {
    // Choose kernel based on size and alignment
    if (size >= 1024 && (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0)) {
        // Use vectorized kernel for large, aligned arrays
        int blockSize = 256;
        int gridSize = (size + blockSize * 4 - 1) / (blockSize * 4);
        add_vectorized_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, size);
    } else {
        // Use simple kernel
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        add_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, size);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("Add kernel launch failed: ") +
                               cudaGetErrorString(error));
    }
}

void launchAddScalar(const float* A, float scalar, float* C, int size, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    add_scalar_kernel<<<gridSize, blockSize, 0, stream>>>(A, scalar, C, size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("AddScalar kernel launch failed: ") +
                               cudaGetErrorString(error));
    }
}

// CPU fallback
void addCPU(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Multi-threaded CPU implementation using OpenMP
void addCPUMultiThreaded(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}

} // namespace kernels
} // namespace onnx_runner
