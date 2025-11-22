#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace onnx_runner {
namespace kernels {

// ============================================================================
// Element-wise subtraction: C = A - B
// ============================================================================

__global__ void sub_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

// Vectorized subtraction using float4
__global__ void sub_vectorized_kernel(const float* A, const float* B, float* C, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx]);
        float4 b = *reinterpret_cast<const float4*>(&B[idx]);
        float4 c;
        c.x = a.x - b.x;
        c.y = a.y - b.y;
        c.z = a.z - b.z;
        c.w = a.w - b.w;
        *reinterpret_cast<float4*>(&C[idx]) = c;
    } else if (idx < size) {
        // Handle remaining elements
        for (int i = idx; i < size; ++i) {
            C[i] = A[i] - B[i];
        }
    }
}

// Sub with scalar broadcasting: C = A - scalar
__global__ void sub_scalar_kernel(const float* A, float scalar, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] - scalar;
    }
}

// ============================================================================
// GPU Launchers
// ============================================================================

void launchSub(const float* A, const float* B, float* C, int size, cudaStream_t stream) {
    if (size <= 0) return;

    // Choose kernel based on size and alignment
    if (size >= 1024 && (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0)) {
        int blockSize = 256;
        int gridSize = (size + blockSize * 4 - 1) / (blockSize * 4);
        sub_vectorized_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, size);
    } else {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        sub_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, size);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("Sub kernel launch failed: ") +
                                 cudaGetErrorString(error));
    }
}

void launchSubScalar(const float* A, float scalar, float* C, int size, cudaStream_t stream) {
    if (size <= 0) return;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    sub_scalar_kernel<<<gridSize, blockSize, 0, stream>>>(A, scalar, C, size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("SubScalar kernel launch failed: ") +
                                 cudaGetErrorString(error));
    }
}

} // namespace kernels
} // namespace onnx_runner

