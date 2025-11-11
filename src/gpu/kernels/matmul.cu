#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace onnx_runner {
namespace kernels {

// Simple matrix multiplication kernel (for small matrices or as fallback)
// C = A @ B where A is (M, K) and B is (K, N), C is (M, N)
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication for better performance
// Uses shared memory to reduce global memory accesses
template<int TILE_SIZE>
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launchMatMul(const float* A, const float* B, float* C,
                  int M, int K, int N, cudaStream_t stream) {
    // For larger matrices, use cuBLAS for optimal performance
    // For smaller matrices, use our custom kernel

    const int TILE_SIZE = 16;
    const int USE_CUBLAS_THRESHOLD = 128;

    if (M >= USE_CUBLAS_THRESHOLD || N >= USE_CUBLAS_THRESHOLD || K >= USE_CUBLAS_THRESHOLD) {
        // Use cuBLAS for large matrices
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // cuBLAS uses column-major, so we compute: C^T = B^T @ A^T
        // Which gives us C = A @ B in row-major
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    B, N,
                    A, K,
                    &beta,
                    C, N);

        cublasDestroy(handle);
    } else {
        // Use custom tiled kernel for smaller matrices
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

        matmul_tiled_kernel<16><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("MatMul kernel launch failed: ") +
                               cudaGetErrorString(error));
    }
}

// CPU fallback for debugging
void matmulCPU(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Multi-threaded CPU implementation using OpenMP
void matmulCPUMultiThreaded(const float* A, const float* B, float* C, int M, int K, int N, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

} // namespace kernels
} // namespace onnx_runner
