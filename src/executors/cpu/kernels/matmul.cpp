#include "cpu_kernels.h"

namespace onnx_runner {
namespace kernels {

// Simple CPU matrix multiplication
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
    #pragma omp parallel for num_threads(num_threads)
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
