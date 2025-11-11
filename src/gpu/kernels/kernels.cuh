#pragma once

#include <cuda_runtime.h>

namespace onnx_runner {
namespace kernels {

// MatMul kernel
// Computes C = A @ B where A is (M, K), B is (K, N), C is (M, N)
void launchMatMul(const float* A, const float* B, float* C,
                  int M, int K, int N, cudaStream_t stream = 0);

// CPU fallback for MatMul
void matmulCPU(const float* A, const float* B, float* C, int M, int K, int N);

// Multi-threaded CPU fallback for MatMul
void matmulCPUMultiThreaded(const float* A, const float* B, float* C, int M, int K, int N, int num_threads);

// ReLU activation
void launchReLU(const float* input, float* output, int size, cudaStream_t stream = 0);
void launchReLUInPlace(float* data, int size, cudaStream_t stream = 0);

// CPU fallback for ReLU
void reluCPU(const float* input, float* output, int size);

// Multi-threaded CPU fallback for ReLU
void reluCPUMultiThreaded(const float* input, float* output, int size, int num_threads);

// Element-wise Add
void launchAdd(const float* A, const float* B, float* C, int size, cudaStream_t stream = 0);
void launchAddScalar(const float* A, float scalar, float* C, int size, cudaStream_t stream = 0);

// CPU fallback for Add
void addCPU(const float* A, const float* B, float* C, int size);

// Multi-threaded CPU fallback for Add
void addCPUMultiThreaded(const float* A, const float* B, float* C, int size, int num_threads);

} // namespace kernels
} // namespace onnx_runner
