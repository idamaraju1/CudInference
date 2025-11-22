#pragma once

#include <vector>
#include <cstdint>

namespace onnx_runner {
namespace kernels {

// CPU fallback for MatMul
void matmulCPU(const float* A, const float* B, float* C, int M, int K, int N);

// Multi-threaded CPU fallback for MatMul
void matmulCPUMultiThreaded(const float* A, const float* B, float* C, int M, int K, int N, int num_threads);

// CPU fallback for ReLU
void reluCPU(const float* input, float* output, int size);

// Multi-threaded CPU fallback for ReLU
void reluCPUMultiThreaded(const float* input, float* output, int size, int num_threads);

// CPU fallback for Add
void addCPU(const float* A, const float* B, float* C, int size);

// Multi-threaded CPU fallback for Add
void addCPUMultiThreaded(const float* A, const float* B, float* C, int size, int num_threads);

// Multi-threaded CPU fallback for Sub
void subCPU(const float* A, const float* B, float* C, int size);
void subCPUMultiThreaded(const float* A, const float* B, float* C, int size, int num_threads);

// CPU fallbacks (single-threaded and OpenMP)
void simplifiedLayerNormCPU(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon);

void simplifiedLayerNormCPUMultiThreaded(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon, int num_threads);

} // namespace kernels

// CPU implementations
void gatherCPU(
    const float* data,
    const int64_t* indices,
    float* output,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_size,
    int num_threads
);

void mulCPU(const float* A, const float* B, float* C, int size, int num_threads);
void divCPU(const float* A, const float* B, float* C, int size, int num_threads);
void powCPU(const float* A, const float* B, float* C, int size, int num_threads);
void sqrtCPU(const float* A, float* C, int size, int num_threads);

void reduceMeanCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& axes,
    int num_threads
);

void transposeCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int>& perm,
    int num_threads
);

void sliceCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& steps,
    const std::vector<int64_t>& output_shape,
    int num_threads
);

void concatCPU(
    const std::vector<const float*>& inputs,
    float* output,
    const std::vector<std::vector<int64_t>>& input_shapes,
    int64_t axis,
    const std::vector<int64_t>& output_shape,
    int num_threads
);

} // namespace onnx_runner

