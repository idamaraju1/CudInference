#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace onnx_runner {
namespace kernels {

// MatMul kernel
// Computes C = A @ B where A is (M, K), B is (K, N), C is (M, N)
void launchMatMul(const float* A, const float* B, float* C,
                  int M, int K, int N, cudaStream_t stream = 0);

// ReLU activation
void launchReLU(const float* input, float* output, int size, cudaStream_t stream = 0);
void launchReLUInPlace(float* data, int size, cudaStream_t stream = 0);

// Element-wise Add
void launchAdd(const float* A, const float* B, float* C, int size, cudaStream_t stream = 0);
void launchAddScalar(const float* A, float scalar, float* C, int size, cudaStream_t stream = 0);

// Element-wise Sub
void launchSub(const float* A, const float* B, float* C, int size, cudaStream_t stream = 0);
void launchSubScalar(const float* A, float scalar, float* C, int size, cudaStream_t stream = 0);

// Launch: X[M,N] -> Y[M,N], optional gamma[N], beta[N], epsilon scalar
void launchSimplifiedLayerNorm(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon, cudaStream_t stream);

} // namespace kernels

// Gather operation (outside kernels namespace)
void launchGatherKernel(
    const float* data,
    const int64_t* indices,
    float* output,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size
);

// Element-wise operations
void launchMulKernel(const float* A, const float* B, float* C, int size);
void launchMulScalarKernel(const float* A, float scalar, float* C, int size);
void launchDivKernel(const float* A, const float* B, float* C, int size);
void launchDivScalarKernel(const float* A, float scalar, float* C, int size);
void launchPowKernel(const float* A, const float* B, float* C, int size);
void launchPowScalarKernel(const float* A, float exponent, float* C, int size);
void launchSqrtKernel(const float* A, float* C, int size);

// Reduction operations
void launchReduceMeanKernel(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& axes
);

// Tensor manipulation operations
void launchReshapeKernel(const float* input, float* output, int64_t total_size);
void launchTransposeKernel(const float* input, float* output, const std::vector<int64_t>& input_shape, const std::vector<int>& perm);
void launchUnsqueezeKernel(const float* input, float* output, int64_t total_size);
void launchSliceKernel(const float* input, float* output, const std::vector<int64_t>& input_shape, const std::vector<int64_t>& starts, const std::vector<int64_t>& steps, const std::vector<int64_t>& output_shape);
void launchConcatKernel(const std::vector<const float*>& inputs, float* output, const std::vector<std::vector<int64_t>>& input_shapes, int64_t axis, const std::vector<int64_t>& output_shape);

} // namespace onnx_runner

