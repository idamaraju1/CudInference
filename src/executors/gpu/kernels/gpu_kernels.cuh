#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace onnx_runner {

enum class DataType;
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

// Element-wise Sub
void launchSub(const float* A, const float* B, float* C, int size, cudaStream_t stream = 0);
void launchSubScalar(const float* A, float scalar, float* C, int size, cudaStream_t stream = 0);

// Multi-threaded CPU fallback for Sub
void subCPU(const float* A, const float* B, float* C, int size);
void subCPUMultiThreaded(const float* A, const float* B, float* C, int size, int num_threads);


// Launch: X[M,N] -> Y[M,N], optional gamma[N], beta[N], epsilon scalar
void launchSimplifiedLayerNorm(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon, cudaStream_t stream);

// CPU fallbacks (single-threaded and OpenMP)
void simplifiedLayerNormCPU(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon);
void simplifiedLayerNormCPUMultiThreaded(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon, int num_threads);

// SkipSimplifiedLayerNormalization (residual + RMSNorm)
void launchSkipSimplifiedLayerNorm(const float* X, const float* Skip, const float* gamma, const float* beta, float* Y, float* residual_out, int M, int N, float epsilon, cudaStream_t stream);
void skipSimplifiedLayerNormCPU(const float* X, const float* Skip, const float* gamma, const float* beta, float* Y, float* residual_out, int M, int N, float epsilon);
void skipSimplifiedLayerNormCPUMultiThreaded(const float* X, const float* Skip, const float* gamma, const float* beta, float* Y, float* residual_out, int M, int N, float epsilon, int num_threads);

// Sigmoid activation
void launchSigmoid(const float* input, float* output, int size, cudaStream_t stream = 0);
void sigmoidCPU(const float* input, float* output, int size);
void sigmoidCPUMultiThreaded(const float* input, float* output, int size, int num_threads);

} // namespace kernels

// ReduceSum operation (outside kernels namespace)
void launchReduceSum(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    const std::vector<bool>& reduce_mask,
    bool keepdims,
    bool use_cpu = false,
    int num_threads = 1
);

// RotaryEmbedding operation (outside kernels namespace)
void launchRotaryEmbedding(
    const float* input,        // [batch_seq, num_heads, head_size]
    const float* cos_cache,    // [batch_seq, rotary_half]
    const float* sin_cache,    // [batch_seq, rotary_half]
    float* output,             // [batch_seq, num_heads, head_size]
    int batch_seq,
    int num_heads,
    int head_size,
    int rotary_dim,
    bool interleaved,
    bool use_cpu = false,
    int num_threads = 1
);

// Gather operation (outside kernels namespace)
void launchGatherKernel(
    const void* data,
    const int64_t* indices,
    void* output,
    DataType dtype,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size,
    bool use_cpu = false,
    int num_threads = 1
);

// Element-wise operations
void launchMulKernel(const float* A, const float* B, float* C, int size, bool use_cpu = false, int num_threads = 1);
void launchMulScalarKernel(const float* A, float scalar, float* C, int size, bool use_cpu = false, int num_threads = 1);
void launchDivKernel(const float* A, const float* B, float* C, int size, bool use_cpu = false, int num_threads = 1);
void launchDivScalarKernel(const float* A, float scalar, float* C, int size, bool use_cpu = false, int num_threads = 1);
void launchPowKernel(const float* A, const float* B, float* C, int size, bool use_cpu = false, int num_threads = 1);
void launchPowScalarKernel(const float* A, float exponent, float* C, int size, bool use_cpu = false, int num_threads = 1);
void launchSqrtKernel(const float* A, float* C, int size, bool use_cpu = false, int num_threads = 1);

// Reduction operations
void launchReduceMeanKernel(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& axes,
    bool use_cpu = false,
    int num_threads = 1
);

// Tensor manipulation operations
void launchReshapeKernel(const float* input, float* output, int64_t total_size, bool use_cpu = false, int num_threads = 1);
void launchTransposeKernel(const float* input, float* output, const std::vector<int64_t>& input_shape, const std::vector<int>& perm, bool use_cpu = false, int num_threads = 1);
void launchUnsqueezeKernel(const float* input, float* output, int64_t total_size, bool use_cpu = false, int num_threads = 1);
void launchSliceKernel(const float* input, float* output, const std::vector<int64_t>& input_shape, const std::vector<int64_t>& starts, const std::vector<int64_t>& steps, const std::vector<int64_t>& output_shape, bool use_cpu = false, int num_threads = 1);
void launchConcatKernel(const std::vector<const float*>& inputs, float* output, const std::vector<std::vector<int64_t>>& input_shapes, int64_t axis, const std::vector<int64_t>& output_shape, bool use_cpu = false, int num_threads = 1);

// GroupQueryAttention operation
void launchGroupQueryAttention(
    const float* Q,
    const float* K_storage,
    const float* V_storage,
    float* output,
    int batch,
    int q_seq,
    int q_heads,
    int kv_heads,
    int head_dim,
    int total_seq,
    int past_len,
    float scale,
    float softcap,
    const std::vector<size_t>& valid_lengths,
    bool use_cpu = false,
    int num_threads = 1
);

// KV cache reformatting (for persistent GPU cache)
void launchReformatKV(
    const float* kv_input,     // [batch, seq, kv_hidden]
    float* kv_output,          // [batch, kv_heads, seq, head_dim]
    int batch,
    int seq,
    int kv_heads,
    int head_dim
);

} // namespace onnx_runner
