#pragma once

#include <vector>
#include <cstdint>

namespace onnx_runner {

enum class DataType;

namespace kernels {

// MatMul CPU implementations
void matmulCPU(const float* A, const float* B, float* C, int M, int K, int N);
void matmulCPUMultiThreaded(const float* A, const float* B, float* C, int M, int K, int N, int num_threads);

// ReLU activation
void reluCPU(const float* input, float* output, int size);
void reluCPUMultiThreaded(const float* input, float* output, int size, int num_threads);

// Element-wise Add
void addCPU(const float* A, const float* B, float* C, int size);
void addCPUMultiThreaded(const float* A, const float* B, float* C, int size, int num_threads);

// Element-wise Sub
void subCPU(const float* A, const float* B, float* C, int size);
void subCPUMultiThreaded(const float* A, const float* B, float* C, int size, int num_threads);

// SimplifiedLayerNorm
void simplifiedLayerNormCPU(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon);
void simplifiedLayerNormCPUMultiThreaded(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon, int num_threads);

// SkipSimplifiedLayerNormalization (residual + RMSNorm)
void skipSimplifiedLayerNormCPU(const float* X, const float* Skip, const float* gamma, const float* beta, float* Y, float* residual_out, int M, int N, float epsilon);
void skipSimplifiedLayerNormCPUMultiThreaded(const float* X, const float* Skip, const float* gamma, const float* beta, float* Y, float* residual_out, int M, int N, float epsilon, int num_threads);

// Sigmoid activation
void sigmoidCPU(const float* input, float* output, int size);
void sigmoidCPUMultiThreaded(const float* input, float* output, int size, int num_threads);

} // namespace kernels

// ReduceSum operation
void reduceSumCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    const std::vector<bool>& reduce_mask,
    bool keepdims,
    int num_threads = 1
);

// RotaryEmbedding operation
void rotaryEmbeddingCPU(
    const float* input,
    const float* cos_cache,
    const float* sin_cache,
    float* output,
    int batch_seq,
    int num_heads,
    int head_size,
    int rotary_dim,
    bool interleaved,
    int num_threads = 1
);

// Gather operation
void gatherCPU(
    const void* data,
    const int64_t* indices,
    void* output,
    DataType dtype,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size,
    int num_threads = 1
);

// Element-wise operations
void mulCPU(const float* A, const float* B, float* C, int size, int num_threads = 1);
void mulScalarCPU(const float* A, float scalar, float* C, int size, int num_threads = 1);
void divCPU(const float* A, const float* B, float* C, int size, int num_threads = 1);
void divScalarCPU(const float* A, float scalar, float* C, int size, int num_threads = 1);
void powCPU(const float* A, const float* B, float* C, int size, int num_threads = 1);
void powScalarCPU(const float* A, float exponent, float* C, int size, int num_threads = 1);
void sqrtCPU(const float* A, float* C, int size, int num_threads = 1);

// Reduction operations
void reduceMeanCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& axes,
    int num_threads = 1
);

// Tensor manipulation operations
void reshapeCPU(const float* input, float* output, int64_t total_size, int num_threads = 1);
void transposeCPU(const float* input, float* output, const std::vector<int64_t>& input_shape, const std::vector<int>& perm, int num_threads = 1);
void unsqueezeCPU(const float* input, float* output, int64_t total_size, int num_threads = 1);
void sliceCPU(const float* input, float* output, const std::vector<int64_t>& input_shape, const std::vector<int64_t>& starts, const std::vector<int64_t>& steps, const std::vector<int64_t>& output_shape, int num_threads = 1);
void concatCPU(const std::vector<const float*>& inputs, float* output, const std::vector<std::vector<int64_t>>& input_shapes, int64_t axis, const std::vector<int64_t>& output_shape, int num_threads = 1);

// GroupQueryAttention operation
void groupQueryAttentionCPU(
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
    int num_threads = 1
);

} // namespace onnx_runner
