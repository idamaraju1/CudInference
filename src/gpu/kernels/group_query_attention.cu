#include "kernels.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace onnx_runner {

// Warp-level reduction for max finding
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for max
__device__ float blockReduceMax(float val) {
    __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);

    return val;
}

// Block-level reduction for sum
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

// Fused GroupQueryAttention kernel
// Each block handles one (batch, query_head, query_seq_pos) combination
// Threads cooperate to compute attention over all key positions
__global__ void gqaKernel(
    const float* Q,              // [batch, q_seq, q_hidden]
    const float* K_storage,      // [batch, kv_heads, total_seq, head_dim]
    const float* V_storage,      // [batch, kv_heads, total_seq, head_dim]
    float* output,               // [batch, q_seq, q_hidden]
    float* scores_temp,          // Temporary storage [batch * q_heads * q_seq * total_seq]
    int batch,
    int q_seq,
    int q_heads,
    int kv_heads,
    int head_dim,
    int total_seq,
    int past_len,
    int group_size,
    float scale,
    float softcap,
    const int* valid_lengths     // [batch] - max sequence length per batch
) {
    // Grid: (batch, q_heads, q_seq)
    int b = blockIdx.x;
    int qh = blockIdx.y;
    int qs = blockIdx.z;

    if (b >= batch || qh >= q_heads || qs >= q_seq) return;

    int kv_head = qh / group_size;
    int tid = threadIdx.x;

    // Causal masking: can only attend up to past_len + qs
    int causal_limit = past_len + qs + 1;
    int valid = valid_lengths ? valid_lengths[b] : total_seq;
    int allowed = min(valid, causal_limit);
    allowed = min(allowed, total_seq);

    // Load Q vector into shared memory
    extern __shared__ float smem[];
    float* q_vec = smem;

    int q_hidden = q_heads * head_dim;
    const float* q_src = Q + (b * q_seq + qs) * q_hidden + qh * head_dim;

    // Cooperatively load Q vector
    for (int i = tid; i < head_dim; i += blockDim.x) {
        q_vec[i] = q_src[i];
    }
    __syncthreads();

    // Pointers to K and V for this head
    const float* key_head = K_storage + ((b * kv_heads + kv_head) * total_seq * head_dim);
    const float* value_head = V_storage + ((b * kv_heads + kv_head) * total_seq * head_dim);

    // precompute score base index.
    int base_score_idx = ((b * q_heads + qh) * q_seq + qs) * total_seq;

    // Phase 1: Compute attention scores and find max
    float thread_max = -INFINITY;

    // Each thread handles multiple key positions
    for (int t = tid; t < allowed; t += blockDim.x) {
        const float* k_vec = key_head + t * head_dim;

        // Compute dot product Q·K
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }

        // Scale and apply softcap if needed
        float score = dot * scale;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }

        // Store score
        scores_temp[base_score_idx + t] = score;

        thread_max = fmaxf(thread_max, score);
    }

    // Find global max across block
    float max_score = blockReduceMax(thread_max);
    __shared__ float s_max;
    if (tid == 0) s_max = max_score;
    __syncthreads();
    max_score = s_max;

    // Phase 2: Compute exp and sum for softmax
    float thread_sum = 0.0f;

    for (int t = tid; t < allowed; t += blockDim.x) {
        int score_idx = ((b * q_heads + qh) * q_seq + qs) * total_seq + t;
        float score = scores_temp[base_score_idx + t];
        float expval = expf(score - max_score);
        scores_temp[base_score_idx + t] = expval;
        thread_sum += expval;
    }

    // Find global sum across block
    float sum = blockReduceSum(thread_sum);
    __shared__ float s_sum;
    if (tid == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    // Phase 3: Compute weighted sum over values (no atomic ops)
    // Each thread owns a subset of d

    float* out_dst = output + (b * q_seq + qs) * q_hidden + qh * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        // Loop over all allowed key positions
        for (int t = 0; t < allowed; ++t) {
            float weight = scores_temp[base_score_idx + t] * inv_sum;
            const float* v_vec = value_head + t * head_dim;
            acc += weight * v_vec[d];
        }

        out_dst[d] = acc;
    }
}

// CPU implementation (existing logic extracted)
void gqaCPU(
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
    int num_threads
) {
    int q_hidden = q_heads * head_dim;
    size_t group_size = q_heads / kv_heads;

    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (int b = 0; b < batch; ++b) {
        for (int qh = 0; qh < q_heads; ++qh) {
            for (int qs = 0; qs < q_seq; ++qs) {
                size_t kv_head = qh / group_size;
                const float* key_head = K_storage + ((b * kv_heads + kv_head) * total_seq * head_dim);
                const float* value_head = V_storage + ((b * kv_heads + kv_head) * total_seq * head_dim);

                const float* q_vec = Q + ((b * q_seq + qs) * q_hidden) + qh * head_dim;

                size_t causal_limit = past_len + qs + 1;
                size_t valid = valid_lengths[b];
                size_t allowed = std::min(valid, causal_limit);
                if (allowed == 0) allowed = 1;
                allowed = std::min(allowed, static_cast<size_t>(total_seq));

                // Compute scores and find max
                std::vector<float> scores(total_seq, 0.0f);
                float max_score = -std::numeric_limits<float>::infinity();

                for (size_t t = 0; t < allowed; ++t) {
                    const float* k_vec = key_head + t * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_vec[d] * k_vec[d];
                    }
                    float scaled = dot * scale;
                    if (softcap > 0.0f) {
                        scaled = softcap * std::tanh(scaled / softcap);
                    }
                    scores[t] = scaled;
                    if (scaled > max_score) max_score = scaled;
                }

                // Compute softmax
                float denom = 0.0f;
                for (size_t t = 0; t < allowed; ++t) {
                    float expv = std::exp(scores[t] - max_score);
                    scores[t] = expv;
                    denom += expv;
                }
                float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

                // Compute weighted sum
                float* out_vec = output + ((b * q_seq + qs) * q_hidden) + qh * head_dim;
                std::fill(out_vec, out_vec + head_dim, 0.0f);

                for (size_t t = 0; t < allowed; ++t) {
                    float weight = scores[t] * inv_denom;
                    const float* v_vec = value_head + t * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        out_vec[d] += weight * v_vec[d];
                    }
                }
            }
        }
    }
}

// Launcher function
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
    bool use_cpu,
    int num_threads
) {
    int q_hidden = q_heads * head_dim;
    int group_size = q_heads / kv_heads;

    if (use_cpu) {
        gqaCPU(Q, K_storage, V_storage, output, batch, q_seq, q_heads, kv_heads,
               head_dim, total_seq, past_len, scale, softcap, valid_lengths, num_threads);
    } else {
        // GPU path
        // Allocate temporary storage for scores
        size_t scores_size = static_cast<size_t>(batch) * q_heads * q_seq * total_seq;
        float* d_scores_temp;
        cudaMalloc(&d_scores_temp, scores_size * sizeof(float));

        // Copy valid_lengths to device if provided
        int* d_valid_lengths = nullptr;
        if (!valid_lengths.empty()) {
            cudaMalloc(&d_valid_lengths, batch * sizeof(int));
            std::vector<int> valid_int(batch);
            for (int i = 0; i < batch; ++i) {
                valid_int[i] = static_cast<int>(valid_lengths[i]);
            }
            cudaMemcpy(d_valid_lengths, valid_int.data(), batch * sizeof(int), cudaMemcpyHostToDevice);
        }

        // Configure kernel launch
        dim3 grid(batch, q_heads, q_seq);
        int block_size = 256;
        size_t smem_size = head_dim * sizeof(float); // only Q vector

        gqaKernel<<<grid, block_size, smem_size>>>(
            Q, K_storage, V_storage, output, d_scores_temp,
            batch, q_seq, q_heads, kv_heads, head_dim, total_seq, past_len,
            group_size, scale, softcap, d_valid_lengths
        );

        cudaDeviceSynchronize();

        cudaFree(d_scores_temp);
        if (d_valid_lengths) cudaFree(d_valid_lengths);
    }
}

} // namespace onnx_runner
