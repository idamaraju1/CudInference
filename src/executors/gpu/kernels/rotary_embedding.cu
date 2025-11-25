#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>

namespace onnx_runner {

// GPU kernel for Rotary Position Embedding (RoPE)
// Non-interleaved mode: [x0, x1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
// Interleaved mode: [x0, x1, x2, x3, ...] where pairs are (x0,x1), (x2,x3)
__global__ void ropeKernel(
    const float* __restrict__ input,        // [batch * seq, num_heads, head_size]
    const float* __restrict__ cos_cache,    // [batch * seq, rotary_half]
    const float* __restrict__ sin_cache,    // [batch * seq, rotary_half]
    float* __restrict__ output,             // [batch * seq, num_heads, head_size]
    int batch_seq,             // batch * sequence_length
    int num_heads,
    int head_size,
    int rotary_dim,
    bool interleaved
) {
    int token_idx = blockIdx.x;  // Which token (batch * seq)
    int head_idx = blockIdx.y;   // Which head
    int tid = threadIdx.x;       // Thread within block

    if (token_idx >= batch_seq || head_idx >= num_heads) return;

    int rotary_half = rotary_dim / 2;

    const float* input_head = input + (token_idx * num_heads + head_idx) * head_size;
    float* output_head = output + (token_idx * num_heads + head_idx) * head_size;
    const float* cos_vals = cos_cache + token_idx * rotary_half;
    const float* sin_vals = sin_cache + token_idx * rotary_half;

    for (int pair = tid; pair < rotary_half; pair += blockDim.x) {
        int idx0, idx1;

        if (!interleaved) {
            // Non-interleaved: first half [0...rotary_half-1], second half [rotary_half...rotary_dim-1]
            idx0 = pair;
            idx1 = pair + rotary_half;
        } else {
            // Interleaved: pairs at [2i, 2i+1]
            idx0 = pair * 2;
            idx1 = idx0 + 1;
        }

        float x0 = input_head[idx0];
        float x1 = input_head[idx1];
        float c  = cos_vals[pair];
        float s  = sin_vals[pair];

        // standard RoPE rotation.
        output_head[idx0] = x0 * c - x1 * s;
        output_head[idx1] = x1 * c + x0 * s;
    }



    // Copy non-rotated dimensions
    for (int i = rotary_dim + tid; i < head_size; i += blockDim.x) {
        output_head[i] = input_head[i];
    }
}

// CPU implementation for RoPE
void ropeCPU(
    const float* input,
    const float* cos_cache,
    const float* sin_cache,
    float* output,
    int batch_seq,
    int num_heads,
    int head_size,
    int rotary_dim,
    bool interleaved,
    int num_threads
) {
    int rotary_half = rotary_dim / 2;

    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int token_idx = 0; token_idx < batch_seq; ++token_idx) {
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
            const float* input_head = input + (token_idx * num_heads + head_idx) * head_size;
            float* output_head = output + (token_idx * num_heads + head_idx) * head_size;
            const float* cos_vals = cos_cache + token_idx * rotary_half;
            const float* sin_vals = sin_cache + token_idx * rotary_half;

            if (!interleaved) {
                for (int i = 0; i < rotary_half; ++i) {
                    float x1 = input_head[i];
                    float x2 = input_head[i + rotary_half];
                    float c = cos_vals[i];
                    float s = sin_vals[i];

                    output_head[i] = x1 * c - x2 * s;
                    output_head[i + rotary_half] = x2 * c + x1 * s;
                }
            } else {
                for (int i = 0; i < rotary_half; ++i) {
                    int even_idx = i * 2;
                    int odd_idx = even_idx + 1;
                    float x_even = input_head[even_idx];
                    float x_odd = input_head[odd_idx];
                    float c = cos_vals[i];
                    float s = sin_vals[i];

                    output_head[even_idx] = x_even * c - x_odd * s;
                    output_head[odd_idx] = x_odd * c + x_even * s;
                }
            }

            // Copy non-rotated dimensions
            for (int i = rotary_dim; i < head_size; ++i) {
                output_head[i] = input_head[i];
            }
        }
    }
}

// Launcher function
void launchRotaryEmbedding(
    const float* input,
    const float* cos_cache,
    const float* sin_cache,
    float* output,
    int batch_seq,
    int num_heads,
    int head_size,
    int rotary_dim,
    bool interleaved,
    bool use_cpu,
    int num_threads
) {
    if (use_cpu) {
        ropeCPU(input, cos_cache, sin_cache, output, batch_seq, num_heads,
                head_size, rotary_dim, interleaved, num_threads);
    } else {
        // GPU path
        int rotary_half = rotary_dim / 2;
        dim3 grid_size(batch_seq, num_heads);
        int block_size = (rotary_half + 31) / 32 * 32;  // Round up to warp size
        block_size = min(block_size, 256);  // Cap at 256 threads

        ropeKernel<<<grid_size, block_size>>>(
            input, cos_cache, sin_cache, output,
            batch_seq, num_heads, head_size, rotary_dim, interleaved
        );

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("RoPE CUDA kernel error: ") +
                                   cudaGetErrorString(error));
        }
    }
}

} // namespace onnx_runner
