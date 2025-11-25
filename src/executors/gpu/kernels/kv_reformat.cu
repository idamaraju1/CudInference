#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace onnx_runner {

// Reformat K/V from [batch, seq, kv_hidden] to [batch, kv_heads, seq, head_dim]
// This allows the data to be directly used in attention and appended to persistent cache
__global__ void reformatKVKernel(
    const float* kv_input,     // [batch, seq, kv_hidden]
    float* kv_output,          // [batch, kv_heads, seq, head_dim]
    int batch,
    int seq,
    int kv_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * seq * head_dim;

    if (idx < total) {
        // Output layout: [batch, kv_heads, seq, head_dim]
        int d = idx % head_dim;
        int s = (idx / head_dim) % seq;
        int h = (idx / (head_dim * seq)) % kv_heads;
        int b = idx / (head_dim * seq * kv_heads);

        // Input layout: [batch, seq, kv_hidden] where kv_hidden = kv_heads * head_dim
        int src_idx = (b * seq + s) * (kv_heads * head_dim) + h * head_dim + d;

        kv_output[idx] = kv_input[src_idx];
    }
}

// Launcher function
void launchReformatKV(
    const float* kv_input,     // Device pointer [batch, seq, kv_hidden]
    float* kv_output,          // Device pointer [batch, kv_heads, seq, head_dim]
    int batch,
    int seq,
    int kv_heads,
    int head_dim
) {
    int total = batch * kv_heads * seq * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    reformatKVKernel<<<blocks, threads>>>(
        kv_input, kv_output, batch, seq, kv_heads, head_dim
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

} // namespace onnx_runner
