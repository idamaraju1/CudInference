#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) do {                                     \
    cudaError_t __err = (expr);                                   \
    if (__err != cudaSuccess) {                                   \
        throw std::runtime_error(std::string("CUDA error: ") +    \
            cudaGetErrorString(__err));                           \
    }                                                             \
} while (0)
#endif

namespace onnx_runner {
namespace kernels {

// ---- block reduction helper ----
template <int BLOCK_SIZE>
__device__ float blockReduceSum(float v) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();

    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        volatile float* vsmem = smem; // NOLINT
        if (BLOCK_SIZE >=  64) vsmem[tid] += vsmem[tid + 32];
        if (BLOCK_SIZE >=  32) vsmem[tid] += vsmem[tid + 16];
        if (BLOCK_SIZE >=  16) vsmem[tid] += vsmem[tid +  8];
        if (BLOCK_SIZE >=   8) vsmem[tid] += vsmem[tid +  4];
        if (BLOCK_SIZE >=   4) vsmem[tid] += vsmem[tid +  2];
        if (BLOCK_SIZE >=   2) vsmem[tid] += vsmem[tid +  1];
    }
    return smem[0];
}

// --- kernel 1: per-row stats (mean, inv_std) ---
template <int BLOCK_SIZE>
__global__ void layernorm_stats_kernel(const float* __restrict__ X,
                                       float* __restrict__ mean,
                                       float* __restrict__ inv_std,
                                       int N, float eps) {
    int row = blockIdx.x;                  // one block per row
    const float* x = X + row * N;

    float sum = 0.f;
    float sumsq = 0.f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float v = x[i];
        sum   += v;
        sumsq += v * v;
    }

    sum   = blockReduceSum<BLOCK_SIZE>(sum);
    sumsq = blockReduceSum<BLOCK_SIZE>(sumsq);

    if (threadIdx.x == 0) {
        float m   = sum   / static_cast<float>(N);
        float ex2 = sumsq / static_cast<float>(N);
        float var = fmaxf(ex2 - m * m, 0.f);
        mean[row]    = m;
        inv_std[row] = rsqrtf(var + eps);
    }
}

// --- kernel 2: apply normalization, optional gamma/beta ---
template <int BLOCK_SIZE, bool HasGamma, bool HasBeta>
__global__ void layernorm_apply_kernel(const float* __restrict__ X,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ inv_std,
                                       float* __restrict__ Y,
                                       int N) {
    int row = blockIdx.x;
    const float m = mean[row];
    const float s = inv_std[row];

    const float* x = X + row * N;
    float* y       = Y + row * N;

    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float v = (x[i] - m) * s;
        if constexpr (HasGamma) v *= gamma[i];
        if constexpr (HasBeta)  v += beta[i];
        y[i] = v;
    }
}

// ---- public launcher (matches header) ----
void launchSimplifiedLayerNorm(const float* X,
                               const float* gamma,
                               const float* beta,
                               float* Y,
                               int M, int N,
                               float epsilon,
                               cudaStream_t stream) {
    if (M <= 0 || N <= 0) {
        throw std::runtime_error("launchSimplifiedLayerNorm: invalid M or N");
    }

    constexpr int BLOCK = 256;

    float* d_mean = nullptr;
    float* d_inv  = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_mean, sizeof(float) * M, stream));
    CUDA_CHECK(cudaMallocAsync(&d_inv,  sizeof(float) * M, stream));

    // 1) stats
    layernorm_stats_kernel<BLOCK><<<M, BLOCK, 0, stream>>>(X, d_mean, d_inv, N, epsilon);

    // 2) apply
    if (gamma && beta) {
        layernorm_apply_kernel<BLOCK, true, true><<<M, BLOCK, 0, stream>>>(X, gamma, beta, d_mean, d_inv, Y, N);
    } else if (gamma) {
        layernorm_apply_kernel<BLOCK, true, false><<<M, BLOCK, 0, stream>>>(X, gamma, nullptr, d_mean, d_inv, Y, N);
    } else if (beta) {
        layernorm_apply_kernel<BLOCK, false, true><<<M, BLOCK, 0, stream>>>(X, nullptr, beta, d_mean, d_inv, Y, N);
    } else {
        layernorm_apply_kernel<BLOCK, false, false><<<M, BLOCK, 0, stream>>>(X, nullptr, nullptr, d_mean, d_inv, Y, N);
    }

    // catch kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeAsync(d_mean, stream);
        cudaFreeAsync(d_inv,  stream);
        throw std::runtime_error(std::string("LayerNorm kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }

    CUDA_CHECK(cudaFreeAsync(d_mean, stream));
    CUDA_CHECK(cudaFreeAsync(d_inv,  stream));
}

} // namespace kernels
} // namespace onnx_runner

