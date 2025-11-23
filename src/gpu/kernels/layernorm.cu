#include "kernels.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <string>
#include <algorithm> 

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

template <int BLOCK_SIZE, bool HasGamma, bool HasBeta>
__global__ void layernorm_fused_kernel(const float* __restrict__ X,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ Y,
                                       int N,
                                       float eps) {
    int row = blockIdx.x;
    const float* x = X + row * N;
    float*       y = Y + row * N;

    // compute per-row mean/var
    float sum   = 0.f;
    float sumsq = 0.f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float v = x[i];
        sum   += v;
        sumsq += v * v;
    }

    sum   = blockReduceSum<BLOCK_SIZE>(sum);
    sumsq = blockReduceSum<BLOCK_SIZE>(sumsq);

    __shared__ float s_mean;
    __shared__ float s_invstd;

    if (threadIdx.x == 0) {
        float m   = sum   / static_cast<float>(N);
        float ex2 = sumsq / static_cast<float>(N);
        float var = fmaxf(ex2 - m * m, 0.f);
        s_mean   = m;
        s_invstd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean   = s_mean;
    float invstd = s_invstd;

    // apply normalization, gamma, beta
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float v = (x[i] - mean) * invstd;
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
    dim3 grid(M);
    dim3 block(BLOCK);

    //apply
    if (gamma && beta) {
        layernorm_fused_kernel<BLOCK, true, true>
            <<<grid, block, 0, stream>>>(X, gamma, beta, Y, N, epsilon);
    } else if (gamma) {
        layernorm_fused_kernel<BLOCK, true, false>
            <<<grid, block, 0, stream>>>(X, gamma, nullptr, Y, N, epsilon);
    } else if (beta) {
        layernorm_fused_kernel<BLOCK, false, true>
            <<<grid, block, 0, stream>>>(X, nullptr, beta, Y, N, epsilon);
    } else {
        layernorm_fused_kernel<BLOCK, false, false>
            <<<grid, block, 0, stream>>>(X, nullptr, nullptr, Y, N, epsilon);
    }

    // catch kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("LayerNorm kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }
}

// ---- CPU fallbacks (match header names exactly) ----
void simplifiedLayerNormCPU(const float* X,
                            const float* gamma,
                            const float* beta,
                            float* Y,
                            int M, int N,
                            float epsilon) {
    for (int r = 0; r < M; ++r) {
        const float* x = X + r * N;
        float* y       = Y + r * N;

        double sum = 0.0, sumsq = 0.0;  // better accumulator precision
        for (int i = 0; i < N; ++i) {
            float v = x[i];
            sum   += v;
            sumsq += double(v) * v;
        }
        float mean = static_cast<float>(sum / N);
        float var  = static_cast<float>(sumsq / N - double(mean) * mean);
        float invs = 1.0f / std::sqrt(std::max(var + epsilon, 0.0f));

        for (int i = 0; i < N; ++i) {
            float v = (x[i] - mean) * invs;
            if (gamma) v *= gamma[i];
            if (beta)  v += beta[i];
            y[i] = v;
        }
    }
}

void simplifiedLayerNormCPUMultiThreaded(const float* X,
                                         const float* gamma,
                                         const float* beta,
                                         float* Y,
                                         int M, int N,
                                         float epsilon,
                                         int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int r = 0; r < M; ++r) {
        const float* x = X + r * N;
        float* y       = Y + r * N;

        double sum = 0.0, sumsq = 0.0;
        for (int i = 0; i < N; ++i) {
            float v = x[i];
            sum   += v;
            sumsq += double(v) * v;
        }
        float mean = static_cast<float>(sum / N);
        float var  = static_cast<float>(sumsq / N - double(mean) * mean);
        float invs = 1.0f / std::sqrt(std::max(var + epsilon, 0.0f));

        for (int i = 0; i < N; ++i) {
            float v = (x[i] - mean) * invs;
            if (gamma) v *= gamma[i];
            if (beta)  v += beta[i];
            y[i] = v;
        }
    }
}

} // namespace kernels
} // namespace onnx_runner
