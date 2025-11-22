#include "cpu_kernels.h"
#include <omp.h>
#include <cmath>

namespace onnx_runner {
namespace kernels {

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

