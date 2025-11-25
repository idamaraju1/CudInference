#include "cpu_kernels.h"
#include <cmath>

namespace onnx_runner {
namespace kernels {

// SimplifiedLayerNorm CPU implementation
void simplifiedLayerNormCPU(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon) {
    for (int m = 0; m < M; ++m) {
        const float* row = X + m * N;
        float* out_row = Y + m * N;

        // Compute mean
        float mean = 0.0f;
        for (int n = 0; n < N; ++n) {
            mean += row[n];
        }
        mean /= N;

        // Compute variance
        float variance = 0.0f;
        for (int n = 0; n < N; ++n) {
            float diff = row[n] - mean;
            variance += diff * diff;
        }
        variance /= N;

        // Normalize and apply affine transform
        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for (int n = 0; n < N; ++n) {
            float normalized = (row[n] - mean) * inv_std;
            out_row[n] = normalized * (gamma ? gamma[n] : 1.0f) + (beta ? beta[n] : 0.0f);
        }
    }
}

// Multi-threaded CPU implementation using OpenMP
void simplifiedLayerNormCPUMultiThreaded(const float* X, const float* gamma, const float* beta, float* Y, int M, int N, float epsilon, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int m = 0; m < M; ++m) {
        const float* row = X + m * N;
        float* out_row = Y + m * N;

        // Compute mean
        float mean = 0.0f;
        for (int n = 0; n < N; ++n) {
            mean += row[n];
        }
        mean /= N;

        // Compute variance
        float variance = 0.0f;
        for (int n = 0; n < N; ++n) {
            float diff = row[n] - mean;
            variance += diff * diff;
        }
        variance /= N;

        // Normalize and apply affine transform
        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for (int n = 0; n < N; ++n) {
            float normalized = (row[n] - mean) * inv_std;
            out_row[n] = normalized * (gamma ? gamma[n] : 1.0f) + (beta ? beta[n] : 0.0f);
        }
    }
}

// SkipSimplifiedLayerNormalization CPU implementation
void skipSimplifiedLayerNormCPU(const float* X, const float* Skip, const float* gamma, const float* beta, float* Y, float* residual_out, int M, int N, float epsilon) {
    for (int m = 0; m < M; ++m) {
        const float* x_row = X + m * N;
        const float* skip_row = Skip + m * N;
        float* y_row = Y + m * N;
        float* residual_row = residual_out + m * N;

        // Add residual
        for (int n = 0; n < N; ++n) {
            residual_row[n] = x_row[n] + skip_row[n];
        }

        // Compute mean
        float mean = 0.0f;
        for (int n = 0; n < N; ++n) {
            mean += residual_row[n];
        }
        mean /= N;

        // Compute variance
        float variance = 0.0f;
        for (int n = 0; n < N; ++n) {
            float diff = residual_row[n] - mean;
            variance += diff * diff;
        }
        variance /= N;

        // Normalize and apply affine transform
        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for (int n = 0; n < N; ++n) {
            float normalized = (residual_row[n] - mean) * inv_std;
            y_row[n] = normalized * (gamma ? gamma[n] : 1.0f) + (beta ? beta[n] : 0.0f);
        }
    }
}

// Multi-threaded SkipSimplifiedLayerNormalization
void skipSimplifiedLayerNormCPUMultiThreaded(const float* X, const float* Skip, const float* gamma, const float* beta, float* Y, float* residual_out, int M, int N, float epsilon, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int m = 0; m < M; ++m) {
        const float* x_row = X + m * N;
        const float* skip_row = Skip + m * N;
        float* y_row = Y + m * N;
        float* residual_row = residual_out + m * N;

        // Add residual
        for (int n = 0; n < N; ++n) {
            residual_row[n] = x_row[n] + skip_row[n];
        }

        // Compute mean
        float mean = 0.0f;
        for (int n = 0; n < N; ++n) {
            mean += residual_row[n];
        }
        mean /= N;

        // Compute variance
        float variance = 0.0f;
        for (int n = 0; n < N; ++n) {
            float diff = residual_row[n] - mean;
            variance += diff * diff;
        }
        variance /= N;

        // Normalize and apply affine transform
        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for (int n = 0; n < N; ++n) {
            float normalized = (residual_row[n] - mean) * inv_std;
            y_row[n] = normalized * (gamma ? gamma[n] : 1.0f) + (beta ? beta[n] : 0.0f);
        }
    }
}

} // namespace kernels
} // namespace onnx_runner
