#include "cpu_kernels.h"
#include <omp.h>
#include <cmath>

namespace onnx_runner {

void mulCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] * B[i];
    }
}

void divCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] / B[i];
    }
}

void powCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = std::pow(A[i], B[i]);
    }
}

void sqrtCPU(const float* A, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = std::sqrt(A[i]);
    }
}

} // namespace onnx_runner

