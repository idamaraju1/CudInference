#include "cpu_kernels.h"
#include <omp.h>

namespace onnx_runner {
namespace kernels {

// CPU fallback
void addCPU(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Multi-threaded CPU implementation using OpenMP
void addCPUMultiThreaded(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}

} // namespace kernels
} // namespace onnx_runner

