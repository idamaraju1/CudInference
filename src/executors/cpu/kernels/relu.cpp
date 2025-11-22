#include "cpu_kernels.h"
#include <omp.h>

namespace onnx_runner {
namespace kernels {

// CPU fallback
void reluCPU(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// Multi-threaded CPU implementation using OpenMP
void reluCPUMultiThreaded(const float* input, float* output, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

} // namespace kernels
} // namespace onnx_runner

