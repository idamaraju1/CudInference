#include "cpu_kernels.h"
#include <cmath>

namespace onnx_runner {
namespace kernels {

// CPU implementation
void sigmoidCPU(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

// Multi-threaded CPU implementation using OpenMP
void sigmoidCPUMultiThreaded(const float* input, float* output, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

} // namespace kernels
} // namespace onnx_runner
