#include "cpu_kernels.h"
#include <algorithm>

namespace onnx_runner {
namespace kernels {

// CPU implementation
void reluCPU(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// Multi-threaded CPU implementation using OpenMP
void reluCPUMultiThreaded(const float* input, float* output, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

} // namespace kernels
} // namespace onnx_runner
