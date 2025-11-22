#include "cpu_kernels.h"
#include <omp.h>
#include <vector>

namespace onnx_runner {

// ReduceMean: Compute mean along specified axes
// This implementation handles reduction along one or more axes

void reduceMeanCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& axes,
    int num_threads
) {
    // Compute output shape
    std::vector<int64_t> output_shape;
    std::vector<bool> is_reduced(input_shape.size(), false);

    for (int64_t axis : axes) {
        is_reduced[axis] = true;
    }

    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (!is_reduced[i]) {
            output_shape.push_back(input_shape[i]);
        }
    }

    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    // Compute strides
    std::vector<int64_t> input_strides(input_shape.size());
    input_strides[input_shape.size() - 1] = 1;
    for (int i = input_shape.size() - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Compute total output size
    int64_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    // Compute reduction count
    int64_t reduce_count = 1;
    for (int64_t axis : axes) {
        reduce_count *= input_shape[axis];
    }

    // Initialize output to zero
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < output_size; ++i) {
        output[i] = 0.0f;
    }

    // Simple approach: iterate over all input elements
    int64_t input_size = 1;
    for (auto dim : input_shape) {
        input_size *= dim;
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int64_t idx = 0; idx < input_size; ++idx) {
        // Compute multi-dimensional index
        std::vector<int64_t> multi_idx(input_shape.size());
        int64_t temp = idx;
        for (int i = input_shape.size() - 1; i >= 0; --i) {
            multi_idx[i] = temp % input_shape[i];
            temp /= input_shape[i];
        }

        // Compute output index (skipping reduced dimensions)
        int64_t output_idx = 0;
        int64_t output_stride = 1;
        for (int i = output_shape.size() - 1; i >= 0; --i) {
            // Find which input dimension this corresponds to
            int input_dim = 0;
            int output_dim_count = 0;
            for (size_t j = 0; j < input_shape.size(); ++j) {
                if (!is_reduced[j]) {
                    if (output_dim_count == i) {
                        input_dim = j;
                        break;
                    }
                    output_dim_count++;
                }
            }

            output_idx += multi_idx[input_dim] * output_stride;
            output_stride *= output_shape[i];
        }

        #pragma omp atomic
        output[output_idx] += input[idx];
    }

    // Divide by count
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < output_size; ++i) {
        output[i] /= reduce_count;
    }
}

} // namespace onnx_runner

