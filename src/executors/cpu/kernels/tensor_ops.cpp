#include "cpu_kernels.h"
#include <omp.h>
#include <vector>
#include <cstring>

namespace onnx_runner {

void transposeCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int>& perm,
    int num_threads
) {
    int ndim = input_shape.size();

    // Compute output strides
    std::vector<int64_t> output_shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        output_shape[i] = input_shape[perm[i]];
    }

    std::vector<int64_t> output_strides(ndim);
    output_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    int64_t total_size = 1;
    for (auto dim : input_shape) {
        total_size *= dim;
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int64_t idx = 0; idx < total_size; ++idx) {
        // Compute multi-dimensional index in input
        std::vector<int64_t> indices(ndim);
        int64_t temp = idx;
        for (int i = ndim - 1; i >= 0; --i) {
            indices[i] = temp % input_shape[i];
            temp /= input_shape[i];
        }

        // Compute linear index in output using permutation
        int64_t output_idx = 0;
        for (int i = 0; i < ndim; ++i) {
            output_idx += indices[perm[i]] * output_strides[i];
        }

        output[output_idx] = input[idx];
    }
}

void sliceCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& steps,
    const std::vector<int64_t>& output_shape,
    int num_threads
) {
    int ndim = input_shape.size();
    int64_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int64_t idx = 0; idx < output_size; ++idx) {
        // Compute multi-dimensional index in output
        std::vector<int64_t> output_indices(ndim);
        int64_t temp = idx;
        for (int i = ndim - 1; i >= 0; --i) {
            output_indices[i] = temp % output_shape[i];
            temp /= output_shape[i];
        }

        // Compute corresponding input index
        int64_t input_idx = 0;
        int64_t input_stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            int64_t input_coord = starts[i] + output_indices[i] * steps[i];
            input_idx += input_coord * input_stride;
            input_stride *= input_shape[i];
        }

        output[idx] = input[input_idx];
    }
}

void concatCPU(
    const std::vector<const float*>& inputs,
    float* output,
    const std::vector<std::vector<int64_t>>& input_shapes,
    int64_t axis,
    const std::vector<int64_t>& output_shape,
    int num_threads
) {
    int ndim = output_shape.size();
    int64_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= output_shape[i];
    }

    int64_t inner_size = 1;
    for (size_t i = axis + 1; i < output_shape.size(); ++i) {
        inner_size *= output_shape[i];
    }

    int64_t output_offset = 0;
    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        int64_t axis_dim = input_shapes[input_idx][axis];
        int64_t input_size = axis_dim * inner_size;

        #pragma omp parallel for num_threads(num_threads)
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const float* src = inputs[input_idx] + outer * input_size;
            float* dst = output + outer * output_shape[axis] * inner_size + output_offset;
            std::memcpy(dst, src, input_size * sizeof(float));
        }

        output_offset += axis_dim * inner_size;
    }
}

} // namespace onnx_runner

