#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>

namespace onnx_runner {

// GPU kernel for ReduceMean
__global__ void reduceMeanKernel(
    const float* input,
    float* output,
    const int64_t* input_shape_gpu,
    const int64_t* output_shape_gpu,
    const bool* is_reduced_gpu,
    int64_t input_size,
    int64_t output_size,
    int ndim,
    float scale
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size) {
        // Compute multi-dimensional index
        int64_t temp = idx;
        int64_t multi_idx[8];  // Support up to 8 dimensions
        for (int i = ndim - 1; i >= 0; --i) {
            multi_idx[i] = temp % input_shape_gpu[i];
            temp /= input_shape_gpu[i];
        }

        // Compute output index
        int64_t output_idx = 0;
        int64_t output_stride = 1;

        int output_dim = 0;
        for (int i = 0; i < ndim; ++i) {
            if (!is_reduced_gpu[i]) {
                output_dim++;
            }
        }

        for (int i = output_dim - 1; i >= 0; --i) {
            int input_dim = 0;
            int dim_count = 0;
            for (int j = 0; j < ndim; ++j) {
                if (!is_reduced_gpu[j]) {
                    if (dim_count == i) {
                        input_dim = j;
                        break;
                    }
                    dim_count++;
                }
            }
            output_idx += multi_idx[input_dim] * output_stride;
            int64_t out_shape_i = 0;
            int count = 0;
            for (int j = 0; j < ndim; ++j) {
                if (!is_reduced_gpu[j]) {
                    if (count == i) {
                        out_shape_i = input_shape_gpu[j];
                        break;
                    }
                    count++;
                }
            }
            output_stride *= out_shape_i;
        }

        atomicAdd(&output[output_idx], input[idx] * scale);
    }
}

void launchReduceMeanKernel(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& axes
) {
    // Prepare data for GPU
    std::vector<bool> is_reduced(input_shape.size(), false);
    for (int64_t axis : axes) {
        is_reduced[axis] = true;
    }

    int64_t input_size = 1;
    for (auto dim : input_shape) {
        input_size *= dim;
    }

    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (!is_reduced[i]) {
            output_shape.push_back(input_shape[i]);
        }
    }
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    int64_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    int64_t reduce_count = 1;
    for (int64_t axis : axes) {
        reduce_count *= input_shape[axis];
    }
    float scale = 1.0f / reduce_count;

    // Allocate and copy metadata to GPU
    // Convert vector<bool> to vector<char> since vector<bool> is specialized
    std::vector<char> is_reduced_char(is_reduced.begin(), is_reduced.end());

    int64_t* input_shape_gpu;
    int64_t* output_shape_gpu;
    bool* is_reduced_gpu;

    cudaMalloc(&input_shape_gpu, input_shape.size() * sizeof(int64_t));
    cudaMalloc(&output_shape_gpu, output_shape.size() * sizeof(int64_t));
    cudaMalloc(&is_reduced_gpu, is_reduced.size() * sizeof(bool));

    cudaMemcpy(input_shape_gpu, input_shape.data(), input_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(output_shape_gpu, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(is_reduced_gpu, is_reduced_char.data(), is_reduced_char.size() * sizeof(char), cudaMemcpyHostToDevice);

    // Initialize output to zero
    cudaMemset(output, 0, output_size * sizeof(float));

    // Launch kernel
    int block_size = 256;
    int grid_size = (input_size + block_size - 1) / block_size;

    reduceMeanKernel<<<grid_size, block_size>>>(
        input, output,
        input_shape_gpu, output_shape_gpu, is_reduced_gpu,
        input_size, output_size, input_shape.size(), scale
    );

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(input_shape_gpu);
    cudaFree(output_shape_gpu);
    cudaFree(is_reduced_gpu);
}

} // namespace onnx_runner

