#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>
#include <stdexcept>
#include <string>

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

/**
 * For standard row major tensors, you can optimize by only reducing over
 * the last dimension.
 */
__global__ void reduceMeanLastDimKernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int64_t outer_size,
                                        int64_t inner_size) {
    int row = blockIdx.x;
    if (row >= outer_size) return;

    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread traverses part of the row
    for (int64_t col = tid; col < inner_size; col += blockDim.x) {
        sum += input[row * inner_size + col];
    }

    // Block-wide reduction
    __shared__ float sdata[256];  // assumes blockDim.x <= 256
    sdata[tid] = sum;
    __syncthreads();

    // Reduce in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write mean
    if (tid == 0) {
        output[row] = sdata[0] / static_cast<float>(inner_size);
    }
}


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
    const std::vector<int64_t>& axes,
    bool use_cpu,
    int num_threads
) {
    if (use_cpu) {
        reduceMeanCPU(input, output, input_shape, axes, num_threads);
        return;
    }

    // Fast path: reduce over last dimension only (typical for LayerNorm)
    const int ndim = static_cast<int>(input_shape.size());
    if (axes.size() == 1 && axes[0] == ndim - 1) {
        int64_t inner = input_shape.back();  // size of last dimension
        int64_t outer = 1;
        for (int i = 0; i < ndim - 1; ++i) outer *= input_shape[i];

        if (outer == 0 || inner == 0) return;

        int block = 256;  // must match shared memory size in kernel
        int grid  = static_cast<int>(outer);

        reduceMeanLastDimKernel<<<grid, block>>>(input, output, outer, inner);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("ReduceMeanLastDim kernel failed: ") +
                                     cudaGetErrorString(err));
        }
        return;
    }
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
