#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>
#include <cstring>

namespace onnx_runner {

// ============================================================================
// Reshape: Change tensor shape without changing data
// ============================================================================

void launchReshapeKernel(
    const float* input,
    float* output,
    int64_t total_size,
    bool use_cpu,
    int num_threads
) {
    // Reshape is just a memory copy since data layout doesn't change
    if (use_cpu) {
        std::memcpy(output, input, total_size * sizeof(float));
    } else {
        cudaMemcpy(output, input, total_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

// ============================================================================
// Transpose: Permute tensor dimensions
// ============================================================================

__global__ void transposeKernel(
    const float* input,
    float* output,
    const int64_t* input_shape,
    const int64_t* output_strides,
    const int* perm,
    int ndim,
    int64_t total_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_size) {
        // Compute multi-dimensional index in input
        int64_t temp = idx;
        int64_t indices[8];  // Support up to 8 dimensions
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

void launchTransposeKernel(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int>& perm,
    bool use_cpu,
    int num_threads
) {
    if (use_cpu) {
        transposeCPU(input, output, input_shape, perm, num_threads);
    } else {
        int ndim = input_shape.size();

        // Compute output shape and strides
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

        // Allocate and copy metadata to GPU
        int64_t* input_shape_gpu;
        int64_t* output_strides_gpu;
        int* perm_gpu;

        cudaMalloc(&input_shape_gpu, ndim * sizeof(int64_t));
        cudaMalloc(&output_strides_gpu, ndim * sizeof(int64_t));
        cudaMalloc(&perm_gpu, ndim * sizeof(int));

        cudaMemcpy(input_shape_gpu, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(output_strides_gpu, output_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(perm_gpu, perm.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int grid_size = (total_size + block_size - 1) / block_size;

        transposeKernel<<<grid_size, block_size>>>(
            input, output, input_shape_gpu, output_strides_gpu, perm_gpu, ndim, total_size
        );

        cudaDeviceSynchronize();

        // Cleanup
        cudaFree(input_shape_gpu);
        cudaFree(output_strides_gpu);
        cudaFree(perm_gpu);
    }
}

// ============================================================================
// Unsqueeze: Add dimensions of size 1
// ============================================================================

void launchUnsqueezeKernel(
    const float* input,
    float* output,
    int64_t total_size,
    bool use_cpu,
    int num_threads
) {
    // Unsqueeze is just a memory copy since data layout doesn't change
    if (use_cpu) {
        std::memcpy(output, input, total_size * sizeof(float));
    } else {
        cudaMemcpy(output, input, total_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

// ============================================================================
// Slice: Extract a slice from a tensor
// ============================================================================

__global__ void sliceKernel(
    const float* input,
    float* output,
    const int64_t* input_shape,
    const int64_t* starts,
    const int64_t* steps,
    const int64_t* output_shape,
    int ndim,
    int64_t output_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        // Compute multi-dimensional index in output
        int64_t temp = idx;
        int64_t output_indices[8];
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

void launchSliceKernel(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& steps,
    const std::vector<int64_t>& output_shape,
    bool use_cpu,
    int num_threads
) {
    if (use_cpu) {
        sliceCPU(input, output, input_shape, starts, steps, output_shape, num_threads);
    } else {
        int ndim = input_shape.size();
        int64_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }

        // Allocate and copy metadata to GPU
        int64_t* input_shape_gpu;
        int64_t* starts_gpu;
        int64_t* steps_gpu;
        int64_t* output_shape_gpu;

        cudaMalloc(&input_shape_gpu, ndim * sizeof(int64_t));
        cudaMalloc(&starts_gpu, ndim * sizeof(int64_t));
        cudaMalloc(&steps_gpu, ndim * sizeof(int64_t));
        cudaMalloc(&output_shape_gpu, ndim * sizeof(int64_t));

        cudaMemcpy(input_shape_gpu, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(starts_gpu, starts.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(steps_gpu, steps.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(output_shape_gpu, output_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int grid_size = (output_size + block_size - 1) / block_size;

        sliceKernel<<<grid_size, block_size>>>(
            input, output,
            input_shape_gpu, starts_gpu, steps_gpu, output_shape_gpu,
            ndim, output_size
        );

        cudaDeviceSynchronize();

        // Cleanup
        cudaFree(input_shape_gpu);
        cudaFree(starts_gpu);
        cudaFree(steps_gpu);
        cudaFree(output_shape_gpu);
    }
}

// ============================================================================
// Concat: Concatenate tensors along an axis
// ============================================================================

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

void launchConcatKernel(
    const std::vector<const float*>& inputs,
    float* output,
    const std::vector<std::vector<int64_t>>& input_shapes,
    int64_t axis,
    const std::vector<int64_t>& output_shape,
    bool use_cpu,
    int num_threads
) {
    if (use_cpu) {
        concatCPU(inputs, output, input_shapes, axis, output_shape, num_threads);
    } else {
        // GPU concat using multiple memcpy operations
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
            int64_t chunk_size = axis_dim * inner_size * sizeof(float);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const float* src = inputs[input_idx] + outer * axis_dim * inner_size;
                float* dst = output + outer * output_shape[axis] * inner_size + output_offset;
                cudaMemcpy(dst, src, chunk_size, cudaMemcpyDeviceToDevice);
            }

            output_offset += axis_dim * inner_size;
        }
        cudaDeviceSynchronize();
    }
}

} // namespace onnx_runner
