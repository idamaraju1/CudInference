#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <stdexcept> 
#include <string>   

namespace onnx_runner {

// Simple GPU kernel using atomic adds for reduction
// This is a straightforward approach - for better performance, use shared memory reductions
__global__ void reduceSumKernel(
    const float* input,
    float* output,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const bool* reduce_mask,
    int64_t ndim,
    int64_t total_input,
    int64_t output_size,
    bool keepdims
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_input) {
        // Decompose linear index into coordinates
        int64_t remainder = idx;
        int64_t coords[8];  // Max 8 dimensions
        for (int64_t dim = 0; dim < ndim; ++dim) {
            coords[dim] = remainder / input_strides[dim];
            remainder %= input_strides[dim];
        }

        // Compute output index
        int64_t out_idx = 0;
        int64_t out_dim = 0;
        for (int64_t dim = 0; dim < ndim; ++dim) {
            if (reduce_mask[dim]) {
                if (keepdims) {
                    // Coordinate becomes 0
                    out_dim++;
                }
            } else {
                int64_t stride = keepdims ? output_strides[out_dim] :
                                 (out_dim < ndim ? output_strides[out_dim] : 1);
                out_idx += coords[dim] * stride;
                out_dim++;
            }
        }

        // Atomic add to output
        atomicAdd(&output[out_idx], input[idx]);
    }
}

/**
 * Faster path for reduce sum last dimension (see reducemean.cu) 
 */
__global__ void reduceSumLastDimKernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int64_t outer_size,
                                       int64_t inner_size) {
    int row = blockIdx.x;
    if (row >= outer_size) return;

    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread walks part of the row
    for (int64_t col = tid; col < inner_size; col += blockDim.x) {
        sum += input[row * inner_size + col];
    }

    __shared__ float sdata[256];  // assume blockDim.x <= 256
    sdata[tid] = sum;
    __syncthreads();

    // Block-wide reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = sdata[0];
    }
}


// Optimized single-axis reduction using warp shuffles
__global__ void reduceSumSingleAxisKernel(
    const float* input,
    float* output,
    int64_t outer_size,
    int64_t reduce_dim,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.y;
    int64_t inner_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    float sum = 0.0f;
    int64_t base = outer_idx * reduce_dim * inner_size + inner_idx;

    // Accumulate across reduction dimension
    for (int64_t r = 0; r < reduce_dim; ++r) {
        sum += input[base + r * inner_size];
    }

    output[outer_idx * inner_size + inner_idx] = sum;
}

// CPU implementation
void reduceSumCPU(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& output_strides,
    const std::vector<bool>& reduce_mask,
    int64_t ndim,
    int64_t total_input,
    int64_t output_size,
    bool keepdims,
    int num_threads
) {
    // Initialize output to zero
    std::memset(output, 0, output_size * sizeof(float));

    #pragma omp parallel for num_threads(num_threads)
    for (int64_t idx = 0; idx < total_input; ++idx) {
        // Decompose linear index into coordinates
        int64_t remainder = idx;
        std::vector<int64_t> coords(ndim);
        for (int64_t dim = 0; dim < ndim; ++dim) {
            coords[dim] = remainder / input_strides[dim];
            remainder %= input_strides[dim];
        }

        // Compute output index
        int64_t out_idx = 0;
        int64_t out_dim = 0;
        for (int64_t dim = 0; dim < ndim; ++dim) {
            if (reduce_mask[dim]) {
                if (keepdims) {
                    out_dim++;
                }
            } else {
                int64_t stride = keepdims ? output_strides[out_dim] :
                                 (out_dim < static_cast<int64_t>(output_strides.size()) ?
                                  output_strides[out_dim] : 1);
                out_idx += coords[dim] * stride;
                out_dim++;
            }
        }

        #pragma omp atomic
        output[out_idx] += input[idx];
    }
}

// Launcher function
void launchReduceSum(
    const float* input,
    float* output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    const std::vector<bool>& reduce_mask,
    bool keepdims,
    bool use_cpu,
    int num_threads
) {
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Compute strides
    std::vector<int64_t> input_strides(ndim, 1);
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        input_strides[i] = stride;
        stride *= input_shape[i];
    }
    int64_t total_input = stride;

    std::vector<int64_t> output_strides(output_shape.size(), 1);
    stride = 1;
    for (int64_t i = static_cast<int64_t>(output_shape.size()) - 1; i >= 0; --i) {
        output_strides[i] = stride;
        stride *= output_shape[i];
    }
    int64_t output_size = stride;

    if (use_cpu) {
        reduceSumCPU(input, output, input_strides, output_strides, reduce_mask,
                     ndim, total_input, output_size, keepdims, num_threads);
        return;
    }
    // Initialize output to zero
    cudaMemset(output, 0, output_size * sizeof(float));

    // Check if this is a single-axis reduction (more common and efficient)
    int num_reduce_axes = 0;
    int64_t reduce_axis = -1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_mask[i]) {
            num_reduce_axes++;
            reduce_axis = i;
        }
    }

    // fast path.
    if (num_reduce_axes == 1 && reduce_axis == ndim - 1) {
        int64_t inner = input_shape.back();  // last dim
        int64_t outer = 1;
        for (int64_t i = 0; i < ndim - 1; ++i) {
            outer *= input_shape[i];
        }

        if (outer > 0 && inner > 0) {
            int block = 256;                 // must match sdata size in kernel
            int grid  = static_cast<int>(outer);

            reduceSumLastDimKernel<<<grid, block>>>(input, output, outer, inner);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("ReduceSumLastDim kernel failed: ") +
                                         cudaGetErrorString(err));
            }
        }

        cudaDeviceSynchronize();
        return;
    }

    if (num_reduce_axes == 1 && reduce_axis >= 0) {
        // Optimized path for single-axis reduction
        int64_t outer_size = 1;
        for (int64_t i = 0; i < reduce_axis; ++i) {
            outer_size *= input_shape[i];
        }
        int64_t reduce_dim = input_shape[reduce_axis];
        int64_t inner_size = 1;
        for (int64_t i = reduce_axis + 1; i < ndim; ++i) {
            inner_size *= input_shape[i];
        }

        int block_size = 256;
        dim3 grid_size((inner_size + block_size - 1) / block_size, outer_size);
        reduceSumSingleAxisKernel<<<grid_size, block_size>>>(
            input, output, outer_size, reduce_dim, inner_size
        );
        return;
    } 
    // General path using atomic adds
    int64_t* d_input_strides;
    int64_t* d_output_strides;
    bool* d_reduce_mask;

    cudaMalloc(&d_input_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_output_strides, output_strides.size() * sizeof(int64_t));
    cudaMalloc(&d_reduce_mask, ndim * sizeof(bool));

    cudaMemcpy(d_input_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_strides, output_strides.data(), output_strides.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Convert vector<bool> to regular bool array for GPU transfer
    std::vector<bool> reduce_mask_copy(reduce_mask.begin(), reduce_mask.end());
    bool* temp_mask = new bool[ndim];
    for (int64_t i = 0; i < ndim; ++i) {
        temp_mask[i] = reduce_mask[i];
    }
    cudaMemcpy(d_reduce_mask, temp_mask, ndim * sizeof(bool), cudaMemcpyHostToDevice);
    delete[] temp_mask;

    int block_size = 256;
    int grid_size = (total_input + block_size - 1) / block_size;

    reduceSumKernel<<<grid_size, block_size>>>(
        input, output, d_input_strides, d_output_strides, d_reduce_mask,
        ndim, total_input, output_size, keepdims
    );

    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    cudaFree(d_reduce_mask);

    cudaDeviceSynchronize();
}

} // namespace onnx_runner
