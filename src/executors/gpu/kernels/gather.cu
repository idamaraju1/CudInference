#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace onnx_runner {

// CUDA kernel for Gather operation
// Gathers slices from data along the specified axis according to indices
__global__ void gatherKernel(
    const float* data,
    const int64_t* indices,
    float* output,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_size) {
        // Decompose linear index into outer, axis, inner components
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t axis_idx = temp % axis_dim_indices;
        int64_t outer_idx = temp / axis_dim_indices;

        // Get the index to gather from
        int64_t gather_idx = indices[outer_idx * axis_dim_indices + axis_idx];

        // Handle negative indices (Python-style)
        if (gather_idx < 0) {
            gather_idx += axis_dim_data;
        }

        // Compute source position
        int64_t src_idx = (outer_idx * axis_dim_data + gather_idx) * inner_size + inner_idx;

        output[idx] = data[src_idx];
    }
}

void launchGatherKernel(
    const float* data,
    const int64_t* indices,
    float* output,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size
) {
    int64_t total_size = outer_size * axis_dim_indices * inner_size;

    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    gatherKernel<<<grid_size, block_size>>>(
        data, indices, output,
        axis_dim_data, axis_dim_indices,
        outer_size, inner_size, total_size
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") +
                               cudaGetErrorString(error));
    }
}

} // namespace onnx_runner

