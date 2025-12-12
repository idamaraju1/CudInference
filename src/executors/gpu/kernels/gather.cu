#include "gpu_kernels.cuh"
#include "utils/tensor.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace onnx_runner {

template <typename T>
__global__ void gatherKernel(
    const T* data,
    const int64_t* indices,
    T* output,
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

        // Indices are pre-validated and normalized on host
        int64_t gather_idx = indices[axis_idx];
        int64_t src_idx = (outer_idx * axis_dim_data + gather_idx) * inner_size + inner_idx;

        output[idx] = data[src_idx];
    }
}

template <typename T>
void gatherCPU(
    const T* data,
    const int64_t* indices,
    T* output,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_size,
    int num_threads
) {
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t idx = 0; idx < total_size; ++idx) {
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t axis_idx = temp % axis_dim_indices;
        int64_t outer_idx = temp / axis_dim_indices;

        int64_t gather_idx = indices[axis_idx];
        int64_t src_idx = (outer_idx * axis_dim_data + gather_idx) * inner_size + inner_idx;

        output[idx] = data[src_idx];
    }
}

template <typename T>
void launchGatherTyped(
    const void* data,
    const int64_t* indices,
    void* output,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size,
    bool use_cpu,
    int num_threads
) {
    int64_t total_size = outer_size * axis_dim_indices * inner_size;
    if (total_size == 0) {
        return;
    }

    if (use_cpu) {
        gatherCPU(static_cast<const T*>(data), indices, static_cast<T*>(output),
                  axis_dim_data, axis_dim_indices, outer_size, inner_size,
                  total_size, num_threads);
    } else {
        constexpr int block_size = 256;
        int64_t grid_size_64 = (total_size + block_size - 1) / block_size;
        if (grid_size_64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Gather kernel launch size too large");
        }
        int grid_size = static_cast<int>(grid_size_64);

        gatherKernel<<<grid_size, block_size>>>(
            static_cast<const T*>(data), indices, static_cast<T*>(output),
            axis_dim_data, axis_dim_indices, outer_size, inner_size, total_size
        );

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel error: ") +
                                     cudaGetErrorString(error));
        }
    }
}

void launchGatherKernel(
    const void* data,
    const int64_t* indices,
    void* output,
    DataType dtype,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size,
    bool use_cpu,
    int num_threads
) {
    switch (dtype) {
        case DataType::FLOAT32:
            launchGatherTyped<float>(data, indices, output, axis_dim_data, axis_dim_indices,
                                     outer_size, inner_size, use_cpu, num_threads);
            break;
        case DataType::INT32:
            launchGatherTyped<int32_t>(data, indices, output, axis_dim_data, axis_dim_indices,
                                       outer_size, inner_size, use_cpu, num_threads);
            break;
        case DataType::INT64:
            launchGatherTyped<int64_t>(data, indices, output, axis_dim_data, axis_dim_indices,
                                       outer_size, inner_size, use_cpu, num_threads);
            break;
        case DataType::UINT8:
            launchGatherTyped<uint8_t>(data, indices, output, axis_dim_data, axis_dim_indices,
                                       outer_size, inner_size, use_cpu, num_threads);
            break;
        case DataType::FLOAT16:
            launchGatherTyped<__half>(data, indices, output, axis_dim_data, axis_dim_indices,
                                      outer_size, inner_size, use_cpu, num_threads);
            break;
        default:
            throw std::runtime_error("Gather: unsupported data type");
    }
}

} // namespace onnx_runner
