#include "cpu_kernels.h"
#include <omp.h>

namespace onnx_runner {

// CPU implementation with OpenMP
void gatherCPU(
    const float* data,
    const int64_t* indices,
    float* output,
    int64_t axis_dim_data,
    int64_t axis_dim_indices,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_size,
    int num_threads
) {
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t idx = 0; idx < total_size; ++idx) {
        // Decompose linear index
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t axis_idx = temp % axis_dim_indices;
        int64_t outer_idx = temp / axis_dim_indices;

        // Get the index to gather from
        int64_t gather_idx = indices[outer_idx * axis_dim_indices + axis_idx];

        // Handle negative indices
        if (gather_idx < 0) {
            gather_idx += axis_dim_data;
        }

        // Compute source position
        int64_t src_idx = (outer_idx * axis_dim_data + gather_idx) * inner_size + inner_idx;

        output[idx] = data[src_idx];
    }
}

} // namespace onnx_runner

