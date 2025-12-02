#include "gather_operation.hpp"
#include "operation_registry.hpp"
#include "operation_utils.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"
#include <cuda_fp16.h>
#include <type_traits>
#include <cmath>

namespace onnx_runner {

using namespace operation_utils;

void GatherOperation::execute(const Node& node, ExecutionContext& ctx) {
    // Gather: output = data[indices] along specified axis
    validateInputOutputCount(node, 2, 1);

    auto data = getTensor(node.inputs()[0], ctx);
    auto indices_tensor = getTensor(node.inputs()[1], ctx);
    DataType dtype = data->dtype();

    // Get axis attribute (default 0)
    int64_t axis = node.getIntAttr("axis", 0);
    if (axis < 0) {
        axis += data->ndim();
    }
    if (axis < 0 || axis >= static_cast<int64_t>(data->ndim())) {
        throw std::runtime_error("Gather axis out of range");
    }

    // Materialize indices as int64_t on host
    size_t indices_count = indices_tensor->size();
    std::vector<int64_t> host_indices(indices_count, 0);

    switch (indices_tensor->dtype()) {
        case DataType::INT64: {
            std::vector<uint8_t> cache;
            const int64_t* src = getHostData<int64_t>(indices_tensor, cache);
            std::copy(src, src + indices_count, host_indices.begin());
            break;
        }
        case DataType::INT32: {
            std::vector<uint8_t> cache;
            const int32_t* src = getHostData<int32_t>(indices_tensor, cache);
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(src[i]);
            }
            break;
        }
        case DataType::UINT8: {
            std::vector<uint8_t> cache;
            const uint8_t* src = getHostData<uint8_t>(indices_tensor, cache);
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(src[i]);
            }
            break;
        }
        default: {
            std::vector<uint8_t> cache;
            const float* src = getHostData<float>(indices_tensor, cache);
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(std::llround(src[i]));
            }
            break;
        }
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < axis; ++i) {
        output_shape.push_back(data->dim(i));
    }
    for (size_t i = 0; i < indices_tensor->ndim(); ++i) {
        output_shape.push_back(indices_tensor->dim(i));
    }
    for (size_t i = axis + 1; i < data->ndim(); ++i) {
        output_shape.push_back(data->dim(i));
    }

    auto output = allocateOutput(output_shape, ctx, dtype);

    // Compute dimensions for gather kernel
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= data->dim(i);
    }

    int64_t axis_dim_data = data->dim(axis);
    if (axis_dim_data == 0 && !host_indices.empty()) {
        throw std::runtime_error("Gather axis has zero length but indices are non-empty");
    }

    // Normalize indices
    for (auto& idx : host_indices) {
        if (idx < -axis_dim_data || idx >= axis_dim_data) {
            throw std::runtime_error("Gather index out of range");
        }
        if (idx < 0) {
            idx += axis_dim_data;
        }
    }

    int64_t inner_size = 1;
    for (size_t i = axis + 1; i < data->ndim(); ++i) {
        inner_size *= data->dim(i);
    }

    int64_t axis_dim_indices = static_cast<int64_t>(host_indices.size());

    logDebug(ctx, "  Gather: axis=", axis, ", outer=", outer_size,
              ", axis_dim_data=", axis_dim_data, ", axis_dim_indices=", axis_dim_indices,
              ", inner=", inner_size);

    int64_t total_size = outer_size * axis_dim_indices * inner_size;
    if (total_size == 0) {
        storeOutput(node.outputs()[0], output, ctx);
        return;
    }

    auto dispatchGather = [&](auto* type_tag) {
        using T = typename std::remove_pointer<decltype(type_tag)>::type;

        if (ctx.use_cpu) {
            std::vector<uint8_t> data_cache;
            const T* host_data = getHostData<T>(data, data_cache);

            launchGatherKernel(
                host_data, host_indices.data(), output->data<T>(),
                dtype, axis_dim_data, axis_dim_indices, outer_size, inner_size,
                true, ctx.num_cpu_threads
            );
            storeOutput(node.outputs()[0], output, ctx);
            return;
        }

        // GPU path
        const T* d_data;
        std::vector<uint8_t> data_cache;
        bool need_free_data = false;

        if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT && data->isOnGPU()) {
            d_data = data->deviceData<T>();
        } else if (data->device() == DeviceType::CUDA) {
            d_data = data->data<T>();
        } else {
            const T* host_data = getHostData<T>(data, data_cache);
            T* temp_data;
            size_t data_bytes = data->size() * sizeof(T);
            CUDA_CHECK(cudaMalloc(&temp_data, data_bytes));
            CUDA_CHECK(cudaMemcpy(temp_data, host_data, data_bytes, cudaMemcpyHostToDevice));
            d_data = temp_data;
            need_free_data = true;
        }

        // Allocate and copy indices to GPU
        int64_t* d_indices;
        size_t indices_bytes = host_indices.size() * sizeof(int64_t);
        CUDA_CHECK(cudaMalloc(&d_indices, indices_bytes));
        CUDA_CHECK(cudaMemcpy(d_indices, host_indices.data(), indices_bytes, cudaMemcpyHostToDevice));

        // Get output pointer (GPU or CPU depending on mode)
        T* d_output;
        bool need_copy_output = false;
        if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
            d_output = output->mutableDeviceData<T>();
        } else {
            // GPU_COPY mode: allocate temporary GPU output
            size_t output_bytes = output->size() * sizeof(T);
            CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
            need_copy_output = true;
        }

        launchGatherKernel(
            d_data, d_indices, d_output,
            dtype, axis_dim_data, axis_dim_indices, outer_size, inner_size,
            false, ctx.num_cpu_threads
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to CPU if needed
        if (need_copy_output) {
            size_t output_bytes = output->size() * sizeof(T);
            CUDA_CHECK(cudaMemcpy(output->data<T>(), d_output, output_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_output));
        }

        CUDA_CHECK(cudaFree(d_indices));
        if (need_free_data) {
            CUDA_CHECK(cudaFree(const_cast<T*>(d_data)));
        }
    };

    switch (dtype) {
        case DataType::FLOAT32:
            dispatchGather(static_cast<float*>(nullptr));
            break;
        case DataType::INT32:
            dispatchGather(static_cast<int32_t*>(nullptr));
            break;
        case DataType::INT64:
            dispatchGather(static_cast<int64_t*>(nullptr));
            break;
        case DataType::UINT8:
            dispatchGather(static_cast<uint8_t*>(nullptr));
            break;
        case DataType::FLOAT16:
            dispatchGather(static_cast<__half*>(nullptr));
            break;
        default:
            throw std::runtime_error("Gather: unsupported input dtype");
    }

    storeOutput(node.outputs()[0], output, ctx);
}

// Register the operation
REGISTER_OPERATION(OpType::GATHER, GatherOperation)

} // namespace onnx_runner
