#include "tensor.hpp"

namespace onnx_runner {

// Static factory method to create tensor directly on GPU
std::shared_ptr<Tensor> Tensor::createOnGPU(
    const std::vector<int64_t>& shape,
    DataType dtype
) {
    // Create tensor with shape and dtype (will allocate CPU storage)
    auto tensor = std::make_shared<Tensor>(shape, dtype);

    // Calculate bytes needed
    size_t bytes = tensor->size() * tensor->dataTypeSize();

    // Allocate directly on GPU
    void* gpu_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_ptr, bytes));

    // Zero-initialize to prevent uninitialized memory issues
    CUDA_CHECK(cudaMemset(gpu_ptr, 0, bytes));

    // Set GPU data and mark as CUDA device
    tensor->gpu_data_.reset(gpu_ptr, CudaDeleter());
    tensor->device_ = DeviceType::CUDA;

    // Clear CPU data since we're GPU-only
    tensor->cpu_data_.clear();

    return tensor;
}

} // namespace onnx_runner
