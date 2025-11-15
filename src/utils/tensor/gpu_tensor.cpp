#include "gpu_tensor.hpp"
#include "cpu_tensor.hpp"
#include <cstring>
#include <stdexcept>

namespace onnx_runner {

GpuTensor::GpuTensor(const std::vector<int64_t>& shape, DataType dtype)
    : TensorBase(shape, dtype) {
    allocate();
    zero();  // Zero-initialize by default
}

GpuTensor::GpuTensor(const std::vector<int64_t>& shape, DataType dtype, bool zero_init)
    : TensorBase(shape, dtype) {
    allocate();
    if (zero_init) {
        zero();
    }
}

std::shared_ptr<TensorBase> GpuTensor::toCPU() const {
    // Create CPU tensor and copy data
    auto cpu_tensor = std::make_shared<CpuTensor>(shape_, dtype_);
    cpu_tensor->copyFrom(*this);
    return cpu_tensor;
}

std::shared_ptr<TensorBase> GpuTensor::toGPU() const {
    // Already on GPU, return shared_ptr to self
    // Note: This requires the object to be managed by shared_ptr
    return const_cast<GpuTensor*>(this)->shared_from_this();
}

void GpuTensor::copyFrom(const TensorBase& other) {
    if (other.size() != size()) {
        throw std::runtime_error("Tensor sizes don't match for copy");
    }

    size_t bytes = size() * dataTypeSize();

    if (other.device() == DeviceType::CUDA) {
        // GPU to GPU copy
        const GpuTensor& gpu_other = static_cast<const GpuTensor&>(other);
        CUDA_CHECK(cudaMemcpy(gpu_data_.get(), gpu_other.gpu_data_.get(),
                              bytes, cudaMemcpyDeviceToDevice));
    } else {
        // CPU to GPU copy
        const CpuTensor& cpu_other = static_cast<const CpuTensor&>(other);
        CUDA_CHECK(cudaMemcpy(gpu_data_.get(), cpu_other.data(),
                              bytes, cudaMemcpyHostToDevice));
    }
}

void GpuTensor::allocate() {
    if (gpu_data_) return;  // Already allocated

    size_t bytes = size() * dataTypeSize();
    void* gpu_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_ptr, bytes));
    gpu_data_.reset(gpu_ptr, CudaDeleter());
}

void GpuTensor::zero() {
    if (!gpu_data_) {
        allocate();
    }
    size_t bytes = size() * dataTypeSize();
    CUDA_CHECK(cudaMemset(gpu_data_.get(), 0, bytes));
}

} // namespace onnx_runner

