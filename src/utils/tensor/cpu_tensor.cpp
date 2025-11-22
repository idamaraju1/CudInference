#include "cpu_tensor.hpp"
#ifndef CPU_ONLY
#include "gpu_tensor.hpp"
#include <cuda_runtime.h>
#endif
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace onnx_runner {

CpuTensor::CpuTensor(const std::vector<int64_t>& shape, DataType dtype)
    : TensorBase(shape, dtype) {
    size_t total_size = computeSize();
    cpu_data_.resize(total_size * dataTypeSize());
}

CpuTensor::CpuTensor(const std::vector<int64_t>& shape, const std::vector<float>& data,
                     DataType dtype)
    : TensorBase(shape, dtype) {
    if (data.size() != computeSize()) {
        throw std::runtime_error("Data size doesn't match shape");
    }
    cpu_data_.resize(data.size() * sizeof(float));
    std::memcpy(cpu_data_.data(), data.data(), cpu_data_.size());
}

std::shared_ptr<TensorBase> CpuTensor::toCPU() const {
    // Already on CPU
    return const_cast<CpuTensor*>(this)->shared_from_this();
}

std::shared_ptr<TensorBase> CpuTensor::toGPU() const {
#ifdef CPU_ONLY
    throw std::runtime_error("GPU operations are disabled (CPU_ONLY)");
#else
    auto gpu_tensor = std::make_shared<GpuTensor>(shape_, dtype_);
    gpu_tensor->copyFrom(*this);
    return gpu_tensor;
#endif
}

void CpuTensor::copyFrom(const TensorBase& other) {
    if (other.size() != size()) {
        throw std::runtime_error("Tensor sizes don't match for copy");
    }

    size_t bytes = size() * dataTypeSize();

    if (other.device() == DeviceType::CPU) {
        // CPU to CPU copy
        const CpuTensor& cpu_other = static_cast<const CpuTensor&>(other);
        std::memcpy(cpu_data_.data(), cpu_other.cpu_data_.data(), bytes);
    }
#ifdef CPU_ONLY
    else {
        throw std::runtime_error("GPU copy not allowed (CPU_ONLY)");
    }
#else
    else {
        // GPU to CPU copy
        const GpuTensor& gpu_other = static_cast<const GpuTensor&>(other);
        cudaError_t error = cudaMemcpy(cpu_data_.data(), gpu_other.data(),
                                       bytes, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in copyFrom: ") +
                                     cudaGetErrorString(error));
        }
    }
#endif
}

void CpuTensor::fill(float value) {
    // float* ptr = data<float>();
    float* ptr = static_cast<float*>(data());
    for (size_t i = 0; i < size(); ++i) {
        ptr[i] = value;
    }
}

} // namespace onnx_runner
