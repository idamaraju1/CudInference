#pragma once

#include <vector>
#include <memory>
#include <string>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

namespace onnx_runner {

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64,
    UINT8
};

enum class DeviceType {
    CPU,
    CUDA
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                cudaGetErrorString(error) + " at " + __FILE__ + ":" + \
                std::to_string(__LINE__)); \
        } \
    } while(0)

// Custom deleter for CUDA memory
struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

class Tensor {
public:
    Tensor() : dtype_(DataType::FLOAT32), device_(DeviceType::CPU) {}

    // Constructor for CPU tensor
    Tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32)
        : shape_(shape), dtype_(dtype), device_(DeviceType::CPU) {
        size_t total_size = computeSize();
        cpu_data_.resize(total_size * dataTypeSize());
    }

    // Constructor with initial data
    Tensor(const std::vector<int64_t>& shape, const std::vector<float>& data,
           DataType dtype = DataType::FLOAT32)
        : shape_(shape), dtype_(dtype), device_(DeviceType::CPU) {
        if (data.size() != computeSize()) {
            throw std::runtime_error("Data size doesn't match shape");
        }
        cpu_data_.resize(data.size() * sizeof(float));
        std::memcpy(cpu_data_.data(), data.data(), cpu_data_.size());
    }

    // Shape accessors
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(size_t idx) const { return shape_[idx]; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return computeSize(); }

    // Data type
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }

    // Data access
    void* data() {
        return device_ == DeviceType::CPU ? cpu_data_.data() : gpu_data_.get();
    }

    const void* data() const {
        return device_ == DeviceType::CPU ? cpu_data_.data() : gpu_data_.get();
    }

    template<typename T>
    T* data() {
        return static_cast<T*>(data());
    }

    template<typename T>
    const T* data() const {
        return static_cast<const T*>(data());
    }

    // Device transfer
    void toGPU() {
        if (device_ == DeviceType::CUDA) return;

        size_t bytes = cpu_data_.size();
        void* gpu_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&gpu_ptr, bytes));
        CUDA_CHECK(cudaMemcpy(gpu_ptr, cpu_data_.data(), bytes, cudaMemcpyHostToDevice));

        gpu_data_.reset(gpu_ptr, CudaDeleter());
        device_ = DeviceType::CUDA;
    }

    void toCPU() {
        if (device_ == DeviceType::CPU) return;

        size_t bytes = size() * dataTypeSize();
        cpu_data_.resize(bytes);
        CUDA_CHECK(cudaMemcpy(cpu_data_.data(), gpu_data_.get(), bytes, cudaMemcpyDeviceToHost));

        gpu_data_.reset();
        device_ = DeviceType::CPU;
    }

    // Allocate GPU memory without copying
    void allocateGPU() {
        if (device_ == DeviceType::CUDA) return;

        size_t bytes = size() * dataTypeSize();
        void* gpu_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&gpu_ptr, bytes));

        gpu_data_.reset(gpu_ptr, CudaDeleter());
        device_ = DeviceType::CUDA;
    }

    // Copy data from another tensor
    void copyFrom(const Tensor& other) {
        if (other.size() != size()) {
            throw std::runtime_error("Tensor sizes don't match for copy");
        }

        if (device_ == DeviceType::CPU && other.device_ == DeviceType::CPU) {
            std::memcpy(cpu_data_.data(), other.cpu_data_.data(), cpu_data_.size());
        } else if (device_ == DeviceType::CUDA && other.device_ == DeviceType::CUDA) {
            CUDA_CHECK(cudaMemcpy(gpu_data_.get(), other.gpu_data_.get(),
                                  size() * dataTypeSize(), cudaMemcpyDeviceToDevice));
        } else if (device_ == DeviceType::CUDA && other.device_ == DeviceType::CPU) {
            CUDA_CHECK(cudaMemcpy(gpu_data_.get(), other.cpu_data_.data(),
                                  size() * dataTypeSize(), cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(cpu_data_.data(), other.gpu_data_.get(),
                                  size() * dataTypeSize(), cudaMemcpyDeviceToHost));
        }
    }

    // Reshape (view only, doesn't copy data)
    void reshape(const std::vector<int64_t>& new_shape) {
        if (computeSize(new_shape) != size()) {
            throw std::runtime_error("New shape has different total size");
        }
        shape_ = new_shape;
    }

    // Fill with constant value (CPU only for simplicity)
    void fill(float value) {
        if (device_ == DeviceType::CUDA) {
            toCPU();
        }
        float* ptr = data<float>();
        for (size_t i = 0; i < size(); ++i) {
            ptr[i] = value;
        }
    }

    // Debug: print shape
    std::string shapeStr() const {
        std::string result = "[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            result += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) result += ", ";
        }
        result += "]";
        return result;
    }

private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    DeviceType device_;

    // CPU storage
    std::vector<uint8_t> cpu_data_;

    // GPU storage with smart pointer
    std::shared_ptr<void> gpu_data_{nullptr, CudaDeleter()};

    size_t computeSize() const {
        return computeSize(shape_);
    }

    size_t computeSize(const std::vector<int64_t>& shape) const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    }

    size_t dataTypeSize() const {
        switch (dtype_) {
            case DataType::FLOAT32: return sizeof(float);
            case DataType::FLOAT16: return 2;
            case DataType::INT32: return sizeof(int32_t);
            case DataType::INT64: return sizeof(int64_t);
            case DataType::UINT8: return sizeof(uint8_t);
            default: return sizeof(float);
        }
    }
};

} // namespace onnx_runner
