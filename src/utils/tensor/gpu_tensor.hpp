#pragma once

#include "tensor_base.hpp"
#include <memory>
#include <cuda_runtime.h>

namespace onnx_runner {

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

/**
 * GPU tensor implementation.
 * Stores data in CUDA device memory.
 */
class GpuTensor : public TensorBase {
public:
    // Constructor - allocates GPU memory
    GpuTensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32);

    // Constructor - allocates and optionally initializes GPU memory
    GpuTensor(const std::vector<int64_t>& shape, DataType dtype, bool zero_init);

    // Destructor - GPU memory is freed by CudaDeleter
    ~GpuTensor() = default;

    // Implement base class interface
    DeviceType device() const override { return DeviceType::CUDA; }
    void* data() override { return gpu_data_.get(); }
    const void* data() const override { return gpu_data_.get(); }

    // Conversion methods
    std::shared_ptr<TensorBase> toCPU() const override;
    std::shared_ptr<TensorBase> toGPU() const override;

    // Copy from another tensor
    void copyFrom(const TensorBase& other) override;

    // Allocate GPU memory (if not already allocated)
    void allocate();

    // Zero-initialize GPU memory
    void zero();

private:
    std::shared_ptr<void> gpu_data_{nullptr, CudaDeleter()};
};

} // namespace onnx_runner

