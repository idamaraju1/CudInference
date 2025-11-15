#pragma once

#include "tensor_base.hpp"
#include <vector>
#include <memory>

namespace onnx_runner {

/**
 * CPU tensor implementation.
 * Stores data in CPU memory (std::vector).
 */
class CpuTensor : public TensorBase {
public:
    // Constructors
    CpuTensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32);
    
    CpuTensor(const std::vector<int64_t>& shape, const std::vector<float>& data,
              DataType dtype = DataType::FLOAT32);

    // Implement base class interface
    DeviceType device() const override { return DeviceType::CPU; }
    void* data() override { return cpu_data_.data(); }
    const void* data() const override { return cpu_data_.data(); }

    // Conversion methods
    std::shared_ptr<TensorBase> toCPU() const override;
    std::shared_ptr<TensorBase> toGPU() const override;

    // Copy from another tensor
    void copyFrom(const TensorBase& other) override;

    // Fill with constant value
    void fill(float value);

private:
    std::vector<uint8_t> cpu_data_;
};

} // namespace onnx_runner

