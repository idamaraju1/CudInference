#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <numeric>
#include <stdexcept>

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

/**
 * Base class for all tensor implementations.
 * Provides common interface for CPU and GPU tensors.
 */
class TensorBase : public std::enable_shared_from_this<TensorBase> {
public:
    virtual ~TensorBase() = default;

    // Shape accessors
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(size_t idx) const { return shape_[idx]; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return computeSize(); }

    // Data type and device
    DataType dtype() const { return dtype_; }
    virtual DeviceType device() const = 0;

    // Data access (must be implemented by derived classes)
    virtual void* data() = 0;
    virtual const void* data() const = 0;

    template<typename T>
    T* data_ptr() {
        return static_cast<T*>(data());
    }

    template<typename T>
    const T* data_ptr() const {
        return static_cast<const T*>(data());
    }

    // Reshape (view only, doesn't copy data)
    void reshape(const std::vector<int64_t>& new_shape) {
        if (computeSize(new_shape) != size()) {
            throw std::runtime_error("New shape has different total size");
        }
        shape_ = new_shape;
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

    // Conversion methods - create a new tensor on a different device
    virtual std::shared_ptr<TensorBase> toCPU() const = 0;
    virtual std::shared_ptr<TensorBase> toGPU() const = 0;

    // Copy data from another tensor (must be same device type)
    virtual void copyFrom(const TensorBase& other) = 0;

protected:
    TensorBase(const std::vector<int64_t>& shape, DataType dtype)
        : shape_(shape), dtype_(dtype) {}

    std::vector<int64_t> shape_;
    DataType dtype_;

    size_t computeSize() const {
        return computeSize(shape_);
    }

    size_t computeSize(const std::vector<int64_t>& shape) const {
        if (shape.empty()) return 1;  // Scalar tensors have size 1
        return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    }

protected:
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

// Type alias for convenience (backward compatibility during migration)
using Tensor = TensorBase;

} // namespace onnx_runner

