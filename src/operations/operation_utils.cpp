#include "operation_utils.hpp"
#include "../utils/logger.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cuda_fp16.h>

namespace onnx_runner {
namespace operation_utils {

DataType mapONNXTypeToDataType(int onnx_type) {
    switch (onnx_type) {
        case 1:  // FLOAT
            return DataType::FLOAT32;
        case 2:  // UINT8
            return DataType::UINT8;
        case 6:  // INT32
            return DataType::INT32;
        case 7:  // INT64
            return DataType::INT64;
        case 10: // FLOAT16
            return DataType::FLOAT16;
        default:
            throw std::runtime_error("Cast: Unsupported target data type id " +
                                     std::to_string(onnx_type));
    }
}

std::string dataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "FLOAT32";
        case DataType::FLOAT16: return "FLOAT16";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::UINT8: return "UINT8";
        default: return "UNKNOWN";
    }
}

size_t dataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::FLOAT16: return 2;
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT64: return sizeof(int64_t);
        case DataType::UINT8: return sizeof(uint8_t);
        default:
            throw std::runtime_error("Unsupported data type size query");
    }
}

template <typename SrcT, typename DstT>
void castArray(const SrcT* src, DstT* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<DstT>(src[i]);
    }
}

template <typename SrcT>
const SrcT* getHostData(const std::shared_ptr<Tensor>& tensor,
                        std::vector<uint8_t>& host_cache) {
    // First try to get CPU data directly if available
    if (tensor->device() == DeviceType::CPU) {
        return tensor->data<SrcT>();
    }

    // If tensor is on GPU, we need to copy it to host
    // First, ensure the tensor is actually on GPU and has data there
    if (!tensor->isOnGPU()) {
        // Tensor claims to be on CPU, return CPU data
        return tensor->data<SrcT>();
    }

    // Tensor is on GPU - copy to host cache
    size_t bytes = tensor->size() * sizeof(SrcT);
    if (bytes == 0) {
        throw std::runtime_error("getHostData: tensor has zero size");
    }

    host_cache.resize(bytes);

    // Use deviceData() for GPU tensors (gets the device pointer)
    const SrcT* device_ptr = nullptr;
    try {
        device_ptr = tensor->deviceData<SrcT>();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("getHostData: failed to get device pointer: ") + e.what());
    }

    if (!device_ptr) {
        throw std::runtime_error("getHostData: tensor is on GPU but deviceData() returned null");
    }

    // Clear any previous CUDA errors (they are sticky!)
    cudaError_t prev_err = cudaGetLastError();
    if (prev_err != cudaSuccess) {
        // Log warning but continue - the previous error might be from a different operation
        // that already handled it
    }

    // Copy from GPU to CPU
    cudaError_t err = cudaMemcpy(host_cache.data(), device_ptr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("getHostData: cudaMemcpy failed: ") +
                               cudaGetErrorString(err) +
                               " (size=" + std::to_string(bytes) +
                               ", device_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(device_ptr)) + ")" +
                               (prev_err != cudaSuccess ? std::string(" [previous error: ") + cudaGetErrorString(prev_err) + "]" : ""));
    }

    return reinterpret_cast<const SrcT*>(host_cache.data());
}

template <typename SrcT>
void dispatchCastToTarget(const SrcT* src,
                          DataType target_dtype,
                          const std::shared_ptr<Tensor>& output,
                          size_t count) {
    switch (target_dtype) {
        case DataType::FLOAT32:
            castArray(src, output->data<float>(), count);
            break;
        case DataType::INT32:
            castArray(src, output->data<int32_t>(), count);
            break;
        case DataType::INT64:
            castArray(src, output->data<int64_t>(), count);
            break;
        case DataType::UINT8:
            castArray(src, output->data<uint8_t>(), count);
            break;
        case DataType::FLOAT16:
            throw std::runtime_error("Cast: FLOAT16 output not supported in executor");
        default:
            throw std::runtime_error("Cast: Unsupported target dtype " +
                                     dataTypeToString(target_dtype));
    }
}

std::vector<int64_t> computeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

int64_t computeSizeFromShape(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;
    int64_t total = 1;
    for (auto dim : shape) {
        total *= dim;
    }
    return total;
}

bool computeBroadcastShape(const std::vector<int64_t>& shapeA,
                           const std::vector<int64_t>& shapeB,
                           std::vector<int64_t>& out_shape) {
    size_t rankA = shapeA.size();
    size_t rankB = shapeB.size();
    size_t out_rank = std::max(rankA, rankB);
    out_shape.assign(out_rank, 1);

    for (size_t i = 0; i < out_rank; ++i) {
        int64_t dimA = (i < out_rank - rankA) ? 1 : shapeA[i - (out_rank - rankA)];
        int64_t dimB = (i < out_rank - rankB) ? 1 : shapeB[i - (out_rank - rankB)];

        if (dimA == dimB) {
            out_shape[i] = dimA;
        } else if (dimA == 1) {
            out_shape[i] = dimB;
        } else if (dimB == 1) {
            out_shape[i] = dimA;
        } else {
            return false;
        }
    }
    return true;
}

template <typename T>
void broadcastCopy(const T* input,
                   T* output,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& output_shape) {
    size_t output_size = computeSizeFromShape(output_shape);
    if (output_size == 0) return;

    auto input_strides = computeStrides(input_shape);
    auto output_strides = computeStrides(output_shape);

    int64_t input_rank = static_cast<int64_t>(input_shape.size());
    int64_t output_rank = static_cast<int64_t>(output_shape.size());
    int64_t offset = output_rank - input_rank;

    for (size_t idx = 0; idx < output_size; ++idx) {
        int64_t remainder = static_cast<int64_t>(idx);
        int64_t input_index = 0;

        for (int64_t dim = 0; dim < output_rank; ++dim) {
            int64_t coord = output_strides[dim] == 0 ? 0 : remainder / output_strides[dim];
            remainder %= output_strides[dim];

            if (dim >= offset) {
                int64_t input_dim = dim - offset;
                int64_t input_coord = (input_shape[input_dim] == 1) ? 0 : coord;
                input_index += input_coord * input_strides[input_dim];
            }
        }

        output[idx] = input[input_index];
    }
}

template <typename T>
void broadcastToBuffer(const T* input,
                       const std::vector<int64_t>& input_shape,
                       std::vector<T>& output,
                       const std::vector<int64_t>& output_shape) {
    size_t out_size = computeSizeFromShape(output_shape);
    output.resize(out_size);
    broadcastCopy(input, output.data(), input_shape, output_shape);
}

std::vector<int64_t> tensorToShapeVector(const std::shared_ptr<Tensor>& tensor) {
    std::vector<int64_t> result;
    size_t count = tensor->size();

    switch (tensor->dtype()) {
        case DataType::INT64: {
            std::vector<uint8_t> cache;
            const int64_t* data = getHostData<int64_t>(tensor, cache);
            result.assign(data, data + count);
            break;
        }
        case DataType::INT32: {
            std::vector<uint8_t> cache;
            const int32_t* data = getHostData<int32_t>(tensor, cache);
            result.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                int64_t val = static_cast<int64_t>(data[i]);
                if (val < 0) {
                    throw std::runtime_error(
                        "tensorToShapeVector: invalid negative dimension " +
                        std::to_string(val) + " at index " + std::to_string(i)
                    );
                }
                result.push_back(val);
            }
            break;
        }
        default: {
            std::vector<uint8_t> cache;
            const float* data = getHostData<float>(tensor, cache);
            result.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                float val = data[i];
                if (std::isnan(val) || std::isinf(val)) {
                    throw std::runtime_error(
                        "tensorToShapeVector: invalid shape value (NaN or Inf) at index " +
                        std::to_string(i)
                    );
                }
                int64_t dim = static_cast<int64_t>(std::round(val));
                if (dim < 0) {
                    throw std::runtime_error(
                        "tensorToShapeVector: invalid negative dimension " +
                        std::to_string(dim) + " at index " + std::to_string(i)
                    );
                }
                result.push_back(dim);
            }
            break;
        }
    }

    return result;
}

template <typename T>
T readScalarValue(const std::shared_ptr<Tensor>& tensor) {
    if (tensor->size() != 1) {
        throw std::runtime_error("Expected scalar tensor");
    }
    std::vector<uint8_t> cache;
    const T* data_ptr = getHostData<T>(tensor, cache);
    return data_ptr[0];
}

double readScalarAsDouble(const std::shared_ptr<Tensor>& tensor) {
    switch (tensor->dtype()) {
        case DataType::FLOAT32:
            return static_cast<double>(readScalarValue<float>(tensor));
        case DataType::INT32:
            return static_cast<double>(readScalarValue<int32_t>(tensor));
        case DataType::INT64:
            return static_cast<double>(readScalarValue<int64_t>(tensor));
        case DataType::UINT8:
            return static_cast<double>(readScalarValue<uint8_t>(tensor));
        default:
            throw std::runtime_error("Unsupported scalar data type: " +
                                     dataTypeToString(tensor->dtype()));
    }
}

int64_t readScalarAsInt(const std::shared_ptr<Tensor>& tensor) {
    return static_cast<int64_t>(readScalarAsDouble(tensor));
}

void transposeMatrix(const float* input, float* output, int rows, int cols) {
    // Simple CPU-based matrix transpose
    // input is rows x cols, output will be cols x rows
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// Explicit template instantiations for common types
template const float* getHostData<float>(const std::shared_ptr<Tensor>&, std::vector<uint8_t>&);
template const int32_t* getHostData<int32_t>(const std::shared_ptr<Tensor>&, std::vector<uint8_t>&);
template const int64_t* getHostData<int64_t>(const std::shared_ptr<Tensor>&, std::vector<uint8_t>&);
template const uint8_t* getHostData<uint8_t>(const std::shared_ptr<Tensor>&, std::vector<uint8_t>&);
template const __half* getHostData<__half>(const std::shared_ptr<Tensor>&, std::vector<uint8_t>&);

template void dispatchCastToTarget<float>(const float*, DataType, const std::shared_ptr<Tensor>&, size_t);
template void dispatchCastToTarget<int32_t>(const int32_t*, DataType, const std::shared_ptr<Tensor>&, size_t);
template void dispatchCastToTarget<int64_t>(const int64_t*, DataType, const std::shared_ptr<Tensor>&, size_t);
template void dispatchCastToTarget<uint8_t>(const uint8_t*, DataType, const std::shared_ptr<Tensor>&, size_t);

template void broadcastCopy<float>(const float*, float*, const std::vector<int64_t>&, const std::vector<int64_t>&);
template void broadcastToBuffer<float>(const float*, const std::vector<int64_t>&, std::vector<float>&, const std::vector<int64_t>&);

} // namespace operation_utils
} // namespace onnx_runner
