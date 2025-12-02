#pragma once

#include "../utils/tensor.hpp"
#include <vector>
#include <memory>
#include <cstdint>

namespace onnx_runner {
namespace operation_utils {

// Convert ONNX type enum to DataType
DataType mapONNXTypeToDataType(int onnx_type);

// Convert DataType to string
std::string dataTypeToString(DataType dtype);

// Get size in bytes of a data type
size_t dataTypeSize(DataType dtype);

// Get host data from tensor (handles GPU->CPU copy if needed)
template <typename T>
const T* getHostData(const std::shared_ptr<Tensor>& tensor,
                     std::vector<uint8_t>& host_cache);

// Cast array from SrcT to DstT
template <typename SrcT, typename DstT>
void castArray(const SrcT* src, DstT* dst, size_t count);

// Dispatch cast operation to target type
template <typename SrcT>
void dispatchCastToTarget(const SrcT* src,
                          DataType target_dtype,
                          const std::shared_ptr<Tensor>& output,
                          size_t count);

// Compute strides from shape
std::vector<int64_t> computeStrides(const std::vector<int64_t>& shape);

// Compute total size from shape
int64_t computeSizeFromShape(const std::vector<int64_t>& shape);

// Compute broadcast shape from two input shapes
bool computeBroadcastShape(const std::vector<int64_t>& shapeA,
                           const std::vector<int64_t>& shapeB,
                           std::vector<int64_t>& out_shape);

// Broadcast copy from input to output
template <typename T>
void broadcastCopy(const T* input,
                   T* output,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& output_shape);

// Broadcast to buffer (allocates output)
template <typename T>
void broadcastToBuffer(const T* input,
                       const std::vector<int64_t>& input_shape,
                       std::vector<T>& output,
                       const std::vector<int64_t>& output_shape);

// Convert tensor to shape vector
std::vector<int64_t> tensorToShapeVector(const std::shared_ptr<Tensor>& tensor);

// Read scalar as int64
int64_t readScalarAsInt(const std::shared_ptr<Tensor>& tensor);

// Read scalar as double
double readScalarAsDouble(const std::shared_ptr<Tensor>& tensor);

// Transpose matrix (simple 2D transpose)
void transposeMatrix(const float* input, float* output, int rows, int cols);

} // namespace operation_utils
} // namespace onnx_runner
