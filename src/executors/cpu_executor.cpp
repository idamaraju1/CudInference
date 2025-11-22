#include "executor.hpp"
#include "cpu_executor.hpp"
#include "cpu/kernels/cpu_kernels.h"
#include "utils/tensor/cpu_tensor.hpp"
#include "core/graph.hpp"
#include "utils/logger.hpp"
#include "../core/node.hpp"
#include <stdexcept>
#include <map>
#include <memory>
#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace onnx_runner {

// Anonymous namespace for helper functions (shared across operations)
namespace {

DataType mapONNXTypeToDataType(int onnx_type) {
    switch (onnx_type) {
        case 1:  return DataType::FLOAT32;
        case 2:  return DataType::UINT8;
        case 6:  return DataType::INT32;
        case 7:  return DataType::INT64;
        case 10: return DataType::FLOAT16;
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

// CPU version: data is already on host, just return pointer
template <typename T>
const T* getHostData(const std::shared_ptr<TensorBase>& tensor,
                     std::vector<uint8_t>& host_cache) {
    // For CPU executor, tensors are already on CPU, so just return the pointer
    return tensor->data_ptr<T>();
}

template <typename SrcT>
void dispatchCastToTarget(const SrcT* src,
                          DataType target_dtype,
                          const std::shared_ptr<TensorBase>& output,
                          size_t count) {
    switch (target_dtype) {
        case DataType::FLOAT32:
            castArray(src, output->data_ptr<float>(), count);
            break;
        case DataType::INT32:
            castArray(src, output->data_ptr<int32_t>(), count);
            break;
        case DataType::INT64:
            castArray(src, output->data_ptr<int64_t>(), count);
            break;
        case DataType::UINT8:
            castArray(src, output->data_ptr<uint8_t>(), count);
            break;
        case DataType::FLOAT16:
            throw std::runtime_error("Cast: FLOAT16 output not supported in executor");
        default:
            throw std::runtime_error("Cast: Unsupported target dtype " +
                                     dataTypeToString(target_dtype));
    }
}

template <typename T>
T readScalarValue(const std::shared_ptr<TensorBase>& tensor) {
    if (tensor->size() != 1) {
        throw std::runtime_error("Expected scalar tensor for Range inputs");
    }
    return tensor->data_ptr<T>()[0];
}

int64_t computeRangeElementCount(double start, double limit, double delta) {
    if (delta == 0.0) {
        throw std::runtime_error("Range: delta must be non-zero");
    }
    if ((delta > 0 && start >= limit) || (delta < 0 && start <= limit)) {
        return 0;
    }
    double steps = (limit - start) / delta;
    if (steps <= 0.0) {
        return 0;
    }
    int64_t count = static_cast<int64_t>(std::ceil(steps));
    return std::max<int64_t>(0, count);
}

template <typename T>
void fillRangeValues(T* dst, int64_t count, T start, T delta) {
    for (int64_t i = 0; i < count; ++i) {
        double value = static_cast<double>(start) +
                       static_cast<double>(i) * static_cast<double>(delta);
        dst[i] = static_cast<T>(value);
    }
}

template <typename T, typename Comparator>
void elementwiseCompare(const T* A, const T* B, uint8_t* out, size_t count, Comparator cmp) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<uint8_t>(cmp(A[i], B[i]));
    }
}

template <typename T>
void convertToDoubleBuffer(const T* src, size_t count, std::vector<double>& dst) {
    dst.resize(count);
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<double>(src[i]);
    }
}

template <typename T>
void fillTensorWithValue(T* data, size_t count, T value) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = value;
    }
}

template <typename T, typename UnaryFunc>
void applyUnary(const std::shared_ptr<TensorBase>& input,
                std::shared_ptr<TensorBase>& output,
                UnaryFunc func,
                bool move_to_gpu) {
    const T* src = input->data_ptr<T>();
    T* dst = output->data_ptr<T>();
    for (size_t i = 0; i < input->size(); ++i) {
        dst[i] = func(src[i]);
    }
}

double readScalarAsDouble(const std::shared_ptr<TensorBase>& tensor) {
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

int64_t readScalarAsInt(const std::shared_ptr<TensorBase>& tensor) {
    return static_cast<int64_t>(readScalarAsDouble(tensor));
}

DataType promoteDataType(DataType a, DataType b) {
    if (a == b) return a;
    if (a == DataType::FLOAT32 || b == DataType::FLOAT32) return DataType::FLOAT32;
    if (a == DataType::INT64 || b == DataType::INT64) return DataType::INT64;
    if (a == DataType::INT32 || b == DataType::INT32) return DataType::INT32;
    return DataType::UINT8;
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

std::vector<int64_t> tensorToShapeVector(const std::shared_ptr<TensorBase>& tensor) {
    std::vector<int64_t> result;
    size_t count = tensor->size();

    switch (tensor->dtype()) {
        case DataType::INT64: {
            const int64_t* data = tensor->data_ptr<int64_t>();
            result.assign(data, data + count);
            break;
        }
        case DataType::INT32: {
            const int32_t* data = tensor->data_ptr<int32_t>();
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
            const float* data = tensor->data_ptr<float>();
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

template <typename TargetT>
const TargetT* getDataAs(const std::shared_ptr<TensorBase>& tensor,
                         std::vector<uint8_t>& raw_cache,
                         std::vector<TargetT>& convert_buffer) {
    switch (tensor->dtype()) {
        case DataType::FLOAT32: {
            const float* data = tensor->data_ptr<float>();
            if constexpr (std::is_same_v<TargetT, float>) {
                return reinterpret_cast<const TargetT*>(data);
            }
            convert_buffer.resize(tensor->size());
            castArray(data, convert_buffer.data(), tensor->size());
            return convert_buffer.data();
        }
        case DataType::INT32: {
            const int32_t* data = tensor->data_ptr<int32_t>();
            if constexpr (std::is_same_v<TargetT, int32_t>) {
                return reinterpret_cast<const TargetT*>(data);
            }
            convert_buffer.resize(tensor->size());
            castArray(data, convert_buffer.data(), tensor->size());
            return convert_buffer.data();
        }
        case DataType::INT64: {
            const int64_t* data = tensor->data_ptr<int64_t>();
            if constexpr (std::is_same_v<TargetT, int64_t>) {
                return reinterpret_cast<const TargetT*>(data);
            }
            convert_buffer.resize(tensor->size());
            castArray(data, convert_buffer.data(), tensor->size());
            return convert_buffer.data();
        }
        case DataType::UINT8: {
            const uint8_t* data = tensor->data_ptr<uint8_t>();
            if constexpr (std::is_same_v<TargetT, uint8_t>) {
                return reinterpret_cast<const TargetT*>(data);
            }
            convert_buffer.resize(tensor->size());
            castArray(data, convert_buffer.data(), tensor->size());
            return convert_buffer.data();
        }
        default:
            throw std::runtime_error("Unsupported data type for conversion: " +
                                     dataTypeToString(tensor->dtype()));
    }
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
            remainder = output_strides[dim] == 0 ? 0 : remainder % output_strides[dim];

            if (dim >= offset && (dim - offset) < input_rank) {
                int64_t in_dim = input_shape[dim - offset];
                int64_t effective_coord = (in_dim == 1) ? 0 : coord;
                input_index += effective_coord * input_strides[dim - offset];
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

std::vector<uint8_t> tensorToBoolVector(const std::shared_ptr<TensorBase>& tensor) {
    std::vector<uint8_t> result(tensor->size());

    switch (tensor->dtype()) {
        case DataType::UINT8: {
            const uint8_t* data = tensor->data_ptr<uint8_t>();
            for (size_t i = 0; i < tensor->size(); ++i) {
                result[i] = data[i] ? 1 : 0;
            }
            break;
        }
        case DataType::INT32: {
            const int32_t* data = tensor->data_ptr<int32_t>();
            for (size_t i = 0; i < tensor->size(); ++i) {
                result[i] = data[i] != 0 ? 1 : 0;
            }
            break;
        }
        case DataType::INT64: {
            const int64_t* data = tensor->data_ptr<int64_t>();
            for (size_t i = 0; i < tensor->size(); ++i) {
                result[i] = data[i] != 0 ? 1 : 0;
            }
            break;
        }
        default: {
            const float* data = tensor->data_ptr<float>();
            for (size_t i = 0; i < tensor->size(); ++i) {
                result[i] = data[i] != 0.0f ? 1 : 0;
            }
            break;
        }
    }
    return result;
}

} // namespace

std::map<std::string, std::shared_ptr<Tensor>>
CpuExecutor::execute(const Graph& graph,
                     const std::map<std::string, std::shared_ptr<Tensor>>& inputs) {

    LOG_INFO("=== Starting Graph Execution on CPU ===");

    tensors_.clear();

    // Initialize inputs
    for (const auto& [name, tensor] : inputs) {
        LOG_INFO("Input: ", name, " ", tensor->shapeStr());
        tensors_[name] = tensor;
    }

    // Initialize constants / initializers
    for (const auto& [name, tensor] : graph.initializers()) {
        LOG_DEBUG("Initializer: ", name, " ", tensor->shapeStr());
        tensors_[name] = tensor;
    }

    // Execute nodes in topological order
    auto nodes = graph.topologicalSort();
    LOG_INFO("Executing ", nodes.size(), " nodes...");

    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        LOG_INFO("[", i, "/", nodes.size(), "] Executing: ",
                 opTypeToString(node->opType()), " (", node->name(), ")");

        try {
            executeNode(*node);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to execute node ", node->name(), ": ", e.what());
            throw;
        }
    }

    // Collect outputs
    std::map<std::string, std::shared_ptr<Tensor>> outputs;
    for (const auto& out_name : graph.outputs()) {
        auto it = tensors_.find(out_name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Output tensor not found: " + out_name);
        }
        outputs[out_name] = it->second;
        LOG_INFO("Output: ", out_name, " ", it->second->shapeStr());
    }

    LOG_INFO("=== CPU Execution Complete ===");
    return outputs;
}

std::shared_ptr<Tensor> CpuExecutor::allocateOutput(
    const std::vector<int64_t>& shape,
    DataType dtype) {
    return std::make_shared<CpuTensor>(shape, dtype);
}

void CpuExecutor::executeNode(const Node& node) {
    switch (node.opType()) {
        case OpType::MATMUL:
            executeMatMul(node);
            break;
        case OpType::RELU:
            executeReLU(node);
            break;
        case OpType::ADD:
            executeAdd(node);
            break;
        case OpType::SUB:
            executeSub(node);
            break;
        case OpType::GEMM:
            executeGemm(node);
            break;
        case OpType::GATHER:
            executeGather(node);
            break;
        case OpType::MUL:
            executeMul(node);
            break;
        case OpType::DIV:
            executeDiv(node);
            break;
        case OpType::POW:
            executePow(node);
            break;
        case OpType::SQRT:
            executeSqrt(node);
            break;
        case OpType::REDUCEMEAN:
            executeReduceMean(node);
            break;
        case OpType::RESHAPE:
            executeReshape(node);
            break;
        case OpType::TRANSPOSE:
            executeTranspose(node);
            break;
        case OpType::UNSQUEEZE:
            executeUnsqueeze(node);
            break;
        case OpType::SLICE:
            executeSlice(node);
            break;
        case OpType::CONCAT:
            executeConcat(node);
            break;
        case OpType::SHAPE:
            executeShape(node);
            break;
        case OpType::CAST:
            executeCast(node);
            break;
        case OpType::RANGE:
            executeRange(node);
            break;
        case OpType::EQUAL:
            executeEqual(node);
            break;
        case OpType::CONSTANTOFSHAPE:
            executeConstantOfShape(node);
            break;
        case OpType::EXPAND:
            executeExpand(node);
            break;
        case OpType::GREATER:
            executeGreater(node);
            break;
        case OpType::NEG:
            executeNeg(node);
            break;
        case OpType::SIGMOID:
            executeSigmoid(node);
            break;
        case OpType::SIN:
            executeSin(node);
            break;
        case OpType::COS:
            executeCos(node);
            break;
        case OpType::SOFTMAX:
            executeSoftmax(node);
            break;
        case OpType::SCATTERND:
            executeScatterND(node);
            break;
        case OpType::TRILU:
            executeTrilu(node);
            break;
        case OpType::WHERE:
            executeWhere(node);
            break;
        case OpType::REDUCESUM:
            executeReduceSum(node);
            break;
        case OpType::SIMPLIFIEDLAYERNORM:
            executeSimplifiedLayerNormalization(node);
            break;
        case OpType::SKIPSIMPLIFIEDLAYERNORM:
            executeSkipSimplifiedLayerNormalization(node);
            break;
        case OpType::ROTARYEMBEDDING:
            executeRotaryEmbedding(node);
            break;
        case OpType::GROUPQUERYATTENTION:
            executeGroupQueryAttention(node);
            break;
        default:
            throw std::runtime_error("Unsupported operation: " +
                                   opTypeToString(node.opType()));
    }
}

// Include CPU-only operation implementations
#include "cpu/ops/executeAdd.inl"
#include "cpu/ops/executeSub.inl"
#include "cpu/ops/executeMul.inl"
#include "cpu/ops/executeDiv.inl"
#include "cpu/ops/executeReLU.inl"
#include "cpu/ops/executeMatMul.inl"
#include "cpu/ops/executeGemm.inl"
#include "cpu/ops/executeGather.inl"
#include "cpu/ops/executePow.inl"
#include "cpu/ops/executeReduceMean.inl"
#include "cpu/ops/executeReshape.inl"
#include "cpu/ops/executeTranspose.inl"
#include "cpu/ops/executeUnsqueeze.inl"
#include "cpu/ops/executeSlice.inl"
#include "cpu/ops/executeConcat.inl"
#include "cpu/ops/executeShape.inl"
#include "cpu/ops/executeCast.inl"
#include "cpu/ops/executeRange.inl"
#include "cpu/ops/executeEqual.inl"
#include "cpu/ops/executeConstantOfShape.inl"
#include "cpu/ops/executeExpand.inl"
#include "cpu/ops/executeGreater.inl"
#include "cpu/ops/executeNeg.inl"
#include "cpu/ops/executeSigmoid.inl"
#include "cpu/ops/executeSin.inl"
#include "cpu/ops/executeCos.inl"
#include "cpu/ops/executeSoftmax.inl"
#include "cpu/ops/executeScatterND.inl"
#include "cpu/ops/executeTrilu.inl"
#include "cpu/ops/executeWhere.inl"
#include "cpu/ops/executeReduceSum.inl"
#include "cpu/ops/executeSimplifiedLayerNormalization.inl"
#include "cpu/ops/executeSkipSimplifiedLayerNormalization.inl"
#include "cpu/ops/executeSqrt.inl"
#include "cpu/ops/executeRotaryEmbedding.inl"
#include "cpu/ops/executeGroupQueryAttention.inl"

} // namespace onnx_runner
