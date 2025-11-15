#include "gpu_executor.hpp"
#include "../gpu/kernels/kernels.cuh"
#include "../utils/logger.hpp"
#include "../utils/tensor/gpu_tensor.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <limits>
#include <type_traits>

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

template <typename SrcT>
const SrcT* getHostData(const std::shared_ptr<Tensor>& tensor,
                        std::vector<uint8_t>& host_cache) {
    if (tensor->device() == DeviceType::CPU) {
        return tensor->data<SrcT>();
    }

    size_t bytes = tensor->size() * sizeof(SrcT);
    host_cache.resize(bytes);
    CUDA_CHECK(cudaMemcpy(host_cache.data(), tensor->data<SrcT>(), bytes, cudaMemcpyDeviceToHost));
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

template <typename T>
T readScalarValue(const std::shared_ptr<Tensor>& tensor) {
    if (tensor->size() != 1) {
        throw std::runtime_error("Expected scalar tensor for Range inputs");
    }
    std::vector<uint8_t> cache;
    const T* data_ptr = getHostData<T>(tensor, cache);
    return data_ptr[0];
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
void applyUnary(const std::shared_ptr<Tensor>& input,
                std::shared_ptr<Tensor>& output,
                UnaryFunc func,
                bool move_to_gpu) {
    std::vector<uint8_t> cache;
    const T* src = getHostData<T>(input, cache);
    T* dst = output->data<T>();
    for (size_t i = 0; i < input->size(); ++i) {
        dst[i] = func(src[i]);
    }
    if (move_to_gpu) {
        output->toGPU();
    }
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
        default: { // e.g. FLOAT32
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

        // Dimensions must be equal, or one must be exactly 1 (not 0)
        if (dimA == dimB) {
            out_shape[i] = dimA;
        } else if (dimA == 1) {
            out_shape[i] = dimB;
        } else if (dimB == 1) {
            out_shape[i] = dimA;
        } else {
            // Incompatible dimensions (including 0 != N cases)
            return false;
        }
    }
    return true;
}

template <typename TargetT>
const TargetT* getDataAs(const std::shared_ptr<Tensor>& tensor,
                         std::vector<uint8_t>& raw_cache,
                         std::vector<TargetT>& convert_buffer) {
    switch (tensor->dtype()) {
        case DataType::FLOAT32: {
            const float* data = getHostData<float>(tensor, raw_cache);
            if constexpr (std::is_same_v<TargetT, float>) {
                return reinterpret_cast<const TargetT*>(data);
            }
            convert_buffer.resize(tensor->size());
            castArray(data, convert_buffer.data(), tensor->size());
            return convert_buffer.data();
        }
        case DataType::INT32: {
            const int32_t* data = getHostData<int32_t>(tensor, raw_cache);
            if constexpr (std::is_same_v<TargetT, int32_t>) {
                return reinterpret_cast<const TargetT*>(data);
            }
            convert_buffer.resize(tensor->size());
            castArray(data, convert_buffer.data(), tensor->size());
            return convert_buffer.data();
        }
        case DataType::INT64: {
            const int64_t* data = getHostData<int64_t>(tensor, raw_cache);
            if constexpr (std::is_same_v<TargetT, int64_t>) {
                return reinterpret_cast<const TargetT*>(data);
            }
            convert_buffer.resize(tensor->size());
            castArray(data, convert_buffer.data(), tensor->size());
            return convert_buffer.data();
        }
        case DataType::UINT8: {
            const uint8_t* data = getHostData<uint8_t>(tensor, raw_cache);
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

std::vector<uint8_t> tensorToBoolVector(const std::shared_ptr<Tensor>& tensor) {
    std::vector<uint8_t> result(tensor->size());

    switch (tensor->dtype()) {
        case DataType::UINT8: {
            std::vector<uint8_t> cache;
            const uint8_t* data = getHostData<uint8_t>(tensor, cache);
            for (size_t i = 0; i < tensor->size(); ++i) {
                result[i] = data[i] ? 1 : 0;
            }
            break;
        }
        case DataType::INT32: {
            std::vector<uint8_t> cache;
            const int32_t* data = getHostData<int32_t>(tensor, cache);
            for (size_t i = 0; i < tensor->size(); ++i) {
                result[i] = data[i] != 0 ? 1 : 0;
            }
            break;
        }
        case DataType::INT64: {
            std::vector<uint8_t> cache;
            const int64_t* data = getHostData<int64_t>(tensor, cache);
            for (size_t i = 0; i < tensor->size(); ++i) {
                result[i] = data[i] != 0 ? 1 : 0;
            }
            break;
        }
        default: {
            std::vector<uint8_t> cache;
            const float* data = getHostData<float>(tensor, cache);
            for (size_t i = 0; i < tensor->size(); ++i) {
                result[i] = data[i] != 0.0f ? 1 : 0;
            }
            break;
        }
    }

    return result;
}

} // namespace

// Implement Executor interface
std::map<std::string, std::shared_ptr<Tensor>>
GpuExecutor::execute(const Graph& graph,
                     const std::map<std::string, std::shared_ptr<Tensor>>& inputs) {
    LOG_INFO("=== Starting Graph Execution ===");

    // Clear previous execution state
    tensors_.clear();

    // Initialize inputs (with GPU transfer)
    initializeInputs(inputs);

    // Initialize initializers (with GPU transfer)
    initializeInitializers(graph);

    // Get nodes in execution order
    auto nodes = graph.topologicalSort();
    LOG_INFO("Executing ", nodes.size(), " nodes...");

    // Execute each node
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        LOG_INFO("[", i, "/", nodes.size(), "] Executing: ",
                 opTypeToString(node->opType()), " (", node->name(), ")");

        GPUTimer timer;
        timer.start();

        try {
            executeNode(*node);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to execute node ", node->name(), ": ", e.what());
            throw;
        }

        timer.stop();

        if (verbose_) {
            LOG_INFO("  Time: ", timer.elapsedMilliseconds(), " ms");
        }
    }

    // Collect outputs (transfer back to CPU)
    auto outputs = collectOutputs(graph);

    LOG_INFO("=== Execution Complete ===");
    return outputs;
}

// Override base class helpers for GPU-specific behavior
std::shared_ptr<Tensor> GpuExecutor::allocateOutput(
    const std::vector<int64_t>& shape,
    DataType dtype) {
    auto tensor = std::make_shared<GpuTensor>(shape, dtype);
    tensor->allocateGPU();  // Always allocate on GPU
    return tensor;
}

void GpuExecutor::initializeInputs(
    const std::map<std::string, std::shared_ptr<Tensor>>& inputs) {
    for (const auto& [name, tensor] : inputs) {
        LOG_INFO("Input: ", name, " ", tensor->shapeStr());
        tensors_[name] = tensor;

        // Transfer to GPU if on CPU
        if (tensor->device() == DeviceType::CPU) {
            tensors_[name]->toGPU();
            LOG_DEBUG("  Transferred to GPU");
        }
    }
}

void GpuExecutor::initializeInitializers(const Graph& graph) {
    for (const auto& [name, tensor] : graph.initializers()) {
        LOG_DEBUG("Initializer: ", name, " ", tensor->shapeStr());

        // DEBUG: Print first few values (only for CPU FLOAT32 tensors)
        if (tensor->device() == DeviceType::CPU && tensor->dtype() == DataType::FLOAT32 && tensor->size() > 0) {
            const float* data_ptr = tensor->data<float>();
            std::string values_str = "[";
            for (size_t i = 0; i < std::min<size_t>(5, tensor->size()); ++i) {
                values_str += std::to_string(data_ptr[i]);
                if (i < std::min<size_t>(5, tensor->size()) - 1) values_str += ", ";
            }
            values_str += "]";
            LOG_DEBUG("  First values: ", values_str);
        }

        tensors_[name] = tensor;

        // Transfer to GPU if on CPU
        if (tensor->device() == DeviceType::CPU) {
            tensors_[name]->toGPU();
        }
    }
}

std::map<std::string, std::shared_ptr<Tensor>>
GpuExecutor::collectOutputs(const Graph& graph) {
    std::map<std::string, std::shared_ptr<Tensor>> outputs;
    for (const auto& output_name : graph.outputs()) {
        auto it = tensors_.find(output_name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Output tensor not found: " + output_name);
        }

        // Transfer back to CPU if on GPU
        if (it->second->device() == DeviceType::CUDA) {
            it->second->toCPU();
        }

        outputs[output_name] = it->second;
        LOG_INFO("Output: ", output_name, " ", it->second->shapeStr());
    }
    return outputs;
}

/**
 * Allocate a tensor on GPU for output.
 */
std::shared_ptr<Tensor> GpuExecutor::allocateOutput(
    const std::vector<int64_t>& shape,
    DataType dtype) {
    return std::make_shared<GpuTensor>(shape, dtype);
}

// Must implement pure virtual from base class
void GpuExecutor::executeNode(const Node& node) {
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

// Operation implementations (GPU-only, no CPU fallback)
// All operations are inlined here for pure GPU execution

void GpuExecutor::executeReLU(const Node& node) {
    // ReLU: Y = max(0, X)
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("ReLU expects 1 input and 1 output");
    }

    auto X = getTensor(node.inputs()[0]);
    auto Y = allocateOutput(X->shape());

    int size = X->size();
    LOG_DEBUG("  ReLU: size=", size);

    // Always use GPU kernel
    kernels::launchReLU(X->data<float>(), Y->data<float>(), size);
    CUDA_CHECK(cudaDeviceSynchronize());

    tensors_[node.outputs()[0]] = Y;
}

void GpuExecutor::executeAdd(const Node& node) {
    // Add: C = A + B (element-wise)
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Add expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Handle scalar broadcasting
    if (A->shape() != B->shape()) {
        if (B->size() == 1) {
            auto C = allocateOutput(A->shape());
            int size = A->size();

            // Read scalar value (tensor is already on GPU or CPU)
            std::vector<uint8_t> cache;
            const float* b_data = getHostData<float>(B, cache);
            float scalar = b_data[0];

            // Use GPU scalar add kernel
            kernels::launchAddScalar(A->data<float>(), scalar, C->data<float>(), size);
            CUDA_CHECK(cudaDeviceSynchronize());

            tensors_[node.outputs()[0]] = C;
            return;
        }

        throw std::runtime_error("Add shape mismatch: " + A->shapeStr() +
                               " vs " + B->shapeStr());
    }

    auto C = allocateOutput(A->shape());
    int size = A->size();
    LOG_DEBUG("  Add: size=", size);

    // Always use GPU kernel
    kernels::launchAdd(A->data<float>(), B->data<float>(), C->data<float>(), size);
    CUDA_CHECK(cudaDeviceSynchronize());

    tensors_[node.outputs()[0]] = C;
}

void GpuExecutor::executeSub(const Node& node) {
    // Sub: C = A - B (element-wise)
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Sub expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Handle scalar broadcasting
    if (A->shape() != B->shape()) {
        if (B->size() == 1) {
            auto C = allocateOutput(A->shape());
            int size = A->size();

            // Read scalar value
            std::vector<uint8_t> cache;
            const float* b_data = getHostData<float>(B, cache);
            float scalar = b_data[0];

            // Use GPU scalar sub kernel
            kernels::launchSubScalar(A->data<float>(), scalar, C->data<float>(), size);
            CUDA_CHECK(cudaDeviceSynchronize());

            tensors_[node.outputs()[0]] = C;
            return;
        }

        throw std::runtime_error("Sub shape mismatch: " + A->shapeStr() +
                               " vs " + B->shapeStr());
    }

    auto C = allocateOutput(A->shape());
    int size = A->size();
    LOG_DEBUG("  Sub: size=", size);

    // Always use GPU kernel
    kernels::launchSub(A->data<float>(), B->data<float>(), C->data<float>(), size);
    CUDA_CHECK(cudaDeviceSynchronize());

    tensors_[node.outputs()[0]] = C;
}

void GpuExecutor::executeMatMul(const Node& node) {
    // MatMul: Y = A @ B
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("MatMul expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Fast path: strictly 2D matrices
    if (A->ndim() == 2 && B->ndim() == 2) {
        int64_t M = A->dim(0);
        int64_t K = A->dim(1);
        int64_t K2 = B->dim(0);
        int64_t N = B->dim(1);

        if (K != K2) {
            throw std::runtime_error("MatMul dimension mismatch: A(" +
                                   std::to_string(M) + "," + std::to_string(K) + ") @ B(" +
                                   std::to_string(K2) + "," + std::to_string(N) + ")");
        }

        auto Y = allocateOutput({M, N});
        LOG_DEBUG("  MatMul: (", M, ", ", K, ") @ (", K, ", ", N, ") -> (", M, ", ", N, ")");

        // Always use GPU kernel
        kernels::launchMatMul(A->data<float>(), B->data<float>(), Y->data<float>(),
                             M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        tensors_[node.outputs()[0]] = Y;
        return;
    }

    // Higher-dimensional MatMul (batch processing)
    if (A->ndim() < 2 || B->ndim() < 2) {
        throw std::runtime_error("MatMul requires both inputs to have rank >= 2");
    }

    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();

    int64_t M = shapeA[shapeA.size() - 2];
    int64_t K = shapeA.back();
    int64_t K2 = shapeB[shapeB.size() - 2];
    int64_t N = shapeB.back();

    if (K != K2) {
        throw std::runtime_error("MatMul dimension mismatch: A K=" + std::to_string(K) +
                                 " vs B K=" + std::to_string(K2));
    }

    std::vector<int64_t> batchA(shapeA.begin(), shapeA.end() - 2);
    std::vector<int64_t> batchB(shapeB.begin(), shapeB.end() - 2);
    std::vector<int64_t> batch_shape;
    if (!computeBroadcastShape(batchA, batchB, batch_shape)) {
        throw std::runtime_error("MatMul: unable to broadcast batch dimensions");
    }

    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(M);
    output_shape.push_back(N);
    auto Y = allocateOutput(output_shape);

    // For batched MatMul, we need to process on GPU
    // This is a simplified version - full implementation would use batched cuBLAS
    // For now, flatten and process sequentially (can be optimized later)
    size_t batch_count = computeSizeFromShape(batch_shape);
    if (batch_count == 0) {
        tensors_[node.outputs()[0]] = Y;
        return;
    }

    // Note: Full batched MatMul implementation would be more complex
    // This is a placeholder that processes batches sequentially
    throw std::runtime_error("Batched MatMul not yet fully implemented for GPU-only executor");
}

void GpuExecutor::executeGemm(const Node& node) {
    // GEMM: Y = alpha * A @ B + beta * C
    // Simplified: Y = A @ B + C (assuming alpha=1, beta=1)
    if (node.inputs().size() < 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Gemm expects at least 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Check for transpose attributes
    bool transA = node.getIntAttr("transA", 0) != 0;
    bool transB = node.getIntAttr("transB", 0) != 0;
    float alpha = node.getFloatAttr("alpha", 1.0f);
    float beta = node.getFloatAttr("beta", 1.0f);

    LOG_DEBUG("  Gemm: A", A->shapeStr(), " B", B->shapeStr(),
             " transA=", transA, " transB=", transB);

    // Only support alpha=1, beta=1 for now
    if (alpha != 1.0f || beta != 1.0f) {
        throw std::runtime_error("Gemm only supports alpha=1.0 and beta=1.0");
    }

    // Determine dimensions based on transpose flags
    int64_t M = transA ? A->dim(1) : A->dim(0);
    int64_t K = transA ? A->dim(0) : A->dim(1);
    int64_t K_B = transB ? B->dim(1) : B->dim(0);
    int64_t N = transB ? B->dim(0) : B->dim(1);

    LOG_DEBUG("  Result dimensions: M=", M, " K=", K, " N=", N);

    if (K != K_B) {
        throw std::runtime_error("Gemm dimension mismatch: K dimensions don't match");
    }

    // Handle transpose by creating transposed copies if needed
    std::shared_ptr<TensorBase> A_op = A;
    std::shared_ptr<TensorBase> B_op = B;

    if (transA) {
        // Transpose on CPU then copy to GPU
        std::vector<uint8_t> cache;
        const float* a_data = getHostData<float>(A, cache);
        
        auto A_op_cpu = std::make_shared<CpuTensor>(std::vector<int64_t>{M, K}, A->dtype());
        transposeMatrix(a_data, A_op_cpu->data<float>(), A->dim(0), A->dim(1), true);
        A_op = A_op_cpu->toGPU();
    }

    if (transB) {
        // Transpose on CPU then copy to GPU
        std::vector<uint8_t> cache;
        const float* b_data = getHostData<float>(B, cache);
        
        auto B_op_cpu = std::make_shared<CpuTensor>(std::vector<int64_t>{K, N}, B->dtype());
        transposeMatrix(b_data, B_op_cpu->data<float>(), B->dim(0), B->dim(1), true);
        B_op = B_op_cpu->toGPU();
    }

    auto Y = allocateOutput({M, N});

    // Always use GPU kernel
    kernels::launchMatMul(A_op->data<float>(), B_op->data<float>(), Y->data<float>(),
                         M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Add bias if present
    if (node.inputs().size() >= 3) {
        auto C = getTensor(node.inputs()[2]);
        int size = Y->size();

        // Always use GPU kernel
        kernels::launchAdd(Y->data<float>(), C->data<float>(), Y->data<float>(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = Y;
}

// GPU-only operation implementations (continued)

void GpuExecutor::executeSqrt(const Node& node) {
    // Sqrt: Y = sqrt(A)
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Sqrt expects 1 input and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto output = allocateOutput(A->shape());

    // Always use GPU kernel (pass false for use_cpu, 0 for num_threads)
    kernels::launchSqrtKernel(A->data<float>(), output->data<float>(), A->size(), false, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    tensors_[node.outputs()[0]] = output;
}

// TODO: Convert remaining operations to GPU-only
// For now, these still include CPU fallback logic and need to be updated
// They reference use_cpu_fallback_ and num_cpu_threads_ which don't exist in pure GPU executor
//
// Temporary workaround: Define stub variables so included .inl files compile
// These will be removed as operations are converted to pure GPU versions
// Static file-scope variables accessible to included .inl files
static bool use_cpu_fallback_ = false;  // Always use GPU
static int num_cpu_threads_ = 0;         // No CPU threads needed

// Temporary: Include from old location (will be replaced with GPU-only versions)
#include "../gpu/ops/executeGather.inl"
#include "../gpu/ops/executeMul.inl"
#include "../gpu/ops/executeDiv.inl"
#include "../gpu/ops/executePow.inl"
#include "../gpu/ops/executeReduceMean.inl"
#include "../gpu/ops/executeReshape.inl"
#include "../gpu/ops/executeTranspose.inl"
#include "../gpu/ops/executeUnsqueeze.inl"
#include "../gpu/ops/executeSlice.inl"
#include "../gpu/ops/executeConcat.inl"
#include "../gpu/ops/executeShape.inl"
#include "../gpu/ops/executeCast.inl"
#include "../gpu/ops/executeRange.inl"
#include "../gpu/ops/executeEqual.inl"
#include "../gpu/ops/executeConstantOfShape.inl"
#include "../gpu/ops/executeExpand.inl"
#include "../gpu/ops/executeGreater.inl"
#include "../gpu/ops/executeNeg.inl"
#include "../gpu/ops/executeSigmoid.inl"
#include "../gpu/ops/executeSin.inl"
#include "../gpu/ops/executeCos.inl"
#include "../gpu/ops/executeSoftmax.inl"
#include "../gpu/ops/executeScatterND.inl"
#include "../gpu/ops/executeTrilu.inl"
#include "../gpu/ops/executeWhere.inl"
#include "../gpu/ops/executeReduceSum.inl"
#include "../gpu/ops/executeSimplifiedLayerNormalization.inl"
#include "../gpu/ops/executeSkipSimplifiedLayerNormalization.inl"
#include "../gpu/ops/executeRotaryEmbedding.inl"
#include "../gpu/ops/executeGroupQueryAttention.inl"

// Helper: transpose a matrix (CPU-based, used for transpose operations)
void GpuExecutor::transposeMatrix(const float* input, float* output, int rows, int cols, bool use_cpu) {
    // Simple CPU-based matrix transpose
    // Note: use_cpu parameter is kept for compatibility but always true in GPU executor
    // (transpose happens on CPU then data is copied to GPU)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

} // namespace onnx_runner

