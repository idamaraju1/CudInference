#include "gpu_executor.hpp"
#include "kernels/kernels.cuh"
#include "../utils/logger.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <limits>
#include <type_traits>

namespace onnx_runner {

namespace {

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

std::map<std::string, std::shared_ptr<Tensor>>
GpuExecutor::execute(const Graph& graph,
                     const std::map<std::string, std::shared_ptr<Tensor>>& inputs) {
    LOG_INFO("=== Starting Graph Execution ===");

    // Clear previous execution state
    tensors_.clear();

    // Initialize input tensors
    for (const auto& [name, tensor] : inputs) {
        LOG_INFO("Input: ", name, " ", tensor->shapeStr());
        tensors_[name] = tensor;

        // Transfer to GPU if not using CPU fallback
        if (!use_cpu_fallback_ && tensor->device() == DeviceType::CPU) {
            tensors_[name]->toGPU();
            LOG_DEBUG("  Transferred to GPU");
        }
    }

    // Initialize constant tensors (initializers/weights)
    for (const auto& [name, tensor] : graph.initializers()) {
        LOG_DEBUG("Initializer: ", name, " ", tensor->shapeStr());

        // DEBUG: Print first few values (only for CPU FLOAT32 tensors to avoid segfault)
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

        // Transfer to GPU if not using CPU fallback
        if (!use_cpu_fallback_ && tensor->device() == DeviceType::CPU) {
            tensors_[name]->toGPU();
        }
    }

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

    // Collect output tensors
    std::map<std::string, std::shared_ptr<Tensor>> outputs;
    for (const auto& output_name : graph.outputs()) {
        auto it = tensors_.find(output_name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Output tensor not found: " + output_name);
        }

        // Transfer back to CPU only if NOT in GPU_PERSISTENT mode
        // In GPU_PERSISTENT mode, outputs stay on GPU for next iteration
        if (exec_mode_ != ExecutionMode::GPU_PERSISTENT && it->second->device() == DeviceType::CUDA) {
            it->second->toCPU();
        }

        outputs[output_name] = it->second;
        LOG_INFO("Output: ", output_name, " ", it->second->shapeStr());
    }

    LOG_INFO("=== Execution Complete ===");
    return outputs;
}

void GpuExecutor::executeNode(const Node& node) {
    switch (node.opType()) {
        case OpType::MATMUL:
            executeMatMul(node);
            break;
        case OpType::GEMM:
            executeGemm(node);
            break;
        case OpType::ADD:
            executeAdd(node);
            break;
        case OpType::SUB:
            executeSub(node);
            break;
        case OpType::MUL:
            executeMul(node);
            break;
        case OpType::RELU:
            executeReLU(node);
            break;
        case OpType::GATHER:
            executeGather(node);
            break;
        case OpType::TRANSPOSE:
            executeTranspose(node);
            break;
        case OpType::SHAPE:
            executeShape(node);
            break;
        case OpType::CAST:
            executeCast(node);
            break;
        case OpType::SIGMOID:
            executeSigmoid(node);
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

#include "ops/executeMatMul.inl"
#include "ops/executeReLU.inl"
#include "ops/executeAdd.inl"
#include "ops/executeSub.inl"
#include "ops/executeGemm.inl"
#include "ops/executeGather.inl"
#include "ops/executeMul.inl"
#include "ops/executeTranspose.inl"
#include "ops/executeShape.inl"
#include "ops/executeCast.inl"
#include "ops/executeSigmoid.inl"
#include "ops/executeReduceSum.inl"
#include "ops/executeSimplifiedLayerNormalization.inl"
#include "ops/executeSkipSimplifiedLayerNormalization.inl"
#include "ops/executeRotaryEmbedding.inl"
#include "ops/executeGroupQueryAttention.inl"

std::shared_ptr<Tensor> GpuExecutor::getTensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

bool GpuExecutor::hasTensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::shared_ptr<Tensor> GpuExecutor::allocateOutput(const std::vector<int64_t>& shape,
                                                    DataType dtype) {
    // Select allocation strategy based on execution mode
    if (use_cpu_fallback_ || exec_mode_ == ExecutionMode::CPU_ONLY) {
        // CPU tensor
        return std::make_shared<Tensor>(shape, dtype);
    } else if (exec_mode_ == ExecutionMode::GPU_PERSISTENT) {
        // NEW: Create directly on GPU (no CPU allocation)
        return Tensor::createOnGPU(shape, dtype);
    } else {
        // GPU_COPY mode (legacy): create on CPU then allocate GPU
        auto tensor = std::make_shared<Tensor>(shape, dtype);
        tensor->allocateGPU();
        return tensor;
    }
}

void GpuExecutor::transposeMatrix(const float* input, float* output, int rows, int cols, bool use_cpu) {
    // Simple CPU-based matrix transpose
    // input is rows x cols, output will be cols x rows
    // This works for both CPU and GPU modes (we transpose on CPU then copy to GPU if needed)

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// ============================================================================
// Persistent KV Cache Management (Phase 3.1)
// ============================================================================

void GpuExecutor::initializeKVCache(
    const std::string& cache_key,
    int batch,
    int kv_heads,
    int max_seq_length,
    int head_dim
) {
    if (exec_mode_ != ExecutionMode::GPU_PERSISTENT) {
        throw std::runtime_error("KV cache requires GPU_PERSISTENT mode");
    }

    // Allocate large cache on GPU
    std::vector<int64_t> shape = {batch, kv_heads, max_seq_length, head_dim};

    auto key_cache = Tensor::createOnGPU(shape, DataType::FLOAT32);
    auto value_cache = Tensor::createOnGPU(shape, DataType::FLOAT32);

    // Zero initialize (already done in createOnGPU via cudaMemset)

    kv_cache_[cache_key] = {key_cache, value_cache, 0, max_seq_length};

    if (verbose_) {
        LOG_INFO("Initialized KV cache '", cache_key, "' on GPU: batch=", batch,
                 ", kv_heads=", kv_heads, ", max_seq=", max_seq_length,
                 ", head_dim=", head_dim);
    }
}

void GpuExecutor::appendKVCache(
    const std::string& cache_key,
    const std::shared_ptr<Tensor>& new_keys,
    const std::shared_ptr<Tensor>& new_values
) {
    if (exec_mode_ != ExecutionMode::GPU_PERSISTENT) {
        throw std::runtime_error("KV cache requires GPU_PERSISTENT mode");
    }

    if (kv_cache_.find(cache_key) == kv_cache_.end()) {
        throw std::runtime_error("KV cache '" + cache_key + "' not initialized");
    }

    auto& entry = kv_cache_[cache_key];

    if (!new_keys->isOnGPU() || !new_values->isOnGPU()) {
        throw std::runtime_error("appendKVCache requires GPU tensors");
    }

    // new_keys/new_values: [batch, kv_heads, new_seq, head_dim]
    int batch = static_cast<int>(entry.key_cache->dim(0));
    int kv_heads = static_cast<int>(entry.key_cache->dim(1));
    int head_dim = static_cast<int>(entry.key_cache->dim(3));
    int new_seq = static_cast<int>(new_keys->dim(2));

    if (entry.current_length + new_seq > entry.max_length) {
        throw std::runtime_error("KV cache overflow: current=" + std::to_string(entry.current_length) +
                                 " + new=" + std::to_string(new_seq) +
                                 " > max=" + std::to_string(entry.max_length));
    }

    // Copy new K/V into cache at position current_length (GPU-to-GPU)
    // Use 2D memcpy for each batch/head combination
    size_t cache_seq_stride = entry.max_length * head_dim;
    size_t new_seq_stride = new_seq * head_dim;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kv_heads; ++h) {
            // Destination offset in cache
            float* dst_key = entry.key_cache->mutableDeviceData<float>() +
                ((b * kv_heads + h) * cache_seq_stride) +
                (entry.current_length * head_dim);

            float* dst_val = entry.value_cache->mutableDeviceData<float>() +
                ((b * kv_heads + h) * cache_seq_stride) +
                (entry.current_length * head_dim);

            // Source offset in new tensors
            const float* src_key = new_keys->deviceData<float>() +
                ((b * kv_heads + h) * new_seq_stride);

            const float* src_val = new_values->deviceData<float>() +
                ((b * kv_heads + h) * new_seq_stride);

            // Device-to-device copy (fast!)
            CUDA_CHECK(cudaMemcpy(dst_key, src_key, new_seq * head_dim * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(dst_val, src_val, new_seq * head_dim * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
    }

    entry.current_length += new_seq;

    if (verbose_) {
        LOG_DEBUG("Appended ", new_seq, " tokens to KV cache '", cache_key,
                  "', new length: ", entry.current_length);
    }
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
GpuExecutor::getKVCache(const std::string& cache_key) {
    if (kv_cache_.find(cache_key) == kv_cache_.end()) {
        throw std::runtime_error("KV cache '" + cache_key + "' not found");
    }

    auto& entry = kv_cache_[cache_key];

    // Return the full cache tensors (caller will use current_length to know valid range)
    // Note: In a more sophisticated implementation, we'd return sliced views
    // For now, the attention kernel will use a "valid_length" parameter

    return {entry.key_cache, entry.value_cache};
}

} // namespace onnx_runner
