#include "cpu_executor.hpp"
#include "cpu/kernels/cpu_kernels.h"
#include "../utils/logger.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace onnx_runner {

std::map<std::string, std::shared_ptr<Tensor>>
CpuExecutor::execute(const Graph& graph,
                     const std::map<std::string, std::shared_ptr<Tensor>>& inputs) {
    LOG_INFO("=== Starting Graph Execution (CPU) ===");

    // Clear previous execution state
    tensors_.clear();

    // Initialize input tensors
    for (const auto& [name, tensor] : inputs) {
        LOG_INFO("Input: ", name, " ", tensor->shapeStr());
        tensors_[name] = tensor;
    }

    // Initialize constant tensors (initializers/weights)
    for (const auto& [name, tensor] : graph.initializers()) {
        LOG_DEBUG("Initializer: ", name, " ", tensor->shapeStr());
        tensors_[name] = tensor;
    }

    // Get nodes in execution order
    auto nodes = graph.topologicalSort();
    LOG_INFO("Executing ", nodes.size(), " nodes...");

    // Execute each node
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        LOG_INFO("[", i, "/", nodes.size(), "] Executing: ",
                 opTypeToString(node->opType()), " (", node->name(), ")");

        CPUTimer timer;
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

        outputs[output_name] = it->second;
        LOG_INFO("Output: ", output_name, " ", it->second->shapeStr());
    }

    LOG_INFO("=== Execution Complete ===");
    return outputs;
}

void CpuExecutor::executeNode(const Node& node) {
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

// Include operation implementations
#include "cpu/ops/executeMatMul.inl"
#include "cpu/ops/executeReLU.inl"
#include "cpu/ops/executeAdd.inl"
#include "cpu/ops/executeSub.inl"

// Placeholder implementations for operations not yet ported
void CpuExecutor::executeGemm(const Node& node) {
    throw std::runtime_error("Gemm not yet implemented in CPU executor");
}

void CpuExecutor::executeGather(const Node& node) {
    throw std::runtime_error("Gather not yet implemented in CPU executor");
}

void CpuExecutor::executeMul(const Node& node) {
    throw std::runtime_error("Mul not yet implemented in CPU executor");
}

void CpuExecutor::executeTranspose(const Node& node) {
    throw std::runtime_error("Transpose not yet implemented in CPU executor");
}

void CpuExecutor::executeShape(const Node& node) {
    throw std::runtime_error("Shape not yet implemented in CPU executor");
}

void CpuExecutor::executeCast(const Node& node) {
    throw std::runtime_error("Cast not yet implemented in CPU executor");
}

void CpuExecutor::executeSigmoid(const Node& node) {
    throw std::runtime_error("Sigmoid not yet implemented in CPU executor");
}

void CpuExecutor::executeReduceSum(const Node& node) {
    throw std::runtime_error("ReduceSum not yet implemented in CPU executor");
}

void CpuExecutor::executeSimplifiedLayerNormalization(const Node& node) {
    throw std::runtime_error("SimplifiedLayerNormalization not yet implemented in CPU executor");
}

void CpuExecutor::executeRotaryEmbedding(const Node& node) {
    throw std::runtime_error("RotaryEmbedding not yet implemented in CPU executor");
}

void CpuExecutor::executeGroupQueryAttention(const Node& node) {
    throw std::runtime_error("GroupQueryAttention not yet implemented in CPU executor");
}

void CpuExecutor::executeSkipSimplifiedLayerNormalization(const Node& node) {
    throw std::runtime_error("SkipSimplifiedLayerNormalization not yet implemented in CPU executor");
}

std::shared_ptr<Tensor> CpuExecutor::getTensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

bool CpuExecutor::hasTensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::shared_ptr<Tensor> CpuExecutor::allocateOutput(const std::vector<int64_t>& shape,
                                                    DataType dtype) {
    // CPU executor always creates CPU tensors
    return std::make_shared<Tensor>(shape, dtype);
}

void CpuExecutor::transposeMatrix(const float* input, float* output, int rows, int cols) {
    // Simple CPU-based matrix transpose
    // input is rows x cols, output will be cols x rows
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

} // namespace onnx_runner
