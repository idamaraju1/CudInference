#include "cpu_executor.hpp"
#include "cpu/kernels/cpu_kernels.h"
#include "../utils/logger.hpp"
#include "../operations/operation_registry.hpp"
#include "../operations/add_operation.hpp"
#include "../operations/relu_operation.hpp"
#include "../operations/sub_operation.hpp"
#include "../operations/sigmoid_operation.hpp"
#include "../operations/mul_operation.hpp"
#include "../operations/shape_operation.hpp"
#include "../operations/transpose_operation.hpp"
#include "../operations/cast_operation.hpp"
#include "../operations/reducesum_operation.hpp"
#include "../operations/gemm_operation.hpp"
#include "../operations/gather_operation.hpp"
#include "../operations/matmul_operation.hpp"
#include "../operations/simplified_layernorm_operation.hpp"
#include "../operations/skip_simplified_layernorm_operation.hpp"
#include "../operations/rotary_embedding_operation.hpp"
#include "../operations/group_query_attention_operation.hpp"
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
    // Create execution context for CPU execution
    ExecutionContext ctx(
        tensors_,
        true,  // use_cpu = true
        num_cpu_threads_,
        verbose_,
        ExecutionContext::GPUMode::NONE
    );

    // Use operation registry for all operations
    auto& registry = OperationRegistry::getInstance();
    if (!registry.hasOperation(node.opType())) {
        throw std::runtime_error("Unsupported operation: " +
                               opTypeToString(node.opType()));
    }

    auto op = registry.getOperation(node.opType());
    op->execute(node, ctx);
}

// All operations now use the operation registry system

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
