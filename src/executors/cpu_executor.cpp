#include "executor.hpp"
#include "cpu_executor.hpp"
#include "utils/tensor/cpu_tensor.hpp"
#include "core/graph.hpp"
#include "utils/logger.hpp"
#include "../core/node.hpp"
#include <stdexcept>
#include <map>
#include <memory>
#include <vector>

namespace onnx_runner {

std::map<std::string, std::shared_ptr<TensorBase>>
CpuExecutor::execute(const Graph& graph,
                     const std::map<std::string, std::shared_ptr<TensorBase>>& inputs) {

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
    std::map<std::string, std::shared_ptr<TensorBase>> outputs;
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

std::shared_ptr<TensorBase> CpuExecutor::allocateOutput(
    const std::vector<int64_t>& shape,
    DataType dtype) {
    return std::make_shared<CpuTensor>(shape, dtype);
}

void CpuExecutor::executeNode(const Node& node) {
    // Gather input tensors
    std::vector<std::shared_ptr<TensorBase>> inputs;
    for (const auto& name : node.inputs()) {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("Input tensor not found: " + name);
        inputs.push_back(it->second);
    }

    // Dispatch based on op type (matches GPU executor)
    switch (node.opType()) {
        case OpType::ADD:
            LOG_INFO("Stub: Add operation");
            break;
        case OpType::MATMUL:
            LOG_INFO("Stub: MatMul operation");
            break;
        case OpType::RELU:
            LOG_INFO("Stub: ReLU operation");
            break;
        default:
            LOG_WARN("Unsupported op type, creating placeholder output");
            break;
    }

    // Allocate outputs
    for (const auto& out_name : node.outputs()) {
        if (!inputs.empty()) {
            tensors_[out_name] = allocateOutput(inputs[0]->shape(), inputs[0]->dtype());
        } else {
            tensors_[out_name] = allocateOutput({1}, DataType::FLOAT32);
        }
    }
}

} // namespace onnx_runner
