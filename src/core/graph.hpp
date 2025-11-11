#pragma once

#include "node.hpp"
#include "../utils/tensor.hpp"
#include <vector>
#include <map>
#include <memory>
#include <string>

namespace onnx_runner {

// Graph represents the entire computation graph
class Graph {
public:
    Graph() = default;

    // Add a node to the graph
    void addNode(std::shared_ptr<Node> node) {
        nodes_.push_back(node);
    }

    // Add an initializer (constant tensor, e.g., weights)
    void addInitializer(const std::string& name, std::shared_ptr<Tensor> tensor) {
        initializers_[name] = tensor;
    }

    // Set graph inputs
    void addInput(const std::string& name) {
        inputs_.push_back(name);
    }

    // Set graph input with shape
    void addInput(const std::string& name, const std::vector<int64_t>& shape) {
        inputs_.push_back(name);
        input_shapes_[name] = shape;
    }

    // Set graph outputs
    void addOutput(const std::string& name) {
        outputs_.push_back(name);
    }

    // Accessors
    const std::vector<std::shared_ptr<Node>>& nodes() const { return nodes_; }
    const std::map<std::string, std::shared_ptr<Tensor>>& initializers() const { return initializers_; }
    const std::vector<std::string>& inputs() const { return inputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }

    // Get input shape
    std::vector<int64_t> getInputShape(const std::string& name) const {
        auto it = input_shapes_.find(name);
        if (it != input_shapes_.end()) {
            return it->second;
        }
        return {};  // Return empty if not found
    }

    // Get initializer by name
    std::shared_ptr<Tensor> getInitializer(const std::string& name) const {
        auto it = initializers_.find(name);
        if (it != initializers_.end()) {
            return it->second;
        }
        return nullptr;
    }

    // Check if a name is an initializer
    bool isInitializer(const std::string& name) const {
        return initializers_.find(name) != initializers_.end();
    }

    // Get summary
    void printSummary() const;

    // Topological sort (for execution ordering)
    // Returns nodes in execution order
    std::vector<std::shared_ptr<Node>> topologicalSort() const;

private:
    std::vector<std::shared_ptr<Node>> nodes_;
    std::map<std::string, std::shared_ptr<Tensor>> initializers_;  // Constant tensors (weights)
    std::vector<std::string> inputs_;   // Graph input names
    std::vector<std::string> outputs_;  // Graph output names
    std::map<std::string, std::vector<int64_t>> input_shapes_;  // Input tensor shapes
};

} // namespace onnx_runner
