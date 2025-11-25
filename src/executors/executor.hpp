#pragma once

#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include <map>
#include <string>
#include <memory>

namespace onnx_runner {

// Base executor interface for running ONNX graphs
class Executor {
public:
    virtual ~Executor() = default;

    // Execute the graph with given inputs
    // inputs: map of input names to input tensors
    // Returns: map of output names to output tensors
    virtual std::map<std::string, std::shared_ptr<Tensor>>
    execute(const Graph& graph,
            const std::map<std::string, std::shared_ptr<Tensor>>& inputs) = 0;

    // Set whether to print detailed timing information
    virtual void setVerbose(bool verbose) = 0;
};

} // namespace onnx_runner
