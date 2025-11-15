#pragma once

#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include "../utils/logger.hpp"
#include <map>
#include <string>
#include <memory>
#include <stdexcept>

namespace onnx_runner {

/**
 * Abstract base class for all executors.
 * Provides a polymorphic interface for executing ONNX computation graphs
 * on different backends (GPU, CPU single-threaded, CPU multi-threaded).
 * 
 * Includes common helper methods and tensor storage that all executors share.
 */
class Executor {
public:
    virtual ~Executor() = default;

    /**
     * Execute the computation graph with given inputs.
     * 
     * @param graph The computation graph to execute
     * @param inputs Map of input tensor names to input tensors
     * @return Map of output tensor names to output tensors
     */
    virtual std::map<std::string, std::shared_ptr<Tensor>>
    execute(const Graph& graph,
            const std::map<std::string, std::shared_ptr<Tensor>>& inputs) = 0;

    /**
     * Set whether to print detailed timing information during execution.
     * 
     * @param verbose True to enable verbose output, false otherwise
     */
    virtual void setVerbose(bool verbose) = 0;

    /**
     * Get the device type this executor uses.
     * 
     * @return DeviceType::CUDA for GPU executor, DeviceType::CPU for CPU executors
     */
    virtual DeviceType deviceType() const = 0;

    /**
     * Get a human-readable name for this executor.
     * 
     * @return String identifier (e.g., "GPU", "CPU", "CPU-MT")
     */
    virtual std::string name() const = 0;

protected:
    /**
     * Tensor storage during execution.
     * Maps tensor name to tensor data.
     * All executors use this to track intermediate and output tensors.
     */
    std::map<std::string, std::shared_ptr<Tensor>> tensors_;

    /**
     * Verbose flag for detailed logging.
     */
    bool verbose_ = false;

    /**
     * Get a tensor by name from the execution context.
     * 
     * @param name The name of the tensor to retrieve
     * @return Shared pointer to the tensor
     * @throws std::runtime_error if tensor is not found
     */
    std::shared_ptr<Tensor> getTensor(const std::string& name) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }

    /**
     * Check if a tensor exists in the execution context.
     * 
     * @param name The name of the tensor to check
     * @return True if tensor exists, false otherwise
     */
    bool hasTensor(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    /**
     * Allocate an output tensor with the given shape and dtype.
     * The tensor will be allocated on the executor's device.
     * Subclasses should override this if they need device-specific allocation.
     * 
     * @param shape The shape of the output tensor
     * @param dtype The data type of the output tensor (default: FLOAT32)
     * @return Shared pointer to the allocated tensor
     */
    virtual std::shared_ptr<Tensor> allocateOutput(
        const std::vector<int64_t>& shape,
        DataType dtype = DataType::FLOAT32) = 0;

    /**
     * Execute a single node in the graph.
     * Each executor implementation must provide this to handle
     * operation dispatch based on available kernels.
     * 
     * @param node The node to execute
     */
    virtual void executeNode(const Node& node) = 0;

    /**
     * Initialize input tensors into the execution context.
     * This is a common pattern that can be shared across executors.
     * Subclasses can override if they need custom input initialization.
     * 
     * @param inputs Map of input tensor names to input tensors
     */
    virtual void initializeInputs(
        const std::map<std::string, std::shared_ptr<Tensor>>& inputs) {
        for (const auto& [name, tensor] : inputs) {
            LOG_INFO("Input: ", name, " ", tensor->shapeStr());
            tensors_[name] = tensor;
        }
    }

    /**
     * Initialize constant tensors (initializers/weights) into the execution context.
     * This is a common pattern that can be shared across executors.
     * Subclasses can override if they need custom initializer handling.
     * 
     * @param graph The computation graph containing initializers
     */
    virtual void initializeInitializers(const Graph& graph) {
        for (const auto& [name, tensor] : graph.initializers()) {
            LOG_DEBUG("Initializer: ", name, " ", tensor->shapeStr());
            tensors_[name] = tensor;
        }
    }

    /**
     * Collect output tensors from the execution context.
     * This is a common pattern that can be shared across executors.
     * Subclasses can override if they need custom output collection.
     * 
     * @param graph The computation graph containing output names
     * @return Map of output tensor names to output tensors
     */
    virtual std::map<std::string, std::shared_ptr<Tensor>>
    collectOutputs(const Graph& graph) {
        std::map<std::string, std::shared_ptr<Tensor>> outputs;
        for (const auto& output_name : graph.outputs()) {
            auto it = tensors_.find(output_name);
            if (it == tensors_.end()) {
                throw std::runtime_error("Output tensor not found: " + output_name);
            }
            outputs[output_name] = it->second;
            LOG_INFO("Output: ", output_name, " ", it->second->shapeStr());
        }
        return outputs;
    }
};

} // namespace onnx_runner

