#pragma once

#include "../core/node.hpp"
#include "../utils/tensor.hpp"
#include "../utils/logger.hpp"
#include <map>
#include <string>
#include <memory>
#include <stdexcept>

namespace onnx_runner {

// Forward declarations
class Executor;

// Execution context provides operations access to executor state
struct ExecutionContext {
    // Tensor storage (shared with executor)
    std::map<std::string, std::shared_ptr<Tensor>>& tensors;

    // Execution parameters
    bool use_cpu;                    // True if executing on CPU
    int num_cpu_threads;             // Number of CPU threads for parallel ops
    bool verbose;                    // Enable verbose logging

    // GPU-specific parameters
    enum class GPUMode {
        NONE,           // CPU only
        COPY,           // Copy data per operation
        PERSISTENT      // Keep tensors on GPU
    };
    GPUMode gpu_mode = GPUMode::NONE;

    ExecutionContext(
        std::map<std::string, std::shared_ptr<Tensor>>& tensor_storage,
        bool cpu,
        int threads = 1,
        bool verb = false,
        GPUMode mode = GPUMode::NONE
    ) : tensors(tensor_storage),
        use_cpu(cpu),
        num_cpu_threads(threads),
        verbose(verb),
        gpu_mode(mode) {}
};

// Base class for all operations
class Operation {
public:
    virtual ~Operation() = default;

    // Execute the operation
    virtual void execute(const Node& node, ExecutionContext& ctx) = 0;

    // Get operation name for logging
    virtual const char* name() const = 0;

protected:
    // Common helper methods for derived classes

    // Validate input/output counts
    void validateInputCount(const Node& node, size_t expected) const {
        if (node.inputs().size() != expected) {
            throw std::runtime_error(
                std::string(name()) + " expects " + std::to_string(expected) +
                " input(s), got " + std::to_string(node.inputs().size())
            );
        }
    }

    void validateOutputCount(const Node& node, size_t expected) const {
        if (node.outputs().size() != expected) {
            throw std::runtime_error(
                std::string(name()) + " expects " + std::to_string(expected) +
                " output(s), got " + std::to_string(node.outputs().size())
            );
        }
    }

    void validateInputOutputCount(const Node& node, size_t inputs, size_t outputs) const {
        validateInputCount(node, inputs);
        validateOutputCount(node, outputs);
    }

    // Get tensor from context
    std::shared_ptr<Tensor> getTensor(const std::string& name, ExecutionContext& ctx) const {
        auto it = ctx.tensors.find(name);
        if (it == ctx.tensors.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }

    // Check if tensor exists
    bool hasTensor(const std::string& name, ExecutionContext& ctx) const {
        return ctx.tensors.find(name) != ctx.tensors.end();
    }

    // Allocate output tensor
    std::shared_ptr<Tensor> allocateOutput(
        const std::vector<int64_t>& shape,
        ExecutionContext& ctx,
        DataType dtype = DataType::FLOAT32
    ) const {
        auto output = std::make_shared<Tensor>(shape, dtype);

        // In GPU_PERSISTENT mode, allocate directly on GPU
        if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
            output->allocateGPU();
        }

        return output;
    }

    // Store output tensor in context
    void storeOutput(const std::string& name, std::shared_ptr<Tensor> tensor, ExecutionContext& ctx) const {
        ctx.tensors[name] = tensor;
    }

    // Log debug message if verbose
    template<typename... Args>
    void logDebug(ExecutionContext& ctx, Args&&... args) const {
        if (ctx.verbose) {
            LOG_DEBUG(std::forward<Args>(args)...);
        }
    }
};

} // namespace onnx_runner
