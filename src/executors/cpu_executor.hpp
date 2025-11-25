#pragma once

#include "executor.hpp"
#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include "../utils/logger.hpp"
#include <map>
#include <string>
#include <memory>
#include <chrono>

namespace onnx_runner {

// Timer for benchmarking CPU operations
class CPUTimer {
public:
    CPUTimer() = default;

    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }

    float elapsedMilliseconds() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
        return duration.count() / 1000.0f;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point stop_;
};

// CpuExecutor manages tensor allocation and graph execution on CPU
class CpuExecutor : public Executor {
public:
    CpuExecutor(int num_threads = 1)
        : num_cpu_threads_(num_threads) {}

    // Execute the graph with given inputs
    // inputs: map of input names to input tensors
    // Returns: map of output names to output tensors
    std::map<std::string, std::shared_ptr<Tensor>>
    execute(const Graph& graph,
            const std::map<std::string, std::shared_ptr<Tensor>>& inputs) override;

    // Set whether to print detailed timing information
    void setVerbose(bool verbose) override { verbose_ = verbose; }

    // Set number of CPU threads
    void setNumThreads(int num_threads) { num_cpu_threads_ = num_threads; }
    int getNumThreads() const { return num_cpu_threads_; }

private:
    bool verbose_ = false;
    int num_cpu_threads_;

    // Tensor storage during execution
    // Maps tensor name to tensor data
    std::map<std::string, std::shared_ptr<Tensor>> tensors_;

    // Execute a single node
    void executeNode(const Node& node);

    // Operation implementations
    void executeMatMul(const Node& node);
    void executeReLU(const Node& node);
    void executeAdd(const Node& node);
    void executeSub(const Node& node);
    void executeGemm(const Node& node);
    void executeGather(const Node& node);
    void executeMul(const Node& node);
    void executeTranspose(const Node& node);
    void executeShape(const Node& node);
    void executeCast(const Node& node);
    void executeSigmoid(const Node& node);
    void executeReduceSum(const Node& node);
    void executeSimplifiedLayerNormalization(const Node& node);
    void executeRotaryEmbedding(const Node& node);
    void executeGroupQueryAttention(const Node& node);
    void executeSkipSimplifiedLayerNormalization(const Node& node);

    // Helper: get or create a tensor
    std::shared_ptr<Tensor> getTensor(const std::string& name);

    // Helper: check if a tensor exists
    bool hasTensor(const std::string& name) const;

    // Helper: allocate output tensor based on operation
    std::shared_ptr<Tensor> allocateOutput(const std::vector<int64_t>& shape,
                                           DataType dtype = DataType::FLOAT32);

    // Helper: transpose a matrix
    void transposeMatrix(const float* input, float* output, int rows, int cols);
};

} // namespace onnx_runner
