#pragma once

#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include "../utils/logger.hpp"
#include <map>
#include <string>
#include <memory>
#include <cuda_runtime.h>

namespace onnx_runner {

// Timer for benchmarking GPU operations
class GPUTimer {
public:
    GPUTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }

    float elapsedMilliseconds() {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// GpuExecutor manages tensor allocation and graph execution on GPU
class GpuExecutor {
public:
    GpuExecutor(bool use_cpu_fallback = false, int num_threads = 1)
        : use_cpu_fallback_(use_cpu_fallback), num_cpu_threads_(num_threads) {}

    // Execute the graph with given inputs
    // inputs: map of input names to input tensors
    // Returns: map of output names to output tensors
    std::map<std::string, std::shared_ptr<Tensor>>
    execute(const Graph& graph,
            const std::map<std::string, std::shared_ptr<Tensor>>& inputs);

    // Set whether to print detailed timing information
    void setVerbose(bool verbose) { verbose_ = verbose; }

private:
    bool use_cpu_fallback_;
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
    void transposeMatrix(const float* input, float* output, int rows, int cols, bool use_cpu);
};

} // namespace onnx_runner
