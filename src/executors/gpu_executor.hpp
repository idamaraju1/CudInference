#pragma once

#include "executor.hpp"
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

/**
 * GPU executor implementation.
 * Executes ONNX computation graphs using CUDA kernels.
 */
class GpuExecutor : public Executor {
public:
    GpuExecutor() = default;

    // Implement Executor interface
    std::map<std::string, std::shared_ptr<Tensor>>
    execute(const Graph& graph,
            const std::map<std::string, std::shared_ptr<Tensor>>& inputs) override;

    void setVerbose(bool verbose) override { verbose_ = verbose; }
    DeviceType deviceType() const override { return DeviceType::CUDA; }
    std::string name() const override { return "GPU"; }

protected:
    // Override base class helpers for GPU-specific behavior
    std::shared_ptr<Tensor> allocateOutput(
        const std::vector<int64_t>& shape,
        DataType dtype = DataType::FLOAT32) override;

    void initializeInputs(
        const std::map<std::string, std::shared_ptr<Tensor>>& inputs) override;

    void initializeInitializers(const Graph& graph) override;

    std::map<std::string, std::shared_ptr<Tensor>>
    collectOutputs(const Graph& graph) override;

    // Must implement pure virtual from base class
    void executeNode(const Node& node) override;

private:
    // Operation implementations
    void executeMatMul(const Node& node);
    void executeReLU(const Node& node);
    void executeAdd(const Node& node);
    void executeSub(const Node& node);
    void executeGemm(const Node& node);
    void executeGather(const Node& node);
    void executeMul(const Node& node);
    void executeDiv(const Node& node);
    void executePow(const Node& node);
    void executeSqrt(const Node& node);
    void executeReduceMean(const Node& node);
    void executeReshape(const Node& node);
    void executeTranspose(const Node& node);
    void executeUnsqueeze(const Node& node);
    void executeSlice(const Node& node);
    void executeConcat(const Node& node);
    void executeShape(const Node& node);
    void executeCast(const Node& node);
    void executeRange(const Node& node);
    void executeEqual(const Node& node);
    void executeConstantOfShape(const Node& node);
    void executeExpand(const Node& node);
    void executeGreater(const Node& node);
    void executeNeg(const Node& node);
    void executeSigmoid(const Node& node);
    void executeSin(const Node& node);
    void executeCos(const Node& node);
    void executeSoftmax(const Node& node);
    void executeScatterND(const Node& node);
    void executeTrilu(const Node& node);
    void executeWhere(const Node& node);
    void executeReduceSum(const Node& node);
    void executeSimplifiedLayerNormalization(const Node& node);
    void executeRotaryEmbedding(const Node& node);
    void executeGroupQueryAttention(const Node& node);
    void executeSkipSimplifiedLayerNormalization(const Node& node);

    // Helper: transpose a matrix
    void transposeMatrix(const float* input, float* output, int rows, int cols, bool use_cpu);
};

} // namespace onnx_runner

