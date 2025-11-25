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

// GpuExecutor manages tensor allocation and graph execution on GPU
class GpuExecutor : public Executor {
public:
    // Execution modes for GPU memory management
    enum class ExecutionMode {
        CPU_ONLY,       // All tensors on CPU, use CPU kernels
        GPU_COPY,       // Current mode: copy per operation (legacy)
        GPU_PERSISTENT  // NEW: Keep tensors on GPU, minimal transfers
    };

    GpuExecutor(bool use_cpu_fallback = false, int num_threads = 1)
        : use_cpu_fallback_(use_cpu_fallback), num_cpu_threads_(num_threads) {
        // Default to GPU_COPY for backwards compatibility
        exec_mode_ = use_cpu_fallback ? ExecutionMode::CPU_ONLY : ExecutionMode::GPU_COPY;
    }

    // Execute the graph with given inputs
    // inputs: map of input names to input tensors
    // Returns: map of output names to output tensors
    std::map<std::string, std::shared_ptr<Tensor>>
    execute(const Graph& graph,
            const std::map<std::string, std::shared_ptr<Tensor>>& inputs) override;

    // Set whether to print detailed timing information
    void setVerbose(bool verbose) override { verbose_ = verbose; }

    // Set execution mode (for persistent GPU memory)
    void setExecutionMode(ExecutionMode mode) { exec_mode_ = mode; }
    ExecutionMode getExecutionMode() const { return exec_mode_; }

private:
    bool use_cpu_fallback_;
    bool verbose_ = false;
    int num_cpu_threads_;
    ExecutionMode exec_mode_ = ExecutionMode::GPU_COPY;

    // Tensor storage during execution
    // Maps tensor name to tensor data
    std::map<std::string, std::shared_ptr<Tensor>> tensors_;

    // NEW: Persistent KV cache for autoregressive generation
    struct KVCacheEntry {
        std::shared_ptr<Tensor> key_cache;    // [batch, kv_heads, max_seq, head_dim]
        std::shared_ptr<Tensor> value_cache;  // [batch, kv_heads, max_seq, head_dim]
        int current_length = 0;               // How many tokens cached
        int max_length = 0;                   // Maximum sequence length
    };

    std::map<std::string, KVCacheEntry> kv_cache_;

    // Initialize cache for a layer (allocates on GPU)
    void initializeKVCache(
        const std::string& cache_key,
        int batch,
        int kv_heads,
        int max_seq_length,
        int head_dim
    );

    // Append new keys/values to cache (GPU-to-GPU copy)
    void appendKVCache(
        const std::string& cache_key,
        const std::shared_ptr<Tensor>& new_keys,
        const std::shared_ptr<Tensor>& new_values
    );

    // Get current cache tensors (returns view slices)
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
    getKVCache(const std::string& cache_key);

    // Clear all KV caches (for new generation session)
    void clearKVCaches() { kv_cache_.clear(); }

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

    // Helper: get GPU pointer from tensor in GPU_PERSISTENT mode
    template<typename T>
    const T* getGPUData(const std::shared_ptr<Tensor>& tensor) {
        if (exec_mode_ != ExecutionMode::GPU_PERSISTENT) {
            throw std::runtime_error("getGPUData() called but not in GPU_PERSISTENT mode");
        }
        if (!tensor->isOnGPU()) {
            throw std::runtime_error("Expected GPU tensor in GPU_PERSISTENT mode");
        }
        return tensor->deviceData<T>();
    }

    template<typename T>
    T* getMutableGPUData(const std::shared_ptr<Tensor>& tensor) {
        if (exec_mode_ != ExecutionMode::GPU_PERSISTENT) {
            throw std::runtime_error("getMutableGPUData() called but not in GPU_PERSISTENT mode");
        }
        if (!tensor->isOnGPU()) {
            throw std::runtime_error("Expected GPU tensor in GPU_PERSISTENT mode");
        }
        return tensor->mutableDeviceData<T>();
    }
};

} // namespace onnx_runner
