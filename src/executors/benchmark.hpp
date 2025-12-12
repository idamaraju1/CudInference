#pragma once

#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include <map>
#include <string>
#include <memory>
#include <vector>

namespace onnx_runner {

// Timing data for a single operation
struct OperationTiming {
    std::string node_name;
    std::string op_type;
    std::map<int, float> cpu_times_ms;  // thread_count -> time_ms (1 to max_threads)
    float gpu_time_ms;
};

// Results from benchmark execution
struct BenchmarkResults {
    std::vector<OperationTiming> operations;
    std::map<int, float> total_cpu_times_ms;  // thread_count -> total_time_ms
    float total_gpu_time_ms;
    int max_threads;  // Maximum number of threads tested

    // Export to JSON format
    std::string toJSON() const;
};

// Benchmark executor - runs CPU (1 to max threads) and GPU benchmarks
class BenchmarkExecutor {
public:
    BenchmarkExecutor(int max_threads = 0);  // 0 = use hardware concurrency

    // Run benchmark comparing CPU (1 to max threads) and GPU execution
    // Returns timing results and output tensors (from GPU execution)
    std::pair<BenchmarkResults, std::map<std::string, std::shared_ptr<Tensor>>>
    runBenchmark(const Graph& graph,
                 const std::map<std::string, std::shared_ptr<Tensor>>& inputs,
                 bool show_live_visualization = true);

private:
    int max_threads_;  // Maximum number of threads to test

    // Display live progress during benchmark
    void displayProgress(const std::string& op_name,
                        const std::map<int, float>& cpu_times,
                        float gpu_time,
                        int current,
                        int total);

    // Display final summary
    void displaySummary(const BenchmarkResults& results);
};

} // namespace onnx_runner
