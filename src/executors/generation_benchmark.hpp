#pragma once

#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include <map>
#include <string>
#include <memory>
#include <vector>

namespace onnx_runner {

// Timing data for a single configuration
struct GenerationTiming {
    std::string config_name;    // e.g., "CPU-1T", "CPU-4T", "GPU"
    int num_threads;             // 0 for GPU, 1-N for CPU
    bool is_gpu;
    int tokens_generated;        // Number of tokens generated
    float total_time_ms;         // Total generation time in milliseconds
    float tokens_per_sec;        // Throughput: tokens/second
    float prefill_time_ms;       // Time for prefill pass (first token)
    float decode_time_ms;        // Total time for decode passes (subsequent tokens)
    float avg_decode_latency_ms; // Average time per decode token
};

// Results from generation benchmark
struct GenerationBenchmarkResults {
    std::vector<GenerationTiming> configurations;
    int max_threads;             // Maximum number of threads tested
    int prompt_tokens;           // Number of tokens in the prompt
    int target_tokens;           // Target number of tokens to generate
    std::string prompt_text;     // Input prompt used for benchmarking

    // Export to JSON format
    std::string toJSON() const;
};

// Benchmark executor for autoregressive text generation
class GenerationBenchmarkExecutor {
public:
    GenerationBenchmarkExecutor(int max_threads = 0);  // 0 = use hardware concurrency

    // Run benchmark comparing CPU (1 to max threads) and GPU text generation
    // Returns timing results for each configuration
    GenerationBenchmarkResults runBenchmark(
        const Graph& graph,
        const std::string& prompt_text,
        const std::string& tokenizer_path,
        int max_tokens = 50,
        float temperature = 0.0f,  // Default to greedy for deterministic results
        bool show_live_visualization = true);

private:
    int max_threads_;  // Maximum number of threads to test

    // Display progress during benchmark
    void displayProgress(const std::string& config_name,
                        int current,
                        int total);

    // Display final summary
    void displaySummary(const GenerationBenchmarkResults& results);
};

} // namespace onnx_runner
