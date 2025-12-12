#include "generation_benchmark.hpp"
#include "gpu_executor.hpp"
#include "cpu_executor.hpp"
#include "autoregressive_generator.hpp"
#include "../utils/logger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>

namespace onnx_runner {

std::string GenerationBenchmarkResults::toJSON() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "{\n";
    ss << "  \"prompt_text\": \"" << prompt_text << "\",\n";
    ss << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    ss << "  \"target_tokens\": " << target_tokens << ",\n";
    ss << "  \"max_threads\": " << max_threads << ",\n";
    ss << "  \"configurations\": [\n";

    for (size_t i = 0; i < configurations.size(); ++i) {
        const auto& config = configurations[i];
        ss << "    {\n";
        ss << "      \"name\": \"" << config.config_name << "\",\n";
        ss << "      \"num_threads\": " << config.num_threads << ",\n";
        ss << "      \"is_gpu\": " << (config.is_gpu ? "true" : "false") << ",\n";
        ss << "      \"tokens_generated\": " << config.tokens_generated << ",\n";
        ss << "      \"total_time_ms\": " << config.total_time_ms << ",\n";
        ss << "      \"tokens_per_sec\": " << config.tokens_per_sec << ",\n";
        ss << "      \"prefill_time_ms\": " << config.prefill_time_ms << ",\n";
        ss << "      \"decode_time_ms\": " << config.decode_time_ms << ",\n";
        ss << "      \"avg_decode_latency_ms\": " << config.avg_decode_latency_ms << "\n";
        ss << "    }";
        if (i < configurations.size() - 1) ss << ",";
        ss << "\n";
    }

    ss << "  ],\n";

    // Add summary statistics for compatibility with visualization
    for (const auto& config : configurations) {
        std::string prefix = config.is_gpu ? "total_gpu" :
                           ("total_cpu_" + std::to_string(config.num_threads) + "_thread");
        if (config.num_threads > 1) prefix += "s";
        ss << "  \"" << prefix << "_tokens_per_sec\": " << config.tokens_per_sec << ",\n";
    }

    // Find fastest configuration
    float max_throughput = 0.0f;
    for (const auto& config : configurations) {
        max_throughput = std::max(max_throughput, config.tokens_per_sec);
    }

    ss << "  \"max_throughput_tokens_per_sec\": " << max_throughput << "\n";
    ss << "}\n";

    return ss.str();
}

// ANSI color codes for terminal output
namespace colors {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string MAGENTA = "\033[35m";
    const std::string RED = "\033[31m";
}

void GenerationBenchmarkExecutor::displayProgress(const std::string& config_name,
                                                  int current,
                                                  int total) {
    std::cout << colors::BOLD << "[" << current << "/" << total << "] "
              << colors::CYAN << "Running: " << config_name << colors::RESET << "\n";
    std::cout.flush();
}

void GenerationBenchmarkExecutor::displaySummary(const GenerationBenchmarkResults& results) {
    std::cout << "\n" << colors::BOLD << colors::CYAN
              << "=== Generation Benchmark Summary ===" << colors::RESET << "\n\n";

    std::cout << "Prompt: \"" << results.prompt_text << "\"\n";
    std::cout << "Prompt tokens: " << results.prompt_tokens << "\n";
    std::cout << "Target tokens: " << results.target_tokens << "\n\n";

    // Table header
    std::cout << std::left << std::setw(15) << "Configuration"
              << std::setw(12) << "Tokens Gen"
              << std::setw(14) << "Total (ms)"
              << std::setw(15) << "Tokens/sec"
              << std::setw(14) << "Prefill (ms)"
              << std::setw(14) << "Decode (ms)"
              << std::setw(18) << "Avg Decode (ms)"
              << "\n";

    int header_width = 15 + 12 + 14 + 15 + 14 + 14 + 18;
    std::cout << std::string(header_width, '-') << "\n";

    // Each configuration
    for (const auto& config : results.configurations) {
        std::string color = config.is_gpu ? colors::GREEN : colors::YELLOW;

        std::cout << color << std::setw(15) << config.config_name << colors::RESET
                  << std::setw(12) << config.tokens_generated
                  << std::setw(14) << std::fixed << std::setprecision(2) << config.total_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << config.tokens_per_sec
                  << std::setw(14) << std::fixed << std::setprecision(2) << config.prefill_time_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << config.decode_time_ms
                  << std::setw(18) << std::fixed << std::setprecision(2) << config.avg_decode_latency_ms
                  << "\n";
    }

    std::cout << std::string(header_width, '-') << "\n";

    // Find best configurations
    auto max_throughput_it = std::max_element(results.configurations.begin(),
                                             results.configurations.end(),
                                             [](const GenerationTiming& a, const GenerationTiming& b) {
                                                 return a.tokens_per_sec < b.tokens_per_sec;
                                             });

    if (max_throughput_it != results.configurations.end()) {
        std::cout << "\n" << colors::BOLD << "Best Throughput: " << colors::RESET
                  << colors::GREEN << max_throughput_it->config_name << colors::RESET
                  << " with " << colors::BOLD << max_throughput_it->tokens_per_sec
                  << " tokens/sec" << colors::RESET << "\n";

        // Show speedup comparisons
        auto cpu1_it = std::find_if(results.configurations.begin(), results.configurations.end(),
                                    [](const GenerationTiming& t) { return !t.is_gpu && t.num_threads == 1; });

        if (cpu1_it != results.configurations.end()) {
            float speedup = max_throughput_it->tokens_per_sec / cpu1_it->tokens_per_sec;
            std::cout << colors::BOLD << "Speedup vs CPU-1T: " << colors::RESET
                      << colors::MAGENTA << speedup << "x" << colors::RESET << "\n";
        }
    }

    std::cout << "\n";
}

GenerationBenchmarkExecutor::GenerationBenchmarkExecutor(int max_threads) {
    if (max_threads <= 0) {
        max_threads_ = std::thread::hardware_concurrency();
        if (max_threads_ == 0) max_threads_ = 4;  // Fallback
    } else {
        max_threads_ = max_threads;
    }
}

GenerationBenchmarkResults GenerationBenchmarkExecutor::runBenchmark(
    const Graph& graph,
    const std::string& prompt_text,
    const std::string& tokenizer_path,
    int max_tokens,
    float temperature,
    bool show_live_visualization) {

    LOG_INFO("=== Starting Generation Benchmark (CPU 1-" + std::to_string(max_threads_)
             + " threads + GPU) ===\n");
    LOG_INFO("Prompt: \"" + prompt_text + "\"");
    LOG_INFO("Max tokens to generate: " + std::to_string(max_tokens));
    LOG_INFO("Temperature: " + std::to_string(temperature));

    GenerationBenchmarkResults results;
    results.max_threads = max_threads_;
    results.prompt_text = prompt_text;
    results.target_tokens = max_tokens;

    int total_configs = max_threads_ + 1;  // CPU 1-N threads + GPU
    int current_config = 0;

    // Store prompt token count (will be set after first run)
    int prompt_token_count = 0;

    // Run CPU benchmarks for each thread count
    for (int num_threads = 1; num_threads <= max_threads_; ++num_threads) {
        current_config++;
        std::string config_name = "CPU-" + std::to_string(num_threads) + "T";

        if (show_live_visualization) {
            displayProgress(config_name, current_config, total_configs);
        }

        try {
            // Create CPU executor
            GpuExecutor executor(true, num_threads);  // CPU mode with N threads
            executor.setVerbose(false);

            // Configure generation
            AutoregressiveGenerator::GenerationConfig gen_config;
            gen_config.max_tokens = max_tokens;
            gen_config.temperature = temperature;
            gen_config.verbose = false;
            gen_config.stream_stdout = false;

            AutoregressiveGenerator generator(executor, graph, tokenizer_path, gen_config);

            // Measure generation time (tokenization happens inside generate())
            auto start = std::chrono::high_resolution_clock::now();
            std::string generated_text = generator.generate(prompt_text);
            auto end = std::chrono::high_resolution_clock::now();

            // Note: We can't easily get prompt_token_count without tokenizing twice,
            // so we'll set it after the first run when we have timing data

            float total_time_ms = std::chrono::duration<float, std::milli>(end - start).count();

            // Calculate metrics
            // Note: We need to track prefill/decode separately in the generator
            // For now, we'll estimate based on typical patterns
            GenerationTiming timing;
            timing.config_name = config_name;
            timing.num_threads = num_threads;
            timing.is_gpu = false;
            timing.total_time_ms = total_time_ms;

            // We need to actually count tokens in generated text
            // For now, use max_tokens as approximation
            timing.tokens_generated = max_tokens;
            timing.tokens_per_sec = (timing.tokens_generated * 1000.0f) / total_time_ms;

            // Estimate prefill vs decode (these are placeholders until we instrument the generator)
            timing.prefill_time_ms = total_time_ms * 0.1f;  // Rough estimate
            timing.decode_time_ms = total_time_ms * 0.9f;
            timing.avg_decode_latency_ms = timing.decode_time_ms / std::max(1, timing.tokens_generated - 1);

            results.configurations.push_back(timing);

            if (show_live_visualization) {
                std::cout << "  " << colors::YELLOW << "→ " << colors::RESET
                          << timing.tokens_per_sec << " tokens/sec"
                          << " (" << timing.total_time_ms << " ms total)\n\n";
            }

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to benchmark " + config_name + ": " + std::string(e.what()));
            // Also print to stderr for visibility
            std::cerr << "ERROR: Failed to benchmark " << config_name << ": " << e.what() << "\n";

            // Add failed result
            GenerationTiming timing;
            timing.config_name = config_name;
            timing.num_threads = num_threads;
            timing.is_gpu = false;
            timing.tokens_generated = 0;
            timing.total_time_ms = 0;
            timing.tokens_per_sec = 0;
            timing.prefill_time_ms = 0;
            timing.decode_time_ms = 0;
            timing.avg_decode_latency_ms = 0;
            results.configurations.push_back(timing);
        }
    }

    // Run GPU benchmark
    current_config++;
    std::string config_name = "GPU";

    if (show_live_visualization) {
        displayProgress(config_name, current_config, total_configs);
    }

    try {
        // Create GPU executor
        GpuExecutor executor(false);  // GPU mode
        executor.setVerbose(false);
        executor.setExecutionMode(GpuExecutor::ExecutionMode::GPU_PERSISTENT);

        // Configure generation
        AutoregressiveGenerator::GenerationConfig gen_config;
        gen_config.max_tokens = max_tokens;
        gen_config.temperature = temperature;
        gen_config.verbose = false;
        gen_config.stream_stdout = false;

        AutoregressiveGenerator generator(executor, graph, tokenizer_path, gen_config);

        // Measure generation time
        auto start = std::chrono::high_resolution_clock::now();
        std::string generated_text = generator.generate(prompt_text);
        auto end = std::chrono::high_resolution_clock::now();

        float total_time_ms = std::chrono::duration<float, std::milli>(end - start).count();

        // Calculate metrics
        GenerationTiming timing;
        timing.config_name = config_name;
        timing.num_threads = 0;
        timing.is_gpu = true;
        timing.total_time_ms = total_time_ms;
        timing.tokens_generated = max_tokens;
        timing.tokens_per_sec = (timing.tokens_generated * 1000.0f) / total_time_ms;
        timing.prefill_time_ms = total_time_ms * 0.1f;  // Rough estimate
        timing.decode_time_ms = total_time_ms * 0.9f;
        timing.avg_decode_latency_ms = timing.decode_time_ms / std::max(1, timing.tokens_generated - 1);

        results.configurations.push_back(timing);

        if (show_live_visualization) {
            std::cout << "  " << colors::GREEN << "→ " << colors::RESET
                      << timing.tokens_per_sec << " tokens/sec"
                      << " (" << timing.total_time_ms << " ms total)\n\n";
        }

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to benchmark GPU: " + std::string(e.what()));
        // Also print to stderr for visibility
        std::cerr << "ERROR: Failed to benchmark GPU: " << e.what() << "\n";

        // Add failed result
        GenerationTiming timing;
        timing.config_name = config_name;
        timing.num_threads = 0;
        timing.is_gpu = true;
        timing.tokens_generated = 0;
        timing.total_time_ms = 0;
        timing.tokens_per_sec = 0;
        timing.prefill_time_ms = 0;
        timing.decode_time_ms = 0;
        timing.avg_decode_latency_ms = 0;
        results.configurations.push_back(timing);
    }

    results.prompt_tokens = prompt_token_count;

    // Display summary
    if (show_live_visualization) {
        displaySummary(results);
    }

    LOG_INFO("=== Benchmark Complete ===");

    return results;
}

} // namespace onnx_runner
