#include "benchmark.hpp"
#include "gpu_executor.hpp"
#include "../utils/logger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstring>

namespace onnx_runner {

std::string BenchmarkResults::toJSON() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "{\n";
    ss << "  \"operations\": [\n";

    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& op = operations[i];
        ss << "    {\n";
        ss << "      \"node_name\": \"" << op.node_name << "\",\n";
        ss << "      \"op_type\": \"" << op.op_type << "\",\n";

        // Output CPU times for each thread count
        for (const auto& [threads, time_ms] : op.cpu_times_ms) {
            ss << "      \"cpu_" << threads << "_thread";
            if (threads > 1) ss << "s";
            ss << "_ms\": " << time_ms << ",\n";
        }

        ss << "      \"gpu_ms\": " << op.gpu_time_ms << "\n";
        ss << "    }";
        if (i < operations.size() - 1) ss << ",";
        ss << "\n";
    }

    ss << "  ],\n";

    // Output total times for each thread count
    for (const auto& [threads, time_ms] : total_cpu_times_ms) {
        ss << "  \"total_cpu_" << threads << "_thread";
        if (threads > 1) ss << "s";
        ss << "_ms\": " << time_ms << ",\n";
    }

    ss << "  \"total_gpu_ms\": " << total_gpu_time_ms << ",\n";
    ss << "  \"max_threads\": " << max_threads << "\n";
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

// Helper to create a progress bar
std::string createProgressBar(float ratio, int width = 30) {
    int filled = static_cast<int>(ratio * width);
    std::string bar = "[";
    for (int i = 0; i < width; ++i) {
        if (i < filled) {
            bar += "=";
        } else if (i == filled) {
            bar += ">";
        } else {
            bar += " ";
        }
    }
    bar += "]";
    return bar;
}

void BenchmarkExecutor::displayProgress(const std::string& op_name,
                                       const std::map<int, float>& cpu_times,
                                       float gpu_time,
                                       int current,
                                       int total) {
    // Clear previous output
    std::cout << "\r\033[K";

    // Display operation name and progress
    std::cout << colors::BOLD << "[" << current << "/" << total << "] "
              << colors::CYAN << op_name << colors::RESET << "\n";

    // Find max time for scaling bars
    float max_time = gpu_time;
    for (const auto& [threads, time_ms] : cpu_times) {
        max_time = std::max(max_time, time_ms);
    }

    // Display CPU times for each thread count (show only 1, max/2, and max for brevity)
    std::vector<int> threads_to_show;
    for (const auto& [threads, _] : cpu_times) {
        if (threads == 1 || threads == max_threads_ / 2 || threads == max_threads_) {
            threads_to_show.push_back(threads);
        }
    }

    for (int threads : threads_to_show) {
        float time_ms = cpu_times.at(threads);
        float ratio = (max_time > 0) ? (time_ms / max_time) : 0;

        std::string label = "CPU (" + std::to_string(threads) + "T):";
        std::cout << "  " << colors::YELLOW << std::setw(12) << std::left << label << colors::RESET
                  << createProgressBar(ratio, 30)
                  << " " << std::fixed << std::setprecision(3)
                  << time_ms << " ms\n";
    }

    // GPU timing bar
    float gpu_ratio = (max_time > 0) ? (gpu_time / max_time) : 0;
    std::cout << "  " << colors::GREEN << std::setw(12) << std::left << "GPU:" << colors::RESET
              << createProgressBar(gpu_ratio, 30)
              << " " << std::fixed << std::setprecision(3)
              << gpu_time << " ms";

    // Speedup indicator (compare to single-threaded CPU)
    if (gpu_time > 0 && cpu_times.count(1) > 0) {
        float speedup = cpu_times.at(1) / gpu_time;
        std::cout << " " << colors::MAGENTA << "(";
        if (speedup > 1.0f) {
            std::cout << speedup << "x faster)";
        } else {
            std::cout << (1.0f / speedup) << "x slower)";
        }
        std::cout << colors::RESET;
    }

    std::cout << "\n\n";
    std::cout.flush();
}

void BenchmarkExecutor::displaySummary(const BenchmarkResults& results) {
    std::cout << "\n" << colors::BOLD << colors::CYAN
              << "=== Benchmark Summary ===" << colors::RESET << "\n\n";

    // Build header dynamically based on available thread counts
    std::cout << std::left << std::setw(20) << "Operation";

    // Show selected thread counts (1, mid, max)
    std::vector<int> threads_to_show;
    if (results.total_cpu_times_ms.count(1)) threads_to_show.push_back(1);
    if (results.max_threads > 2 && results.total_cpu_times_ms.count(results.max_threads / 2)) {
        threads_to_show.push_back(results.max_threads / 2);
    }
    if (results.total_cpu_times_ms.count(results.max_threads)) {
        threads_to_show.push_back(results.max_threads);
    }

    for (int threads : threads_to_show) {
        std::string header = "CPU-" + std::to_string(threads) + "T (ms)";
        std::cout << std::setw(15) << header;
    }
    std::cout << std::setw(13) << "GPU (ms)"
              << std::setw(12) << "GPU vs 1T"
              << "\n";

    int header_width = 20 + 15 * threads_to_show.size() + 13 + 12;
    std::cout << std::string(header_width, '-') << "\n";

    // Each operation
    for (const auto& op : results.operations) {
        std::cout << std::setw(20) << op.op_type;

        for (int threads : threads_to_show) {
            if (op.cpu_times_ms.count(threads)) {
                std::cout << std::setw(15) << std::fixed << std::setprecision(3)
                          << op.cpu_times_ms.at(threads);
            } else {
                std::cout << std::setw(15) << "N/A";
            }
        }

        std::cout << std::setw(13) << std::fixed << std::setprecision(3) << op.gpu_time_ms;

        // Color-coded speedup (GPU vs single-threaded CPU)
        if (op.cpu_times_ms.count(1) && op.gpu_time_ms > 0) {
            float speedup = op.cpu_times_ms.at(1) / op.gpu_time_ms;
            if (speedup > 1.0f) {
                std::cout << colors::GREEN << std::setw(12) << (std::to_string(speedup) + "x")
                          << colors::RESET;
            } else {
                std::cout << colors::RED << std::setw(12) << (std::to_string(1.0f / speedup) + "x slower")
                          << colors::RESET;
            }
        }
        std::cout << "\n";
    }

    std::cout << std::string(header_width, '-') << "\n";

    // Totals
    std::cout << colors::BOLD << std::setw(20) << "TOTAL";

    for (int threads : threads_to_show) {
        if (results.total_cpu_times_ms.count(threads)) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(3)
                      << results.total_cpu_times_ms.at(threads);
        } else {
            std::cout << std::setw(15) << "N/A";
        }
    }

    std::cout << std::setw(13) << std::fixed << std::setprecision(3) << results.total_gpu_time_ms;

    if (results.total_cpu_times_ms.count(1) && results.total_gpu_time_ms > 0) {
        float speedup = results.total_cpu_times_ms.at(1) / results.total_gpu_time_ms;
        if (speedup > 1.0f) {
            std::cout << colors::GREEN << std::setw(12) << (std::to_string(speedup) + "x")
                      << colors::RESET;
        } else {
            std::cout << colors::RED << std::setw(12) << (std::to_string(1.0f / speedup) + "x slower")
                      << colors::RESET;
        }
    }
    std::cout << colors::RESET << "\n";

    // Additional speedup comparisons
    if (results.total_cpu_times_ms.count(1) && results.total_cpu_times_ms.count(results.max_threads)) {
        float mt_speedup = results.total_cpu_times_ms.at(1) / results.total_cpu_times_ms.at(results.max_threads);
        std::cout << "\n" << colors::BOLD << "Multi-threading Speedup:" << colors::RESET
                  << " " << colors::MAGENTA << mt_speedup << "x" << colors::RESET
                  << " (CPU-1T vs CPU-" << results.max_threads << "T)\n";
    }

    if (results.total_cpu_times_ms.count(results.max_threads) && results.total_gpu_time_ms > 0) {
        float gpu_vs_mt = results.total_cpu_times_ms.at(results.max_threads) / results.total_gpu_time_ms;
        std::cout << colors::BOLD << "GPU vs MT-CPU Speedup:" << colors::RESET
                  << " " << colors::GREEN << gpu_vs_mt << "x" << colors::RESET << "\n\n";
    }
}

BenchmarkExecutor::BenchmarkExecutor(int max_threads) {
    if (max_threads <= 0) {
        max_threads_ = std::thread::hardware_concurrency();
        if (max_threads_ == 0) max_threads_ = 4;  // Fallback if hardware_concurrency fails
    } else {
        max_threads_ = max_threads;
    }
}

std::pair<BenchmarkResults, std::map<std::string, std::shared_ptr<Tensor>>>
BenchmarkExecutor::runBenchmark(const Graph& graph,
                               const std::map<std::string, std::shared_ptr<Tensor>>& inputs,
                               bool show_live_visualization) {
    LOG_INFO("=== Starting Multi-Configuration Benchmark (CPU 1-" + std::to_string(max_threads_) + " threads + GPU) ===\n");

    BenchmarkResults results;
    results.max_threads = max_threads_;

    // Get sorted nodes
    auto nodes = graph.topologicalSort();
    int total_stages = max_threads_ + 1;  // CPU with 1 to max_threads, plus GPU

    // Maps to store total times for each configuration
    std::map<int, float> cpu_total_times;

    // Run CPU benchmarks for each thread count
    for (int num_threads = 1; num_threads <= max_threads_; ++num_threads) {
        if (show_live_visualization) {
            std::cout << colors::BOLD << colors::YELLOW
                      << "\n[Stage " << num_threads << "/" << total_stages
                      << "] Running on CPU (" << num_threads << " thread" << (num_threads > 1 ? "s" : "")
                      << ")..." << colors::RESET << "\n\n";
        }

        GpuExecutor cpu_executor(true, num_threads);
        cpu_executor.setVerbose(false);

        // Create inputs for this run
        std::map<std::string, std::shared_ptr<Tensor>> cpu_inputs;
        for (const auto& [name, tensor] : inputs) {
            auto cpu_tensor = std::make_shared<Tensor>(tensor->shape(), tensor->dtype());
            std::memcpy(cpu_tensor->data<float>(), tensor->data<float>(),
                       tensor->size() * sizeof(float));
            cpu_inputs[name] = cpu_tensor;
        }

        auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_outputs = cpu_executor.execute(graph, cpu_inputs);
        auto cpu_end = std::chrono::high_resolution_clock::now();

        float total_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        cpu_total_times[num_threads] = total_time;
        results.total_cpu_times_ms[num_threads] = total_time;
    }

    // Run GPU benchmark
    if (show_live_visualization) {
        std::cout << colors::BOLD << colors::GREEN
                  << "\n[Stage " << total_stages << "/" << total_stages
                  << "] Running on GPU..." << colors::RESET << "\n\n";
    }

    GpuExecutor gpu_executor(false);
    gpu_executor.setVerbose(false);

    std::map<std::string, std::shared_ptr<Tensor>> gpu_inputs;
    for (const auto& [name, tensor] : inputs) {
        auto gpu_tensor = std::make_shared<Tensor>(tensor->shape(), tensor->dtype());
        std::memcpy(gpu_tensor->data<float>(), tensor->data<float>(),
                   tensor->size() * sizeof(float));
        gpu_inputs[name] = gpu_tensor;
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();
    auto gpu_outputs = gpu_executor.execute(graph, gpu_inputs);
    auto gpu_end = std::chrono::high_resolution_clock::now();

    float total_gpu_time = std::chrono::duration<float, std::milli>(gpu_end - gpu_start).count();
    results.total_gpu_time_ms = total_gpu_time;

    // Build per-operation results (approximation based on total times)
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];

        OperationTiming timing;
        timing.node_name = node->name();
        timing.op_type = opTypeToString(node->opType());

        // Approximate per-operation times by dividing total time by number of operations
        for (const auto& [threads, total_time] : cpu_total_times) {
            timing.cpu_times_ms[threads] = total_time / nodes.size();
        }
        timing.gpu_time_ms = total_gpu_time / nodes.size();

        results.operations.push_back(timing);

        if (show_live_visualization) {
            displayProgress(timing.op_type, timing.cpu_times_ms, timing.gpu_time_ms,
                          i + 1, nodes.size());
            // Small delay to make visualization visible
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
    }

    // Display summary
    if (show_live_visualization) {
        displaySummary(results);
    }

    LOG_INFO("=== Benchmark Complete ===");

    return {results, gpu_outputs};
}

} // namespace onnx_runner
