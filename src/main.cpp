#include "core/model_parser.hpp"
#include "core/graph.hpp"
#include "gpu/gpu_executor.hpp"
#include "gpu/benchmark.hpp"
#include "utils/logger.hpp"
#include "utils/tensor.hpp"
#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>

using namespace onnx_runner;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model.onnx> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --cpu             Use CPU fallback instead of GPU\n";
    std::cout << "  --cpu-threads N   Max CPU threads for benchmark mode (default: auto-detect)\n";
    std::cout << "                    Benchmark will test 1 to N threads\n";
    std::cout << "  --verbose         Print detailed timing information\n";
    std::cout << "  --debug           Enable debug logging\n";
    std::cout << "  --benchmark       Run multi-configuration benchmark (CPU 1-N threads + GPU)\n";
    std::cout << "  --output FILE     Save benchmark results to JSON file (default: results.json)\n";
    std::cout << "  --help            Show this help message\n";
}

// Helper to create a simple test input tensor
std::shared_ptr<Tensor> createTestInput(const std::vector<int64_t>& shape) {
    auto tensor = std::make_shared<Tensor>(shape, DataType::FLOAT32);

    // Fill with simple test data (e.g., sequential values)
    float* data = tensor->data<float>();
    for (size_t i = 0; i < tensor->size(); ++i) {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    return tensor;
}

// Print first few values of a tensor for debugging
void printTensorSample(const std::string& name, const Tensor& tensor, int max_values = 10) {
    std::cout << name << " " << tensor.shapeStr() << ": [";

    const float* data = tensor.data<float>();
    int count = std::min(static_cast<int>(tensor.size()), max_values);

    for (int i = 0; i < count; ++i) {
        std::cout << data[i];
        if (i < count - 1) std::cout << ", ";
    }

    if (tensor.size() > static_cast<size_t>(max_values)) {
        std::cout << ", ...";
    }

    std::cout << "]\n";
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string model_path;
    bool use_cpu = false;
    bool verbose = false;
    bool debug = false;
    bool benchmark = false;
    std::string output_file;
    int cpu_threads = 0;  // Default to 0 (auto-detect hardware concurrency)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") {
            use_cpu = true;
        } else if (arg == "--cpu-threads") {
            if (i + 1 < argc) {
                cpu_threads = std::atoi(argv[++i]);
                if (cpu_threads < 1) {
                    std::cerr << "Error: --cpu-threads must be >= 1\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --cpu-threads requires a number\n";
                return 1;
            }
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cerr << "Error: --output requires a filename\n";
                return 1;
            }
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            model_path = arg;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: No model file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    // Configure logger
    if (debug) {
        Logger::instance().setLevel(LogLevel::DEBUG);
    }

    LOG_INFO("=== OnnxRunner GPU Engine ===");
    LOG_INFO("Model: ", model_path);
    LOG_INFO("Device: ", use_cpu ? "CPU" : "GPU");

    try {
        // Step 1: Parse the model
        auto start_time = std::chrono::high_resolution_clock::now();

        ModelParser parser;
        auto graph = parser.parse(model_path);

        auto parse_time = std::chrono::high_resolution_clock::now();
        auto parse_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            parse_time - start_time).count();

        LOG_INFO("Model parsing took ", parse_duration, " ms");

        // Step 2: Print graph summary
        graph->printSummary();

        // Step 3: Create test inputs
        // For a real application, you would provide actual input data
        std::map<std::string, std::shared_ptr<Tensor>> inputs;

        if (graph->inputs().empty()) {
            LOG_WARN("No graph inputs defined - graph might be self-contained");
        } else {
            LOG_INFO("\n=== Creating Test Inputs ===");
            for (const auto& input_name : graph->inputs()) {
                LOG_INFO("Creating test input for: ", input_name);

                // Get the expected input shape from the model
                auto input_shape = graph->getInputShape(input_name);

                if (input_shape.empty()) {
                    // Fallback to default shape if not specified
                    LOG_WARN("  No shape info for input '", input_name, "', using default [1, 10]");
                    input_shape = {1, 10};
                }

                // Create test input with the correct shape
                auto input_tensor = createTestInput(input_shape);
                inputs[input_name] = input_tensor;

                printTensorSample("Input " + input_name, *input_tensor);
            }
        }

        // Step 4: Execute the graph
        std::map<std::string, std::shared_ptr<Tensor>> outputs;
        BenchmarkResults bench_results;

        if (benchmark) {
            // Run benchmark mode
            BenchmarkExecutor bench_executor(cpu_threads);
            auto [results, bench_outputs] = bench_executor.runBenchmark(*graph, inputs, true);
            outputs = bench_outputs;
            bench_results = results;

            // Save to JSON (default to results.json if not specified)
            std::string json_output = output_file.empty() ? "results.json" : output_file;
            std::ofstream out(json_output);
            if (out.is_open()) {
                out << bench_results.toJSON();
                out.close();
                LOG_INFO("Benchmark results saved to: ", json_output);
            } else {
                LOG_ERROR("Failed to open output file: ", json_output);
            }
        } else {
            // Normal execution mode
            LOG_INFO("\n=== Executing Graph ===");

            GpuExecutor executor(use_cpu);
            executor.setVerbose(verbose);

            auto exec_start = std::chrono::high_resolution_clock::now();

            outputs = executor.execute(*graph, inputs);

            auto exec_end = std::chrono::high_resolution_clock::now();
            auto exec_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                exec_end - exec_start).count();

            LOG_INFO("Graph execution took ", exec_duration, " ms");

            // Step 6: Performance summary
            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                exec_end - start_time).count();

            LOG_INFO("\n=== Performance Summary ===");
            LOG_INFO("Total time: ", total_time, " ms");
            LOG_INFO("  - Parsing: ", parse_duration, " ms");
            LOG_INFO("  - Execution: ", exec_duration, " ms");
        }

        // Step 5: Display outputs
        LOG_INFO("\n=== Outputs ===");
        for (const auto& [name, tensor] : outputs) {
            printTensorSample("Output " + name, *tensor);
        }

        LOG_INFO("\n=== Execution Successful ===");
        return 0;

    } catch (const std::exception& e) {
        LOG_ERROR("Error: ", e.what());
        return 1;
    }
}
