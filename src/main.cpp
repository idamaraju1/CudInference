#include "core/model_parser.hpp"
#include "core/graph.hpp"
#include "executors/gpu_executor.hpp"
#include "executors/cpu_executor.hpp"
#include "executors/benchmark.hpp"
#include "executors/generation_benchmark.hpp"
#include "executors/autoregressive_generator.hpp"
#include "utils/logger.hpp"
#include "utils/tensor.hpp"
#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>
#include <sstream>
#include <array>
#include <cstdio>


using namespace onnx_runner;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model.onnx> [options]\n\n";
    std::cout << "Basic Options:\n";
    std::cout << "  --cpu             Use CPU fallback instead of GPU\n";
    std::cout << "  --verbose         Print detailed timing information\n";
    std::cout << "  --quiet           Suppress logs; stream generated text only\n";
    std::cout << "  --debug           Enable debug logging\n";
    std::cout << "  --help            Show this help message\n\n";
    std::cout << "Benchmark Mode:\n";
    std::cout << "  --benchmark       Run multi-configuration benchmark (CPU 1-N threads + GPU)\n";
    std::cout << "  --benchmark-generation  Run generation benchmark (requires --input and --tokenizer)\n";
    std::cout << "  --cpu-threads N   Max CPU threads for benchmark mode (default: auto-detect)\n";
    std::cout << "                    Benchmark will test 1 to N threads\n";
    std::cout << "  --output FILE     Save benchmark results to JSON file (default: results.json)\n\n";
    std::cout << "Text Generation Mode (for LLM models):\n";
    std::cout << "  --generate        Enable autoregressive text generation\n";
    std::cout << "  --input TEXT      Input text prompt to generate from (required with --generate)\n";
    std::cout << "  --tokenizer FILE  Path to tokenizer.json file (required with --generate)\n";
    std::cout << "  --max-tokens N    Maximum tokens to generate (default: 50)\n";
    std::cout << "  --temperature F   Sampling temperature (default: 1.0, 0.0=greedy)\n\n";
    std::cout << "Example (Text Generation):\n";
    std::cout << "  " << program_name << " model.onnx --input \"The sky is blue because\" \\\n";
    std::cout << "    --tokenizer tokenizer.json --generate --max-tokens 5 --temperature 0.0\n";
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

// Helper function to execute a command and capture output
std::string exec_command(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// Helper function to find the tokenizer script regardless of working directory
std::string findTokenizerScript() {
    // Try multiple possible locations
    std::vector<std::string> possible_paths = {
        "scripts/hf_tokenizer.py",           // From project root
        "../scripts/hf_tokenizer.py",        // From build/ directory
        "./hf_tokenizer.py",                 // Same directory
        "../../scripts/hf_tokenizer.py"      // From nested build directories
    };

    for (const auto& path : possible_paths) {
        std::ifstream file(path);
        if (file.good()) {
            return path;
        }
    }

    // If not found, throw an error with helpful message
    throw std::runtime_error(
        "Could not find hf_tokenizer.py script. Tried:\n"
        "  - scripts/hf_tokenizer.py\n"
        "  - ../scripts/hf_tokenizer.py\n"
        "  - ./hf_tokenizer.py\n"
        "  - ../../scripts/hf_tokenizer.py\n"
        "Make sure you're running from the project root or build/ directory."
    );
}

std::vector<int64_t> tokenizeText(const std::string& text, const std::string& tokenizer_path) {
    // Escape single quotes in text for shell command
    std::string escaped_text = text;
    size_t pos = 0;
    while ((pos = escaped_text.find("'", pos)) != std::string::npos) {
        escaped_text.replace(pos, 1, "'\\''");
        pos += 4;
    }

    // Find tokenizer script and call it
    std::string script_path = findTokenizerScript();
    std::string cmd = "python3 " + script_path + " --tokenizer '" + tokenizer_path +
                     "' --encode '" + escaped_text + "' 2>&1";

    std::string output = exec_command(cmd);

    // Check for errors
    if (output.find("ERROR") != std::string::npos) {
        std::cerr << "[Tokenizer] " << output;
        throw std::runtime_error("Failed to tokenize text");
    }

    // Parse space-separated token IDs
    std::vector<int64_t> token_ids;
    std::istringstream iss(output);
    int64_t id;
    while (iss >> id) {
        token_ids.push_back(id);
    }

    if (token_ids.empty()) {
        std::cerr << "[Tokenizer] No tokens produced from text: " << text << std::endl;
        throw std::runtime_error("Tokenization produced no tokens");
    }

    std::cout << "[Tokenizer] Loaded from: " << tokenizer_path << std::endl;
    std::cout << "[Tokenizer] Token count: " << token_ids.size() << std::endl;

    return token_ids;
}

// Decode token IDs back to text
std::string decodeTokens(const std::vector<int64_t>& token_ids, const std::string& tokenizer_path) {
    // Convert token IDs to space-separated string
    std::ostringstream ids_stream;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (i > 0) ids_stream << " ";
        ids_stream << token_ids[i];
    }
    std::string ids_str = ids_stream.str();

    // Find tokenizer script and call it
    std::string script_path = findTokenizerScript();
    std::string cmd = "python3 " + script_path + " --tokenizer '" + tokenizer_path +
                     "' --decode '" + ids_str + "' 2>&1";

    std::string output = exec_command(cmd);

    // Check for errors
    if (output.find("ERROR") != std::string::npos) {
        std::cerr << "[Tokenizer] Failed to decode tokens: " << output;
        return "";
    }

    // Remove trailing newline if present
    if (!output.empty() && output.back() == '\n') {
        output.pop_back();
    }

    return output;
}

// Print first few values of a tensor for debugging
void printTensorSample(const std::string& name, Tensor& tensor, int max_values = 10) {
    std::cout << name << " " << tensor.shapeStr() << ": [";

    if (tensor.device() == DeviceType::CUDA) {
        tensor.toCPU();
    }

    size_t count = std::min(static_cast<size_t>(max_values), tensor.size());
    switch (tensor.dtype()) {
        case DataType::FLOAT32: {
            const float* data = tensor.data<float>();
            for (size_t i = 0; i < count; ++i) {
                std::cout << data[i];
                if (i + 1 < count) std::cout << ", ";
            }
            break;
        }
        case DataType::INT32: {
            const int32_t* data = tensor.data<int32_t>();
            for (size_t i = 0; i < count; ++i) {
                std::cout << data[i];
                if (i + 1 < count) std::cout << ", ";
            }
            break;
        }
        case DataType::INT64: {
            const int64_t* data = tensor.data<int64_t>();
            for (size_t i = 0; i < count; ++i) {
                std::cout << data[i];
                if (i + 1 < count) std::cout << ", ";
            }
            break;
        }
        case DataType::UINT8: {
            const uint8_t* data = tensor.data<uint8_t>();
            for (size_t i = 0; i < count; ++i) {
                std::cout << static_cast<int>(data[i]);
                if (i + 1 < count) std::cout << ", ";
            }
            break;
        }
        case DataType::FLOAT16: {
            const uint16_t* raw = tensor.data<uint16_t>();
            for (size_t i = 0; i < count; ++i) {
                // Minimal half to float conversion
                uint16_t h = raw[i];
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t mant = (h & 0x03FF) << 13;
                uint32_t exp  = (h & 0x7C00) >> 10;
                uint32_t f;
                if (exp == 0) {
                    if (mant == 0) {
                        f = sign;
                    } else {
                        exp = 1;
                        while ((mant & 0x400000) == 0) {
                            mant <<= 1;
                            --exp;
                        }
                        mant &= 0x3FFFFF;
                        f = sign | ((exp + 127 - 15) << 23) | mant;
                    }
                } else if (exp == 31) {
                    f = sign | 0x7F800000 | mant;
                } else {
                    f = sign | ((exp + 127 - 15) << 23) | mant;
                }
                float val = *reinterpret_cast<float*>(&f);
                std::cout << val;
                if (i + 1 < count) std::cout << ", ";
            }
            break;
        }
        default:
            std::cout << "Unsupported dtype for print";
            break;
    }

    if (tensor.size() > count) {
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
    bool quiet = false;
    bool debug = false;
    bool benchmark = false;
    bool benchmark_generation = false;
    bool generate = false;
    std::string output_file;
    int cpu_threads = 0;  // Default to 0 (auto-detect hardware concurrency)
    int max_tokens = 50;  // Default max tokens for generation
    float temperature = 1.0f;  // Default temperature for sampling
    std::string user_input_text;
    std::string tokenizer_path;

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
        } else if (arg == "--quiet") {
            quiet = true;
            verbose = false;
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--benchmark-generation") {
            benchmark_generation = true;
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
        } else if (arg == "--input") {
            if (i + 1 < argc) {
                user_input_text = argv[++i];
            } else {
                std::cerr << "Error: --input requires a string\n";
                return 1;
            }
        } else if (arg == "--tokenizer") {
            if (i + 1 < argc) {
                tokenizer_path = argv[++i];
            } else {
                std::cerr << "Error: --tokenizer requires a file path\n";
                return 1;
            }
        } else if (arg == "--generate") {
            generate = true;
        } else if (arg == "--max-tokens") {
            if (i + 1 < argc) {
                max_tokens = std::atoi(argv[++i]);
                if (max_tokens < 1) {
                    std::cerr << "Error: --max-tokens must be >= 1\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --max-tokens requires a number\n";
                return 1;
            }
        } else if (arg == "--temperature") {
            if (i + 1 < argc) {
                temperature = std::atof(argv[++i]);
                if (temperature < 0.0f) {
                    std::cerr << "Error: --temperature must be >= 0.0\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --temperature requires a number\n";
                return 1;
            }
        } else if (arg[0] != '-') {
            model_path = arg;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: No model file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    if (quiet) {
        verbose = false;
        debug = false;
    }

    // Configure logger
    if (quiet) {
        Logger::instance().setLevel(LogLevel::OFF);
    } else if (debug) {
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

        // Step 2.5: Handle generation benchmark mode
        if (benchmark_generation) {
            if (user_input_text.empty()) {
                std::cerr << "Error: --benchmark-generation requires --input <text>\n";
                return 1;
            }
            if (tokenizer_path.empty()) {
                std::cerr << "Error: --benchmark-generation requires --tokenizer <path>\n";
                return 1;
            }

            LOG_INFO("\n=== Generation Benchmark Mode ===");

            // Run generation benchmark
            GenerationBenchmarkExecutor bench_executor(cpu_threads);
            GenerationBenchmarkResults results = bench_executor.runBenchmark(
                *graph, user_input_text, tokenizer_path, max_tokens, temperature, true);

            // Save to JSON (default to generation_results.json if not specified)
            std::string json_output = output_file.empty() ? "generation_results.json" : output_file;
            std::ofstream out(json_output);
            if (out.is_open()) {
                out << results.toJSON();
                out.close();
                LOG_INFO("\nGeneration benchmark results saved to: ", json_output);
            } else {
                LOG_ERROR("Failed to open output file: ", json_output);
            }

            return 0;
        }

        // Step 2.6: Handle autoregressive generation mode
        if (generate) {
            if (user_input_text.empty()) {
                std::cerr << "Error: --generate requires --input <text>\n";
                return 1;
            }
            if (tokenizer_path.empty()) {
                std::cerr << "Error: --generate requires --tokenizer <path>\n";
                return 1;
            }

            LOG_INFO("\n=== Autoregressive Generation Mode ===");
            LOG_INFO("Input prompt: \"", user_input_text, "\"");
            LOG_INFO("Max tokens: ", max_tokens);
            LOG_INFO("Temperature: ", temperature);

            // Create generator
            GpuExecutor executor(use_cpu);
            executor.setVerbose(verbose);

            // Enable GPU_PERSISTENT mode for minimal CPU-GPU transfers
            if (!use_cpu) {
                executor.setExecutionMode(GpuExecutor::ExecutionMode::GPU_PERSISTENT);
                LOG_INFO("GPU_PERSISTENT mode enabled - tensors stay on GPU");
            }

            AutoregressiveGenerator::GenerationConfig gen_config;
            gen_config.max_tokens = max_tokens;
            gen_config.temperature = temperature;
            gen_config.verbose = verbose;
            gen_config.stream_stdout = quiet;

            AutoregressiveGenerator generator(executor, *graph, tokenizer_path, gen_config);

            // Generate text
            std::string generated_text = generator.generate(user_input_text);

            // Display result (skip in quiet streaming mode to avoid duplicate text)
            if (!quiet) {
                LOG_INFO("\n=== Generated Text ===");
                std::cout << generated_text << "\n";
            }

            LOG_INFO("\n=== Generation Complete ===");
            return 0;
        }

        // Step 3: Prepare actual input tensors
        std::map<std::string, std::shared_ptr<Tensor>> inputs;

        if (graph->inputs().empty()) {
            LOG_WARN("No graph inputs defined - graph might be self-contained");
        } else {
            LOG_INFO("\n=== Preparing Model Inputs ===");
            for (const auto& input_name : graph->inputs()) {
                LOG_INFO("Preparing input for: ", input_name);

                // Get input shape from graph
                auto shape = graph->getInputShape(input_name);
                if (shape.empty()) {
                    shape = {1, 1};
                }

                if (!user_input_text.empty() && input_name == "input_ids") {
                    if (tokenizer_path.empty()) {
                        std::cerr << "Error: --tokenizer <path/to/tokenizer.model> must be provided\n";
                        return 1;
                    }
                    
                    auto token_ids = tokenizeText(user_input_text, tokenizer_path);
                    shape = {1, static_cast<int64_t>(token_ids.size())};
                    auto tensor = std::make_shared<Tensor>(shape, DataType::INT64);
                    std::memcpy(tensor->data<int64_t>(), token_ids.data(),
                                token_ids.size() * sizeof(int64_t));
                    inputs[input_name] = tensor;

                    LOG_INFO("Tokenized input text: '", user_input_text, "'");
                    LOG_INFO("Token count: ", token_ids.size());
                    
                    // Print first few token IDs for debugging
                    std::cout << "Token IDs: [";
                    for (size_t i = 0; i < std::min(size_t(10), token_ids.size()); ++i) {
                        std::cout << token_ids[i];
                        if (i < std::min(size_t(10), token_ids.size()) - 1) std::cout << ", ";
                    }
                    if (token_ids.size() > 10) std::cout << ", ...";
                    std::cout << "]\n";

                } else if (input_name == "attention_mask" && !user_input_text.empty()) {
                    if (tokenizer_path.empty()) {
                        std::cerr << "Error: --tokenizer <path/to/tokenizer.model> must be provided\n";
                        return 1;
                    }
                    
                    auto token_ids = tokenizeText(user_input_text, tokenizer_path);
                    std::vector<int64_t> mask(token_ids.size(), 1);
                    shape = {1, static_cast<int64_t>(mask.size())};
                    auto tensor = std::make_shared<Tensor>(shape, DataType::INT64);
                    std::memcpy(tensor->data<int64_t>(), mask.data(),
                                mask.size() * sizeof(int64_t));
                    inputs[input_name] = tensor;

                } else {
                    // Fallback for non-text inputs
                    inputs[input_name] = createTestInput(shape);
                }
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

            // Create appropriate executor based on use_cpu flag
            std::shared_ptr<Executor> executor;
            if (use_cpu) {
                auto cpu_exec = std::make_shared<CpuExecutor>(cpu_threads > 0 ? cpu_threads : 1);
                cpu_exec->setVerbose(verbose);
                executor = cpu_exec;
            } else {
                auto gpu_exec = std::make_shared<GpuExecutor>(false);
                gpu_exec->setVerbose(verbose);
                // Enable GPU_PERSISTENT mode for minimal CPU-GPU transfers
                gpu_exec->setExecutionMode(GpuExecutor::ExecutionMode::GPU_PERSISTENT);
                LOG_INFO("GPU_PERSISTENT mode enabled - tensors stay on GPU");
                executor = gpu_exec;
            }

            auto exec_start = std::chrono::high_resolution_clock::now();

            outputs = executor->execute(*graph, inputs);

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
            printTensorSample("Output " + name, *tensor, 100);  // Print up to 100 values for testing
        }

        // Try to decode output if it looks like token IDs or logits
        if (!tokenizer_path.empty()) {
            std::vector<int64_t> token_ids;

            if (outputs.count("output_ids")) {
                // Direct token IDs output
                auto ids = outputs["output_ids"];
                const int64_t* data = ids->data<int64_t>();
                token_ids = std::vector<int64_t>(data, data + ids->size());
            } else if (outputs.count("logits")) {
                // Convert logits to token IDs (argmax along vocab dimension)
                auto logits = outputs["logits"];
                const float* logits_data = logits->data<float>();
                auto shape = logits->shape();

                if (shape.size() >= 2) {
                    // Shape is typically [batch_size, seq_len, vocab_size]
                    int seq_len = shape.size() == 3 ? shape[1] : 1;
                    int vocab_size = shape[shape.size() - 1];

                    LOG_INFO("Decoding logits: seq_len=", seq_len, ", vocab_size=", vocab_size);

                    for (int i = 0; i < seq_len; ++i) {
                        const float* seq_logits = logits_data + i * vocab_size;

                        // Find argmax
                        int max_idx = 0;
                        float max_val = seq_logits[0];
                        for (int j = 1; j < vocab_size; ++j) {
                            if (seq_logits[j] > max_val) {
                                max_val = seq_logits[j];
                                max_idx = j;
                            }
                        }
                        token_ids.push_back(max_idx);
                    }

                    // Print token IDs
                    std::cout << "Decoded Token IDs: [";
                    for (size_t i = 0; i < std::min(size_t(20), token_ids.size()); ++i) {
                        std::cout << token_ids[i];
                        if (i < std::min(size_t(20), token_ids.size()) - 1) std::cout << ", ";
                    }
                    if (token_ids.size() > 20) std::cout << ", ...";
                    std::cout << "]\n";
                }
            }

            // Decode token IDs to text
            if (!token_ids.empty()) {
                std::string decoded_text = decodeTokens(token_ids, tokenizer_path);

                if (!decoded_text.empty()) {
                    // Successful decode
                    LOG_INFO("\n=== Generated Text ===");
                    std::cout << decoded_text << "\n";
                } else {
                    // Decoding failed
                    LOG_WARN("\n[Tokenizer] Failed to decode output tokens");
                }
            }
        }

        LOG_INFO("\n=== Execution Successful ===");
        return 0;

    } catch (const std::exception& e) {
        LOG_ERROR("Error: ", e.what());
        return 1;
    }
}
