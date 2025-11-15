#include "autoregressive_generator.hpp"
#include "../utils/logger.hpp"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <chrono>
#include <cstring>

namespace onnx_runner {

AutoregressiveGenerator::AutoregressiveGenerator(
    Executor& executor,
    const Graph& graph,
    const std::string& tokenizer_path,
    const GenerationConfig& config)
    : executor_(executor),
      graph_(graph),
      tokenizer_path_(tokenizer_path),
      config_(config),
      rng_(std::random_device{}()) {

    if (tokenizer_path_.empty()) {
        throw std::runtime_error("Tokenizer path cannot be empty");
    }

    LOG_INFO("AutoregressiveGenerator initialized");
    LOG_INFO("  Max tokens: ", config_.max_tokens);
    LOG_INFO("  Temperature: ", config_.temperature);
    LOG_INFO("  EOS token ID: ", config_.eos_token_id);
}

std::string AutoregressiveGenerator::generate(const std::string& prompt_text) {
    LOG_INFO("\n=== Starting Autoregressive Generation ===");
    LOG_INFO("Prompt: \"", prompt_text, "\"");

    // Tokenize the prompt
    auto prompt_tokens = tokenize(prompt_text);
    LOG_INFO("Prompt tokens: ", prompt_tokens.size());

    // Generate tokens
    auto generated_tokens = generateTokens(prompt_tokens);

    // Decode back to text
    std::string generated_text;
    try {
        generated_text = decode(generated_tokens);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to decode generated tokens: ", e.what());
        generated_text = "[Decoding failed]";
    }

    LOG_INFO("\n=== Generation Complete ===");
    LOG_INFO("Total tokens generated: ", generated_tokens.size() - prompt_tokens.size());
    LOG_INFO("Total sequence length: ", generated_tokens.size());

    return generated_text;
}

std::vector<int64_t> AutoregressiveGenerator::generateTokens(
    const std::vector<int64_t>& prompt_token_ids) {

    if (prompt_token_ids.empty()) {
        throw std::runtime_error("Prompt token IDs cannot be empty");
    }

    // Start with the prompt tokens
    std::vector<int64_t> current_tokens = prompt_token_ids;

    LOG_INFO("Starting generation with ", current_tokens.size(), " prompt tokens");

    auto generation_start = std::chrono::high_resolution_clock::now();
    int tokens_generated = 0;

    // Generation loop
    for (int i = 0; i < config_.max_tokens; ++i) {
        try {
            // Create input tensors for current sequence
            auto inputs = createInputTensors(current_tokens);

            if (config_.verbose) {
                LOG_INFO("Step ", i + 1, ": sequence length = ", current_tokens.size());
            }

            // Run forward pass
            auto step_start = std::chrono::high_resolution_clock::now();
            auto outputs = executor_.execute(graph_, inputs);
            auto step_end = std::chrono::high_resolution_clock::now();

            if (config_.verbose) {
                auto step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    step_end - step_start).count();
                LOG_INFO("Generation step ", i + 1, " took ", step_ms, " ms");
            }

            // Extract logits for next token
            auto logits = extractNextTokenLogits(outputs);
            if (!logits) {
                LOG_ERROR("Failed to extract logits from model output");
                break;
            }

            // Sample next token
            int64_t next_token = sampleNextToken(*logits);

            if (config_.verbose) {
                LOG_INFO("Sampled token: ", next_token);
            }

            // Sanity check on token ID
            if (next_token < 0) {
                LOG_ERROR("Invalid token ID: ", next_token);
                break;
            }

            // Append to sequence
            current_tokens.push_back(next_token);
            tokens_generated++;

            // Check for EOS token
            if (next_token == config_.eos_token_id) {
                LOG_INFO("EOS token detected, stopping generation");
                break;
            }

            // Optional: Print progress
            if (config_.verbose && (i + 1) % 10 == 0) {
                LOG_INFO("Progress: ", i + 1, "/", config_.max_tokens, " tokens generated");
            }

        } catch (const std::exception& e) {
            LOG_ERROR("Error during generation step ", i + 1, ": ", e.what());
            break;
        } catch (...) {
            LOG_ERROR("Unknown error during generation step ", i + 1);
            break;
        }
    }

    auto generation_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        generation_end - generation_start).count();

    // Calculate tokens per second
    double tokens_per_sec = 0.0;
    if (total_ms > 0) {
        tokens_per_sec = (tokens_generated * 1000.0) / total_ms;
    }

    LOG_INFO("\n=== Generation Statistics ===");
    LOG_INFO("Tokens generated: ", tokens_generated);
    LOG_INFO("Total time: ", total_ms, " ms");
    LOG_INFO("Speed: ", tokens_per_sec, " tokens/sec");

    return current_tokens;
}

std::vector<int64_t> AutoregressiveGenerator::tokenize(const std::string& text) {
    // Escape single quotes in text for shell command
    std::string escaped_text = text;
    size_t pos = 0;
    while ((pos = escaped_text.find("'", pos)) != std::string::npos) {
        escaped_text.replace(pos, 1, "'\\''");
        pos += 4;
    }

    // Call Python tokenizer script
    // Note: Assumes execution from build/ directory, so script is at ../scripts/
    std::string cmd = "python3 ../scripts/hf_tokenizer.py --tokenizer '" + tokenizer_path_ +
                     "' --encode '" + escaped_text + "' 2>&1";

    std::string output = execCommand(cmd);

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
        throw std::runtime_error("Tokenization produced no tokens");
    }

    return token_ids;
}

std::string AutoregressiveGenerator::decode(const std::vector<int64_t>& token_ids) {
    // Convert token IDs to space-separated string
    std::ostringstream ids_stream;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (i > 0) ids_stream << " ";
        ids_stream << token_ids[i];
    }
    std::string ids_str = ids_stream.str();

    // Call Python tokenizer script
    // Note: Assumes execution from build/ directory, so script is at ../scripts/
    std::string cmd = "python3 ../scripts/hf_tokenizer.py --tokenizer '" + tokenizer_path_ +
                     "' --decode '" + ids_str + "' 2>&1";

    std::string output = execCommand(cmd);

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

int64_t AutoregressiveGenerator::sampleNextToken(const Tensor& logits) {
    const float* logits_data = logits.data_ptr<float>();
    int vocab_size = static_cast<int>(logits.size());

    if (vocab_size == 0) {
        throw std::runtime_error("Empty logits tensor");
    }

    return sampleWithTemperature(logits_data, vocab_size, config_.temperature);
}

int64_t AutoregressiveGenerator::sampleWithTemperature(
    const float* logits_data,
    int vocab_size,
    float temperature) {

    // Greedy decoding (temperature = 0 or very close to 0)
    if (temperature < 1e-6f) {
        // Find argmax
        int max_idx = 0;
        float max_val = logits_data[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits_data[i] > max_val) {
                max_val = logits_data[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // Temperature sampling
    // 1. Scale logits by temperature
    std::vector<float> scaled_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        scaled_logits[i] = logits_data[i] / temperature;
    }

    // 2. Compute softmax to get probabilities
    // First, find max for numerical stability
    float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());

    // Compute exp and sum
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(scaled_logits[i] - max_logit);
        sum += probs[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    // 3. Sample from the distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand_val = dist(rng_);

    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (rand_val < cumsum) {
            return i;
        }
    }

    // Fallback (should rarely happen due to floating point precision)
    return vocab_size - 1;
}

std::string AutoregressiveGenerator::execCommand(const std::string& cmd) {
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

std::map<std::string, std::shared_ptr<Tensor>>
AutoregressiveGenerator::createInputTensors(const std::vector<int64_t>& token_ids) {
    std::map<std::string, std::shared_ptr<Tensor>> inputs;

    int64_t seq_len = static_cast<int64_t>(token_ids.size());

    // Validation
    if (seq_len <= 0) {
        throw std::runtime_error("Cannot create input tensors with zero or negative sequence length");
    }
    if (seq_len > 2048) {
        LOG_ERROR("Warning: sequence length ", seq_len, " exceeds typical model max (2048)");
    }

    if (config_.verbose) {
        LOG_DEBUG("Creating input tensors for sequence length: ", seq_len);
    }

    // Get list of graph inputs
    const auto& graph_inputs = graph_.inputs();

    // Create input_ids tensor [1, seq_len]
    if (std::find(graph_inputs.begin(), graph_inputs.end(), "input_ids") != graph_inputs.end()) {
        auto input_ids = std::make_shared<CpuTensor>(
            std::vector<int64_t>{1, seq_len}, DataType::INT64);
        std::memcpy(input_ids->data_ptr<int64_t>(), token_ids.data(),
                    token_ids.size() * sizeof(int64_t));
        inputs["input_ids"] = input_ids;
        if (config_.verbose) {
            LOG_DEBUG("Created input_ids with shape [1, ", seq_len, "]");
        }
    }

    // Create attention_mask tensor [1, seq_len] - all ones
    if (std::find(graph_inputs.begin(), graph_inputs.end(), "attention_mask") != graph_inputs.end()) {
        auto attention_mask = std::make_shared<CpuTensor>(
            std::vector<int64_t>{1, seq_len}, DataType::INT64);
        int64_t* mask_data = attention_mask->data_ptr<int64_t>();
        for (int64_t i = 0; i < seq_len; ++i) {
            mask_data[i] = 1;
        }
        inputs["attention_mask"] = attention_mask;
        if (config_.verbose) {
            LOG_DEBUG("Created attention_mask with shape [1, ", seq_len, "]");
        }
    }

    // Create position_ids tensor [1, seq_len] - sequential positions [0, 1, 2, ..., seq_len-1]
    if (std::find(graph_inputs.begin(), graph_inputs.end(), "position_ids") != graph_inputs.end()) {
        auto position_ids = std::make_shared<CpuTensor>(
            std::vector<int64_t>{1, seq_len}, DataType::INT64);
        int64_t* pos_data = position_ids->data_ptr<int64_t>();
        for (int64_t i = 0; i < seq_len; ++i) {
            pos_data[i] = i;
        }
        inputs["position_ids"] = position_ids;
        if (config_.verbose) {
            LOG_DEBUG("Created position_ids with shape [1, ", seq_len, "], values: [0..", seq_len - 1, "]");
        }
    }

    // Note: We intentionally DON'T create past_key_values.*.key/value inputs
    // These are optional inputs for KV-cache optimization. Since we're not using
    // KV-cache, we omit them entirely and the model will compute attention from scratch
    // each time (slower but simpler).

    return inputs;
}

std::shared_ptr<Tensor> AutoregressiveGenerator::extractNextTokenLogits(
    const std::map<std::string, std::shared_ptr<Tensor>>& outputs) {

    // Look for "logits" output (standard for LLMs)
    if (outputs.count("logits")) {
        auto logits_tensor = outputs.at("logits");
        auto shape = logits_tensor->shape();

        // Expected shape: [batch_size, seq_len, vocab_size]
        if (shape.size() != 3) {
            LOG_ERROR("Unexpected logits shape: expected 3D, got ", shape.size(), "D");
            return nullptr;
        }

        // int batch_size = shape[0];  // Currently unused, batch size is always 1
        int seq_len = shape[1];
        int vocab_size = shape[2];

        // Extract logits for the last position
        // We want logits[0, seq_len-1, :] which is the prediction for the next token
        const float* all_logits = logits_tensor->data_ptr<float>();
        const float* last_position_logits = all_logits + (seq_len - 1) * vocab_size;

        // Create a new tensor for just the last position
        auto next_token_logits = std::make_shared<CpuTensor>(
            std::vector<int64_t>{vocab_size}, DataType::FLOAT32);
        std::memcpy(next_token_logits->data_ptr<float>(), last_position_logits,
                    vocab_size * sizeof(float));

        return next_token_logits;
    }

    // Check for other possible output names
    if (outputs.count("output_ids")) {
        LOG_ERROR("Model outputs 'output_ids' directly - not suitable for autoregressive generation");
        return nullptr;
    }

    LOG_ERROR("Could not find 'logits' output in model outputs");
    return nullptr;
}

} // namespace onnx_runner
