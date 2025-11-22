#include "autoregressive_generator.hpp"
#include "../utils/logger.hpp"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <cstdio>
#include <stdexcept>
#include <chrono>
#include <cstdlib>

namespace onnx_runner {

AutoregressiveGenerator::AutoregressiveGenerator(
    GpuExecutor& executor,
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

    // Log prompt token IDs
    std::ostringstream prompt_ids_str;
    prompt_ids_str << "[";
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        if (i > 0) prompt_ids_str << ", ";
        prompt_ids_str << prompt_tokens[i];
    }
    prompt_ids_str << "]";
    LOG_INFO("Prompt token IDs: ", prompt_ids_str.str());

    // Generate tokens
    auto generated_tokens = generateTokens(prompt_tokens);

    // Log full sequence token IDs
    std::ostringstream full_ids_str;
    full_ids_str << "[";
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        if (i > 0) full_ids_str << ", ";
        full_ids_str << generated_tokens[i];
    }
    full_ids_str << "]";
    LOG_INFO("Full sequence token IDs: ", full_ids_str.str());

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

    // KV-caching is handled by the ONNX model's explicit past_key_values inputs/outputs
    // (not the internal cache in gpu_executor)

    auto generation_start = std::chrono::high_resolution_clock::now();
    int tokens_generated = 0;
    bool is_prefill = true;  // First pass processes all prompt tokens
    std::map<std::string, std::shared_ptr<Tensor>> past_kv;  // Store KV-cache between steps

    const char* dump_env = std::getenv("ONNX_ENGINE_DUMP_KV");
    const bool dump_kv = dump_env && dump_env[0] != '\0' && dump_env[0] != '0';

    auto writeBinaryTensor = [&](const std::string& filename,
                                 const Tensor& tensor) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) return;
        out.write(reinterpret_cast<const char*>(tensor.data<float>()),
                  tensor.size() * sizeof(float));
        LOG_INFO("DEBUG wrote ", filename, " (", tensor.size(), " floats)");
    };

    // Generation loop
    for (int i = 0; i < config_.max_tokens; ++i) {
        try {
            int64_t past_length = inferPastLength(past_kv);
            if (!is_prefill && past_kv.empty()) {
                throw std::runtime_error("Decode step requested but no past_key_values tensors "
                                         "were captured from the previous iteration");
            }

            std::vector<int64_t> step_tokens;
            if (is_prefill) {
                step_tokens = current_tokens;
            } else {
                if (current_tokens.empty()) {
                    throw std::runtime_error("Decode step requested but no tokens are available");
                }
                step_tokens = {current_tokens.back()};
            }

            int64_t seq_len = static_cast<int64_t>(step_tokens.size());
            int64_t start_position = is_prefill ? 0 : past_length;
            int64_t total_length = start_position + seq_len;

            LOG_INFO("Step ", i + 1, (is_prefill ? " [PREFILL]" : " [DECODE]"),
                     ": past_length=", past_length,
                     ", input_seq_len=", seq_len,
                     ", total_seq=", total_length);


            // Create input tensors
            if (dump_kv && !past_kv.empty()) {
                auto it = past_kv.find("past_key_values.0.key");
                if (it != past_kv.end() && it->second) {
                    const float* data = it->second->data<float>();
                    std::ostringstream fname;
                    fname << "engine_input_step" << i << "_key.bin";
                    std::ofstream out(fname.str(), std::ios::binary);
                    if (out) {
                        out.write(reinterpret_cast<const char*>(data),
                                  it->second->size() * sizeof(float));
                        LOG_INFO("DEBUG wrote ", fname.str(), " (", it->second->size(), " floats)");
                    }
                }
            }

            // - Prefill: pass all prompt tokens with position_ids [0, 1, 2, ..., n-1], no past_kv
            // - Decode: pass only the last token with position_ids [current_pos], with past_kv
            auto inputs = createInputTensors(step_tokens, start_position, past_kv);

            if (config_.verbose) {
                if (is_prefill) {
                    LOG_INFO("PREFILL: processing ", seq_len, " prompt tokens");
                } else {
                    LOG_INFO("DECODE token @ position ", start_position);
                }
            }

            // Run forward pass
            auto step_start = std::chrono::high_resolution_clock::now();
            auto outputs = executor_.execute(graph_, inputs);
            auto step_end = std::chrono::high_resolution_clock::now();

            // Extract present.* outputs and store as past_kv for next iteration
            past_kv.clear();
            bool captured_kv = false;
            for (const auto& [name, tensor] : outputs) {
                if (name.find("present.") == 0) {
                    // Convert "present.X.key" -> "past_key_values.X.key"
                    std::string past_name = "past_key_values." + name.substr(8);  // Skip "present."
                    if (tensor->ndim() >= 3) {
                        int64_t cache_len = tensor->dim(2);
                        if (cache_len != total_length) {
                            LOG_WARN("KV-cache tensor ", name, " has length ", cache_len,
                                     " but expected ", total_length);
                        }
                    }
                    past_kv[past_name] = tensor;
                    captured_kv = true;
                    LOG_DEBUG("Captured KV-cache: ", past_name, " shape: ", tensor->shapeStr());

                    if (dump_kv) {
                        std::string sanitized = past_name;
                        std::replace(sanitized.begin(), sanitized.end(), '.', '_');
                        std::ostringstream fname;
                        fname << "engine_present_step" << i << "_" << sanitized << ".bin";
                        writeBinaryTensor(fname.str(), *tensor);
                    }
                } else if (dump_kv && name == "/model/layers.30/final_norm_layernorm/output_0") {
                    writeBinaryTensor("engine_hidden_step" + std::to_string(i) + ".bin", *tensor);
                }
            }
            if (!captured_kv) {
                LOG_WARN("Model did not emit any present.* tensors on step ", i + 1);
            } else {
                LOG_INFO("Stored ", past_kv.size(), " KV-cache tensors for next iteration");
            }

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
            logLogitStatistics(*logits, i + 1, past_length, total_length);
            if (dump_kv) {
                std::ostringstream logits_name;
                logits_name << "engine_logits_step" << (i + 1) << ".bin";
                std::ofstream logits_out(logits_name.str(), std::ios::binary);
                if (logits_out) {
                    logits_out.write(reinterpret_cast<const char*>(logits->data<float>()),
                                     logits->size() * sizeof(float));
                    LOG_INFO("DEBUG wrote ", logits_name.str(), " (", logits->size(), " floats)");
                }
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

            // Always log the generated token for debugging
            LOG_INFO("Generated token ", tokens_generated, ": ", next_token);

            // After first iteration, switch to decode mode (single token input)
            if (is_prefill) {
                is_prefill = false;
                if (config_.verbose) {
                    LOG_INFO("Switching to DECODE mode (KV cache active)");
                }
            }

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
    const float* logits_data = logits.data<float>();
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
AutoregressiveGenerator::createInputTensors(const std::vector<int64_t>& token_ids,
                                           int64_t start_position,
                                           const std::map<std::string, std::shared_ptr<Tensor>>& past_kv) {
    std::map<std::string, std::shared_ptr<Tensor>> inputs;

    int64_t seq_len = static_cast<int64_t>(token_ids.size());
    int64_t cached_past_len = inferPastLength(past_kv);

    // Validation
    if (seq_len <= 0) {
        throw std::runtime_error("Cannot create input tensors with zero or negative sequence length");
    }
    if (start_position + seq_len > 2048) {
        LOG_ERROR("Warning: sequence position ", start_position + seq_len,
                 " exceeds typical model max (2048)");
    }

    if (config_.verbose) {
        LOG_DEBUG("Creating input tensors for sequence length: ", seq_len,
                 ", start_position: ", start_position);
    }

    if (cached_past_len > 0 && seq_len != 1) {
        throw std::runtime_error("When KV-cache is populated (past_length=" +
                                 std::to_string(cached_past_len) +
                                 "), decode inputs must be exactly one token");
    }
    if (cached_past_len > 0 && start_position != cached_past_len) {
        throw std::runtime_error("start_position (" + std::to_string(start_position) +
                                 ") does not match cached past length (" +
                                 std::to_string(cached_past_len) + ")");
    }
    if (cached_past_len == 0 && !past_kv.empty()) {
        LOG_WARN("Received ", past_kv.size(),
                 " KV-cache tensors but inferred past length is zero");
    }
    if (cached_past_len == 0 && start_position != 0) {
        throw std::runtime_error("start_position must be zero when no KV-cache is present");
    }

    // Get list of graph inputs
    const auto& graph_inputs = graph_.inputs();

    // Handle KV-cache inputs
    int kv_count = 0;
    if (!past_kv.empty()) {
        // Decode mode: use existing KV-cache tensors
        for (const auto& [name, tensor] : past_kv) {
            if (std::find(graph_inputs.begin(), graph_inputs.end(), name) != graph_inputs.end()) {
                inputs[name] = tensor;
                kv_count++;
                if (config_.verbose && name == "past_key_values.0.key") {
                    LOG_INFO("Using past KV-cache: ", name, " shape: ", tensor->shapeStr());
                }
            }
        }
        if (config_.verbose && kv_count > 0) {
            LOG_INFO("Using ", kv_count, " past KV-cache tensors as input");
        }
    } else {
        // Prefill mode: create explicit zero-sized KV-cache tensors [batch, kv_heads, 0, head_dim]
        // This matches ONNX Runtime behavior and is required by many models
        for (const auto& input_name : graph_inputs) {
            if (input_name.find("past_key_values.") == 0) {
                // Get the expected shape from the graph
                auto shape = graph_.getInputShape(input_name);
                if (shape.size() == 4) {
                    // Expected shape: [batch, kv_heads, seq_len, head_dim]
                    // Note: Dynamic dimensions are stored as 1 by the parser, but we need 0 for empty cache
                    // Create zero-sized cache: [1, kv_heads, 0, head_dim]
                    int64_t kv_heads = shape[1];
                    int64_t head_dim = shape[3];

                    auto empty_cache = std::make_shared<Tensor>(
                        std::vector<int64_t>{1, kv_heads, 0, head_dim},
                        DataType::FLOAT32);

                    inputs[input_name] = empty_cache;
                    kv_count++;

                    if (config_.verbose && kv_count == 1) {
                        LOG_INFO("Creating zero-sized KV-cache tensors for prefill (batch=1, kv_heads=", kv_heads, ", seq=0, head_dim=", head_dim, ")");
                    }
                }
            }
        }
        if (config_.verbose && kv_count > 0) {
            LOG_INFO("Created ", kv_count, " empty KV-cache input tensors");
        }
    }

    // Create input_ids tensor [1, seq_len]
    if (std::find(graph_inputs.begin(), graph_inputs.end(), "input_ids") != graph_inputs.end()) {
        auto input_ids = std::make_shared<Tensor>(
            std::vector<int64_t>{1, seq_len}, DataType::INT64);
        std::memcpy(input_ids->data<int64_t>(), token_ids.data(),
                    token_ids.size() * sizeof(int64_t));
        inputs["input_ids"] = input_ids;
        if (config_.verbose) {
            LOG_DEBUG("Created input_ids with shape [1, ", seq_len, "]");
        }
    }

    // Create attention_mask tensor
    // For decode mode with KV-cache, mask should cover all tokens (past + current)
    if (std::find(graph_inputs.begin(), graph_inputs.end(), "attention_mask") != graph_inputs.end()) {
        int64_t total_len = cached_past_len + seq_len;  // Total sequence length including past
        auto attention_mask = std::make_shared<Tensor>(
            std::vector<int64_t>{1, total_len}, DataType::INT64);
        int64_t* mask_data = attention_mask->data<int64_t>();
        for (int64_t i = 0; i < total_len; ++i) {
            mask_data[i] = 1;
        }
        inputs["attention_mask"] = attention_mask;
        if (config_.verbose) {
            LOG_DEBUG("Created attention_mask with shape [1, ", total_len, "]");
        }
    }

    // Create position_ids tensor [1, seq_len]
    // For KV-cache decode mode: positions are [start_position, start_position+1, ...]
    // For prefill mode: positions are [0, 1, 2, ..., seq_len-1]
    if (std::find(graph_inputs.begin(), graph_inputs.end(), "position_ids") != graph_inputs.end()) {
        auto position_ids = std::make_shared<Tensor>(
            std::vector<int64_t>{1, seq_len}, DataType::INT64);
        int64_t* pos_data = position_ids->data<int64_t>();
        for (int64_t i = 0; i < seq_len; ++i) {
            pos_data[i] = start_position + i;
        }
        inputs["position_ids"] = position_ids;
        if (config_.verbose) {
            LOG_DEBUG("Created position_ids with shape [1, ", seq_len,
                     "], values: [", start_position, "..", start_position + seq_len - 1, "]");
        }
    }

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

        // Transfer logits to CPU if needed (for sampling)
        if (logits_tensor->device() == DeviceType::CUDA) {
            logits_tensor->toCPU();
        }

        // Extract logits for the last position
        // We want logits[0, seq_len-1, :] which is the prediction for the next token
        const float* all_logits = logits_tensor->data<float>();
        const float* last_position_logits = all_logits + (seq_len - 1) * vocab_size;

        // Create a new tensor for just the last position
        auto next_token_logits = std::make_shared<Tensor>(
            std::vector<int64_t>{vocab_size}, DataType::FLOAT32);
        std::memcpy(next_token_logits->data<float>(), last_position_logits,
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

int64_t AutoregressiveGenerator::inferPastLength(
    const std::map<std::string, std::shared_ptr<Tensor>>& past_kv) const {
    int64_t inferred = -1;
    std::string reference_name;
    for (const auto& [name, tensor] : past_kv) {
        if (!tensor || tensor->ndim() < 3) {
            continue;
        }
        int64_t length = tensor->dim(2);
        if (inferred < 0) {
            inferred = length;
            reference_name = name;
        } else if (inferred != length) {
            throw std::runtime_error(
                "Past KV tensor length mismatch between " + reference_name + " (" +
                std::to_string(inferred) + ") and " + name + " (" +
                std::to_string(length) + ")");
        }
    }
    return inferred < 0 ? 0 : inferred;
}

void AutoregressiveGenerator::logLogitStatistics(const Tensor& logits,
                                                 int step_index,
                                                 int64_t past_length,
                                                 int64_t total_length) const {
    const float* logits_data = logits.data<float>();
    const size_t vocab_size = logits.size();
    if (vocab_size == 0) {
        LOG_WARN("Empty logits tensor at step ", step_index);
        return;
    }

    float min_val = logits_data[0];
    float max_val = logits_data[0];
    double sum = logits_data[0];
    for (size_t idx = 1; idx < vocab_size; ++idx) {
        float value = logits_data[idx];
        min_val = std::min(min_val, value);
        max_val = std::max(max_val, value);
        sum += value;
    }
    double mean = sum / static_cast<double>(vocab_size);

    constexpr size_t kTopK = 5;
    std::array<float, kTopK> top_vals;
    std::array<int64_t, kTopK> top_ids;
    top_vals.fill(-std::numeric_limits<float>::infinity());
    top_ids.fill(-1);

    for (int64_t token = 0; token < static_cast<int64_t>(vocab_size); ++token) {
        float value = logits_data[token];
        for (size_t pos = 0; pos < kTopK; ++pos) {
            if (value > top_vals[pos]) {
                for (size_t shift = kTopK - 1; shift > pos; --shift) {
                    top_vals[shift] = top_vals[shift - 1];
                    top_ids[shift] = top_ids[shift - 1];
                }
                top_vals[pos] = value;
                top_ids[pos] = token;
                break;
            }
        }
    }

    size_t top_count = std::min(kTopK, vocab_size);
    std::ostringstream top_stream;
    top_stream << "[";
    for (size_t idx = 0; idx < top_count; ++idx) {
        if (idx > 0) top_stream << ", ";
        top_stream << top_ids[idx] << ":" << top_vals[idx];
    }
    top_stream << "]";

    LOG_INFO("Logits stats (step ", step_index,
             ", past_len=", past_length,
             ", total_len=", total_length,
             "): min=", min_val,
             ", max=", max_val,
             ", mean=", mean,
             ", top", top_count, "=", top_stream.str());
}

} // namespace onnx_runner
