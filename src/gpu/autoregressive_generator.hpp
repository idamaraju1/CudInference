#pragma once

#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include "gpu_executor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <random>

namespace onnx_runner {

/**
 * AutoregressiveGenerator handles iterative token generation for LLM inference.
 * It manages the generation loop, sampling strategies, and stopping criteria.
 */
class AutoregressiveGenerator {
public:
    /**
     * Configuration for generation
     */
    struct GenerationConfig {
        int max_tokens = 50;          // Maximum number of tokens to generate
        float temperature = 1.0f;     // Temperature for sampling (0.0 = greedy)
        int eos_token_id = -1;        // End-of-sequence token ID (-1 = auto-detect from tokenizer)
        bool verbose = false;         // Print generation progress

        GenerationConfig() = default;
    };

    /**
     * Constructor
     * @param executor Reference to GpuExecutor for running forward passes
     * @param graph Reference to the model graph
     * @param tokenizer_path Path to tokenizer file for encoding/decoding
     * @param config Generation configuration
     */
    AutoregressiveGenerator(GpuExecutor& executor,
                           const Graph& graph,
                           const std::string& tokenizer_path,
                           const GenerationConfig& config);

    /**
     * Generate text from a prompt using autoregressive sampling
     * @param prompt_text Input text prompt
     * @return Generated text (including prompt)
     */
    std::string generate(const std::string& prompt_text);

    /**
     * Generate tokens from initial token IDs
     * @param prompt_token_ids Initial sequence of token IDs
     * @return Complete sequence of token IDs (including prompt)
     */
    std::vector<int64_t> generateTokens(const std::vector<int64_t>& prompt_token_ids);

private:
    GpuExecutor& executor_;
    const Graph& graph_;
    std::string tokenizer_path_;
    GenerationConfig config_;
    std::mt19937 rng_;  // Random number generator for sampling

    /**
     * Tokenize text to token IDs using external Python script
     * @param text Input text
     * @return Vector of token IDs
     */
    std::vector<int64_t> tokenize(const std::string& text);

    /**
     * Decode token IDs back to text using external Python script
     * @param token_ids Vector of token IDs
     * @return Decoded text
     */
    std::string decode(const std::vector<int64_t>& token_ids);

    /**
     * Sample next token from logits tensor
     * @param logits Tensor containing logits for next token [vocab_size]
     * @return Sampled token ID
     */
    int64_t sampleNextToken(const Tensor& logits);

    /**
     * Apply temperature scaling to logits and sample
     * @param logits_data Raw logits data
     * @param vocab_size Size of vocabulary
     * @param temperature Temperature value (0.0 = greedy, higher = more random)
     * @return Sampled token ID
     */
    int64_t sampleWithTemperature(const float* logits_data,
                                   int vocab_size,
                                   float temperature);

    /**
     * Best-effort detection of EOS token id from tokenizer.json. Returns -1 on failure.
     */
    int detectEosTokenId(const std::string& tokenizer_path);

    /**
     * Execute a shell command and capture output
     * @param cmd Command to execute
     * @return Command output as string
     */
    std::string execCommand(const std::string& cmd);

    /**
     * Find the tokenizer script path regardless of working directory
     * @return Path to hf_tokenizer.py script
     */
    std::string findTokenizerScript();

    /**
     * Create input tensors for the model from token IDs
     * @param token_ids Current sequence of token IDs
     * @param start_position Starting position for position_ids (for KV cache decode)
     * @param past_kv Past key-value cache tensors (for decode mode)
     * @return Map of input name to tensor (input_ids, attention_mask, etc.)
     */
    std::map<std::string, std::shared_ptr<Tensor>>
    createInputTensors(const std::vector<int64_t>& token_ids,
                      int64_t start_position,
                      const std::map<std::string, std::shared_ptr<Tensor>>& past_kv);

    /**
     * Extract logits for the last position from model output
     * @param outputs Model output tensors
     * @return Tensor containing logits for next token prediction
     */
    std::shared_ptr<Tensor> extractNextTokenLogits(
        const std::map<std::string, std::shared_ptr<Tensor>>& outputs);

    /**
     * Inspect cached KV tensors to infer the currently cached sequence length.
     * Throws if different layers disagree to avoid silent cache corruption.
     */
    int64_t inferPastLength(const std::map<std::string, std::shared_ptr<Tensor>>& past_kv) const;

    /**
     * Emit detailed statistics about the logits distribution so we can spot
     * pathological domination (e.g., repetitive "the the the").
     */
    void logLogitStatistics(const Tensor& logits,
                            int step_index,
                            int64_t past_length,
                            int64_t total_length) const;
};

} // namespace onnx_runner
