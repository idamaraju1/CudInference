#include "rotary_embedding_operation.hpp"
#include "operation_registry.hpp"
#include "operation_utils.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"

namespace onnx_runner {
using namespace operation_utils;

void RotaryEmbeddingOperation::execute(const Node& node, ExecutionContext& ctx) {
    const auto& inputs = node.inputs();
    const auto& outputs = node.outputs();

    if (inputs.size() < 3 || outputs.empty()) {
        throw std::runtime_error("RotaryEmbedding expects at least 3 inputs and 1 output");
    }

    auto data_tensor = getTensor(inputs[0], ctx);
    if (data_tensor->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("RotaryEmbedding currently supports FLOAT32 only");
    }

    std::shared_ptr<Tensor> position_tensor, cos_tensor, sin_tensor;

    if (inputs.size() == 4) {
        position_tensor = getTensor(inputs[1], ctx);
        cos_tensor = getTensor(inputs[2], ctx);
        sin_tensor = getTensor(inputs[3], ctx);
    } else if (inputs.size() == 3) {
        cos_tensor = getTensor(inputs[1], ctx);
        sin_tensor = getTensor(inputs[2], ctx);
    }

    if (!cos_tensor || !sin_tensor) {
        throw std::runtime_error("RotaryEmbedding requires cos and sin cache tensors");
    }

    const auto& data_shape = data_tensor->shape();
    if (data_shape.size() < 3 || data_shape.size() > 4) {
        throw std::runtime_error("RotaryEmbedding input rank must be 3 or 4");
    }

    size_t batch = static_cast<size_t>(data_shape[0]);
    size_t sequence_length = static_cast<size_t>(data_shape.size() == 4 ? data_shape[2] : data_shape[1]);
    size_t hidden_total = static_cast<size_t>(data_shape.back());

    size_t num_heads, head_size;
    int64_t num_heads_attr = node.getIntAttr("num_heads", 0);

    if (data_shape.size() == 4) {
        num_heads = static_cast<size_t>(data_shape[1]);
        head_size = static_cast<size_t>(data_shape[3]);
    } else {
        if (num_heads_attr > 0) {
            num_heads = static_cast<size_t>(num_heads_attr);
            head_size = hidden_total / num_heads;
        } else {
            head_size = static_cast<size_t>(cos_tensor->shape().back()) * 2;
            num_heads = hidden_total / head_size;
        }
    }

    int64_t rotary_dim_attr = node.getIntAttr("rotary_embedding_dim", static_cast<int64_t>(head_size));
    if (rotary_dim_attr == 0) rotary_dim_attr = static_cast<int64_t>(head_size);
    size_t rotary_dim = static_cast<size_t>(rotary_dim_attr);
    size_t rotary_half = rotary_dim / 2;
    bool interleaved = node.getIntAttr("interleaved", 0) != 0;

    auto output = allocateOutput(data_tensor->shape(), ctx, DataType::FLOAT32);

    // CPU execution or operations that need CPU data
    // For now, always use CPU path since we need to select cos/sin values and potentially reorder
    // TODO: Implement full GPU version with cos/sin selection on GPU

    // Prepare position data
    std::vector<uint8_t> position_cache;
    std::vector<int64_t> position_converted;
    const int64_t* position_data = nullptr;

    if (position_tensor) {
        switch (position_tensor->dtype()) {
            case DataType::INT64:
                position_data = getHostData<int64_t>(position_tensor, position_cache);
                break;
            case DataType::INT32: {
                const int32_t* src = getHostData<int32_t>(position_tensor, position_cache);
                position_converted.resize(position_tensor->size());
                for (size_t i = 0; i < position_converted.size(); ++i) {
                    position_converted[i] = static_cast<int64_t>(src[i]);
                }
                position_data = position_converted.data();
                break;
            }
            default:
                throw std::runtime_error("RotaryEmbedding: unsupported position_ids dtype");
        }
    }

    // Select appropriate cos/sin values
    std::vector<uint8_t> cos_cache, sin_cache;
    const float* cos_data = getHostData<float>(cos_tensor, cos_cache);
    const float* sin_data = getHostData<float>(sin_tensor, sin_cache);

    std::vector<float> selected_cos(batch * sequence_length * rotary_half);
    std::vector<float> selected_sin(batch * sequence_length * rotary_half);

    const auto& cos_shape = cos_tensor->shape();
    if (cos_shape.size() == 2) {
        size_t max_positions = static_cast<size_t>(cos_shape[0]);
        size_t dim = static_cast<size_t>(cos_shape[1]);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < sequence_length; ++s) {
                int64_t pos = position_data ? position_data[s] : static_cast<int64_t>(s);
                if (pos < 0) pos += max_positions;
                pos = std::max(std::min(pos, static_cast<int64_t>(max_positions - 1)), static_cast<int64_t>(0));
                size_t src_offset = static_cast<size_t>(pos) * dim;
                size_t dst_offset = (b * sequence_length + s) * rotary_half;
                std::memcpy(&selected_cos[dst_offset], &cos_data[src_offset], rotary_half * sizeof(float));
                std::memcpy(&selected_sin[dst_offset], &sin_data[src_offset], rotary_half * sizeof(float));
            }
        }
    }

    // Convert input layout if needed: BNSH -> BSNH
    std::vector<uint8_t> data_cache;
    const float* input_data = getHostData<float>(data_tensor, data_cache);
    std::vector<float> reordered_input;

    if (data_shape.size() == 4) {
        reordered_input.resize(data_tensor->size());
        for (size_t b = 0; b < batch; ++b) {
            for (size_t n = 0; n < num_heads; ++n) {
                for (size_t s = 0; s < sequence_length; ++s) {
                    size_t src_idx = ((b * num_heads + n) * sequence_length + s) * head_size;
                    size_t dst_idx = ((b * sequence_length + s) * num_heads + n) * head_size;
                    std::memcpy(&reordered_input[dst_idx], &input_data[src_idx], head_size * sizeof(float));
                }
            }
        }
        input_data = reordered_input.data();
    }

    int batch_seq = static_cast<int>(batch * sequence_length);

    // Execute based on mode
    if (ctx.use_cpu) {
        // Pure CPU execution
        launchRotaryEmbedding(
            input_data, selected_cos.data(), selected_sin.data(), output->data<float>(),
            batch_seq, static_cast<int>(num_heads), static_cast<int>(head_size),
            static_cast<int>(rotary_dim), interleaved, true, ctx.num_cpu_threads
        );
    } else {
        // GPU execution - need to copy prepared data to GPU
        float *d_input, *d_cos, *d_sin, *d_output;
        size_t input_bytes = data_tensor->size() * sizeof(float);
        size_t cossin_bytes = batch * sequence_length * rotary_half * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
        CUDA_CHECK(cudaMalloc(&d_cos, cossin_bytes));
        CUDA_CHECK(cudaMalloc(&d_sin, cossin_bytes));

        if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
            d_output = output->mutableDeviceData<float>();
        } else {
            CUDA_CHECK(cudaMalloc(&d_output, input_bytes));
        }

        CUDA_CHECK(cudaMemcpy(d_input, input_data, input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, selected_cos.data(), cossin_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, selected_sin.data(), cossin_bytes, cudaMemcpyHostToDevice));

        launchRotaryEmbedding(
            d_input, d_cos, d_sin, d_output,
            batch_seq, static_cast<int>(num_heads), static_cast<int>(head_size),
            static_cast<int>(rotary_dim), interleaved, false, ctx.num_cpu_threads
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        if (ctx.gpu_mode != ExecutionContext::GPUMode::PERSISTENT) {
            // Copy result back to CPU
            CUDA_CHECK(cudaMemcpy(output->data<float>(), d_output, input_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_output));
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_cos));
        CUDA_CHECK(cudaFree(d_sin));
    }

    storeOutput(outputs[0], output, ctx);
}

REGISTER_OPERATION(OpType::ROTARYEMBEDDING, RotaryEmbeddingOperation)
} // namespace onnx_runner
