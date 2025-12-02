#include "group_query_attention_operation.hpp"
#include "operation_registry.hpp"
#include "operation_utils.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"
#include <cmath>

namespace onnx_runner {
using namespace operation_utils;

void GroupQueryAttentionOperation::execute(const Node& node, ExecutionContext& ctx) {
    const auto& inputs = node.inputs();
    const auto& outputs = node.outputs();

    if (inputs.size() < 3 || outputs.size() < 3) {
        throw std::runtime_error("GroupQueryAttention expects at least Q, K, V inputs and 3 outputs (attention, present_key, present_value)");
    }

    auto Q = getTensor(inputs[0], ctx);
    auto K = getTensor(inputs[1], ctx);
    auto V = getTensor(inputs[2], ctx);

    if (Q->dtype() != DataType::FLOAT32 || K->dtype() != DataType::FLOAT32 || V->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("GroupQueryAttention currently supports FLOAT32 tensors only");
    }

    if (Q->ndim() != 3 || K->ndim() != 3 || V->ndim() != 3) {
        throw std::runtime_error("GroupQueryAttention expects 3D Q/K/V tensors [batch, seq, hidden]");
    }

    int64_t q_heads_attr = node.getIntAttr("num_heads", 0);
    int64_t kv_heads_attr = node.getIntAttr("kv_num_heads", 0);
    if (q_heads_attr <= 0 || kv_heads_attr <= 0) {
        throw std::runtime_error("GroupQueryAttention requires positive num_heads and kv_num_heads attributes");
    }

    size_t batch = static_cast<size_t>(Q->dim(0));
    size_t q_seq = static_cast<size_t>(Q->dim(1));
    size_t q_hidden = static_cast<size_t>(Q->dim(2));
    size_t q_heads = static_cast<size_t>(q_heads_attr);
    size_t head_dim = q_hidden / q_heads;

    size_t kv_seq = static_cast<size_t>(K->dim(1));
    size_t k_hidden = static_cast<size_t>(K->dim(2));
    size_t kv_heads = static_cast<size_t>(kv_heads_attr);

    // Check for past KV-cache inputs (indices 3 and 4)
    size_t past_len = 0;
    std::shared_ptr<Tensor> past_key, past_value;
    if (inputs.size() > 3 && !inputs[3].empty() && hasTensor(inputs[3], ctx)) {
        past_key = getTensor(inputs[3], ctx);
        if (past_key->ndim() == 4 && past_key->dim(2) > 0) {
            past_len = static_cast<size_t>(past_key->dim(2));
        }
    }
    if (inputs.size() > 4 && !inputs[4].empty() && hasTensor(inputs[4], ctx)) {
        past_value = getTensor(inputs[4], ctx);
    }

    size_t total_seq = past_len + kv_seq;

    float scale_attr = node.getFloatAttr("scale", 0.0f);
    float scale = scale_attr != 0.0f ? scale_attr : (1.0f / std::sqrt(static_cast<float>(head_dim)));
    float softcap = node.getFloatAttr("softcap", 0.0f);

    std::vector<size_t> valid_lengths(batch, total_seq);

    std::vector<uint8_t> q_cache, k_cache, v_cache;
    const float* q_data = getHostData<float>(Q, q_cache);
    const float* k_data = getHostData<float>(K, k_cache);
    const float* v_data = getHostData<float>(V, v_cache);

    size_t key_storage_elems = batch * kv_heads * total_seq * head_dim;
    size_t value_storage_elems = batch * kv_heads * total_seq * head_dim;
    std::vector<float> key_storage(key_storage_elems, 0.f);
    std::vector<float> value_storage(value_storage_elems, 0.f);

    // Copy past KV-cache if it exists
    if (past_key && past_value && past_len > 0) {
        std::vector<uint8_t> past_k_cache, past_v_cache;
        const float* past_k_data = getHostData<float>(past_key, past_k_cache);
        const float* past_v_data = getHostData<float>(past_value, past_v_cache);

        // Past cache is already in [batch, kv_heads, past_len, head_dim] format
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < kv_heads; ++h) {
                const float* src_k = past_k_data + ((b * kv_heads + h) * past_len) * head_dim;
                const float* src_v = past_v_data + ((b * kv_heads + h) * past_len) * head_dim;
                float* dst_k = key_storage.data() + ((b * kv_heads + h) * total_seq) * head_dim;
                float* dst_v = value_storage.data() + ((b * kv_heads + h) * total_seq) * head_dim;
                std::memcpy(dst_k, src_k, past_len * head_dim * sizeof(float));
                std::memcpy(dst_v, src_v, past_len * head_dim * sizeof(float));
            }
        }
    }

    // Reformat new K/V from [batch, seq, hidden] to [batch, kv_heads, seq, head_dim]
    // Append to the end of the storage (after past_len)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t seq = 0; seq < kv_seq; ++seq) {
            for (size_t h = 0; h < kv_heads; ++h) {
                const float* src_k = k_data + ((b * kv_seq + seq) * k_hidden) + h * head_dim;
                const float* src_v = v_data + ((b * kv_seq + seq) * k_hidden) + h * head_dim;
                float* dst_k = key_storage.data() + (((b * kv_heads + h) * total_seq) + past_len + seq) * head_dim;
                float* dst_v = value_storage.data() + (((b * kv_heads + h) * total_seq) + past_len + seq) * head_dim;
                std::memcpy(dst_k, src_k, head_dim * sizeof(float));
                std::memcpy(dst_v, src_v, head_dim * sizeof(float));
            }
        }
    }

    auto output = allocateOutput(Q->shape(), ctx, DataType::FLOAT32);

    // Execute based on mode
    if (ctx.use_cpu) {
        // Pure CPU execution
        launchGroupQueryAttention(
            q_data, key_storage.data(), value_storage.data(), output->data<float>(),
            static_cast<int>(batch), static_cast<int>(q_seq), static_cast<int>(q_heads),
            static_cast<int>(kv_heads), static_cast<int>(head_dim), static_cast<int>(total_seq),
            static_cast<int>(past_len), scale, softcap, valid_lengths,
            true, ctx.num_cpu_threads
        );
    } else {
        // GPU execution - copy prepared data to GPU
        float *d_q, *d_k, *d_v, *d_output;
        size_t q_bytes = Q->size() * sizeof(float);
        size_t k_bytes = key_storage.size() * sizeof(float);
        size_t v_bytes = value_storage.size() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_q, q_bytes));
        CUDA_CHECK(cudaMalloc(&d_k, k_bytes));
        CUDA_CHECK(cudaMalloc(&d_v, v_bytes));

        if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
            d_output = output->mutableDeviceData<float>();
        } else {
            CUDA_CHECK(cudaMalloc(&d_output, q_bytes));
        }

        CUDA_CHECK(cudaMemcpy(d_q, q_data, q_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k, key_storage.data(), k_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, value_storage.data(), v_bytes, cudaMemcpyHostToDevice));

        launchGroupQueryAttention(
            d_q, d_k, d_v, d_output,
            static_cast<int>(batch), static_cast<int>(q_seq), static_cast<int>(q_heads),
            static_cast<int>(kv_heads), static_cast<int>(head_dim), static_cast<int>(total_seq),
            static_cast<int>(past_len), scale, softcap, valid_lengths,
            false, ctx.num_cpu_threads
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        if (ctx.gpu_mode != ExecutionContext::GPUMode::PERSISTENT) {
            CUDA_CHECK(cudaMemcpy(output->data<float>(), d_output, q_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_output));
        }

        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_k));
        CUDA_CHECK(cudaFree(d_v));
    }

    // Store attention output
    storeOutput(outputs[0], output, ctx);

    // Store present key/value caches (from reformatted K/V storage)
    // Shape: [batch, kv_heads, total_seq, head_dim]
    std::vector<int64_t> kv_cache_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(kv_heads),
        static_cast<int64_t>(total_seq),
        static_cast<int64_t>(head_dim)
    };

    auto present_key = std::make_shared<Tensor>(kv_cache_shape, DataType::FLOAT32);
    auto present_value = std::make_shared<Tensor>(kv_cache_shape, DataType::FLOAT32);

    // Copy the reformatted key/value storage to the present tensors
    std::memcpy(present_key->data<float>(), key_storage.data(), key_storage.size() * sizeof(float));
    std::memcpy(present_value->data<float>(), value_storage.data(), value_storage.size() * sizeof(float));

    // Transfer to GPU if in PERSISTENT mode
    if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
        present_key->toGPU();
        present_value->toGPU();
    }

    storeOutput(outputs[1], present_key, ctx);
    storeOutput(outputs[2], present_value, ctx);
}

REGISTER_OPERATION(OpType::GROUPQUERYATTENTION, GroupQueryAttentionOperation)
} // namespace onnx_runner
