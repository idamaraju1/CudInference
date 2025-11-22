void GpuExecutor::executeGroupQueryAttention(const Node& node) {
    const auto& inputs = node.inputs();
    const auto& outputs = node.outputs();

    if (inputs.size() < 3 || outputs.empty()) {
        throw std::runtime_error("GroupQueryAttention expects at least Q, K, V and one output");
    }

    auto Q = getTensor(inputs[0]);
    auto K = getTensor(inputs[1]);
    auto V = getTensor(inputs[2]);

    if (Q->dtype() != DataType::FLOAT32 || K->dtype() != DataType::FLOAT32 ||
        V->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("GroupQueryAttention currently supports FLOAT32 tensors only");
    }

    if (Q->ndim() != 3 || K->ndim() != 3 || V->ndim() != 3) {
        throw std::runtime_error("GroupQueryAttention expects 3D Q/K/V tensors [batch, seq, hidden]");
    }

    auto getOptionalTensor = [&](size_t idx) -> std::shared_ptr<Tensor> {
        if (idx >= inputs.size()) return nullptr;
        const auto& name = inputs[idx];
        if (name.empty()) return nullptr;
        if (!hasTensor(name)) return nullptr;  // Handle missing optional tensors
        return getTensor(name);
    };

    size_t cursor = 3;
    auto past_key = getOptionalTensor(cursor);
    if (past_key && past_key->ndim() != 4) {
        // This wasn't a cache tensor, treat as something else.
        past_key = nullptr;
    } else if (past_key) {
        ++cursor;
    }

    auto past_value = getOptionalTensor(cursor);
    if (past_value && past_value->ndim() == 4) {
        ++cursor;
    } else {
        past_value = nullptr;
    }

    if ((past_key && !past_value) || (!past_key && past_value)) {
        throw std::runtime_error("GroupQueryAttention requires both past_key and past_value when using KV cache");
    }

    auto seq_len_tensor = getOptionalTensor(cursor);
    if (seq_len_tensor) ++cursor;
    auto total_len_tensor = getOptionalTensor(cursor);

    int64_t q_heads_attr = node.getIntAttr("num_heads", 0);
    int64_t kv_heads_attr = node.getIntAttr("kv_num_heads", 0);
    if (q_heads_attr <= 0 || kv_heads_attr <= 0) {
        throw std::runtime_error("GroupQueryAttention requires positive num_heads and kv_num_heads attributes");
    }

    size_t batch = static_cast<size_t>(Q->dim(0));
    size_t q_seq = static_cast<size_t>(Q->dim(1));
    size_t q_hidden = static_cast<size_t>(Q->dim(2));
    size_t q_heads = static_cast<size_t>(q_heads_attr);
    if (q_hidden % q_heads != 0) {
        throw std::runtime_error("GroupQueryAttention: Q hidden size must be divisible by num_heads");
    }
    size_t head_dim = q_hidden / q_heads;

    size_t kv_seq = static_cast<size_t>(K->dim(1));
    size_t k_hidden = static_cast<size_t>(K->dim(2));
    size_t kv_heads = static_cast<size_t>(kv_heads_attr);
    if (k_hidden % kv_heads != 0) {
        throw std::runtime_error("GroupQueryAttention: K hidden size must be divisible by kv_num_heads");
    }
    size_t key_head_dim = k_hidden / kv_heads;
    if (key_head_dim != head_dim) {
        throw std::runtime_error("GroupQueryAttention: head dimensions of Q and K must match");
    }

    size_t v_hidden = static_cast<size_t>(V->dim(2));
    if (v_hidden % kv_heads != 0) {
        throw std::runtime_error("GroupQueryAttention: V hidden size must be divisible by kv_num_heads");
    }
    size_t value_head_dim = v_hidden / kv_heads;
    if (value_head_dim != head_dim) {
        throw std::runtime_error("GroupQueryAttention: value head dimension must equal query head dimension");
    }

    size_t past_len = past_key ? static_cast<size_t>(past_key->dim(2)) : 0;
    size_t total_seq = past_len + kv_seq;

    auto tensorToIntVector = [&](const std::shared_ptr<Tensor>& tensor) -> std::vector<int64_t> {
        if (!tensor) return {};
        std::vector<int64_t> values(tensor->size());
        switch (tensor->dtype()) {
            case DataType::INT64: {
                std::vector<uint8_t> cache;
                const int64_t* data = getHostData<int64_t>(tensor, cache);
                std::copy(data, data + tensor->size(), values.begin());
                break;
            }
            case DataType::INT32: {
                std::vector<uint8_t> cache;
                const int32_t* data = getHostData<int32_t>(tensor, cache);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<int64_t>(data[i]);
                }
                break;
            }
            default: {
                std::vector<uint8_t> cache;
                const float* data = getHostData<float>(tensor, cache);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<int64_t>(std::llround(data[i]));
                }
                break;
            }
        }
        return values;
    };

    std::vector<int64_t> seq_len_values = tensorToIntVector(seq_len_tensor);
    std::vector<int64_t> total_len_values = tensorToIntVector(total_len_tensor);

    std::vector<size_t> valid_lengths(batch, total_seq);
    if (!seq_len_values.empty()) {
        for (size_t b = 0; b < batch && b < seq_len_values.size(); ++b) {
            int64_t len = seq_len_values[b];
            if (len >= 0) {
                size_t inferred = static_cast<size_t>(len + 1);
                if (!total_len_values.empty()) {
                    inferred = static_cast<size_t>(
                        std::min<int64_t>(inferred, total_len_values[0] > 0 ? total_len_values[0] : static_cast<int64_t>(total_seq)));
                }
                valid_lengths[b] = std::min(static_cast<size_t>(total_seq), inferred);
            }
        }
    }

    if (past_key) {
        if (past_key->dim(0) != static_cast<int64_t>(batch) ||
            past_key->dim(1) != static_cast<int64_t>(kv_heads) ||
            past_key->dim(3) != static_cast<int64_t>(head_dim)) {
            throw std::runtime_error("GroupQueryAttention: past_key shape mismatch");
        }
    }
    if (past_value) {
        if (past_value->dim(0) != static_cast<int64_t>(batch) ||
            past_value->dim(1) != static_cast<int64_t>(kv_heads) ||
            past_value->dim(2) != static_cast<int64_t>(past_len) ||
            past_value->dim(3) != static_cast<int64_t>(value_head_dim)) {
            throw std::runtime_error("GroupQueryAttention: past_value shape mismatch");
        }
    }

    size_t group_size = q_heads / kv_heads;
    if (group_size * kv_heads != q_heads) {
        throw std::runtime_error("GroupQueryAttention: q_num_heads must be a multiple of kv_num_heads");
    }

    std::vector<uint8_t> q_cache;
    std::vector<uint8_t> k_cache;
    std::vector<uint8_t> v_cache;
    const float* q_data = getHostData<float>(Q, q_cache);
    const float* k_data = getHostData<float>(K, k_cache);
    const float* v_data = getHostData<float>(V, v_cache);

    auto copyCacheTensor = [&](const std::shared_ptr<Tensor>& tensor, size_t expected) -> std::vector<float> {
        if (!tensor) return {};
        std::vector<uint8_t> cache;
        const float* data = getHostData<float>(tensor, cache);
        std::vector<float> result(expected);
        std::memcpy(result.data(), data, expected * sizeof(float));
        return result;
    };

    size_t key_storage_elems = batch * kv_heads * total_seq * head_dim;
    size_t value_storage_elems = batch * kv_heads * total_seq * value_head_dim;
    std::vector<float> key_storage(key_storage_elems, 0.f);
    std::vector<float> value_storage(value_storage_elems, 0.f);

    if (past_key) {
        size_t elems = batch * kv_heads * past_len * head_dim;
        auto past_data = copyCacheTensor(past_key, elems);
        size_t src_stride = past_len * head_dim;
        size_t dst_stride = total_seq * head_dim;
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < kv_heads; ++h) {
                const float* src = past_data.data() + ((b * kv_heads + h) * src_stride);
                float* dst = key_storage.data() + ((b * kv_heads + h) * dst_stride);
                std::memcpy(dst, src, src_stride * sizeof(float));
            }
        }
    }
    if (past_value) {
        size_t elems = batch * kv_heads * past_len * value_head_dim;
        auto past_data = copyCacheTensor(past_value, elems);
        size_t src_stride = past_len * value_head_dim;
        size_t dst_stride = total_seq * value_head_dim;
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < kv_heads; ++h) {
                const float* src = past_data.data() + ((b * kv_heads + h) * src_stride);
                float* dst = value_storage.data() + ((b * kv_heads + h) * dst_stride);
                std::memcpy(dst, src, src_stride * sizeof(float));
            }
        }
    }

    for (size_t b = 0; b < batch; ++b) {
        for (size_t seq = 0; seq < kv_seq; ++seq) {
            for (size_t h = 0; h < kv_heads; ++h) {
                const float* src_k = k_data + ((b * kv_seq + seq) * k_hidden) + h * head_dim;
                const float* src_v = v_data + ((b * kv_seq + seq) * v_hidden) + h * value_head_dim;
                float* dst_k = key_storage.data() +
                               (((b * kv_heads + h) * total_seq) + (past_len + seq)) * head_dim;
                float* dst_v = value_storage.data() +
                               (((b * kv_heads + h) * total_seq) + (past_len + seq)) * value_head_dim;
                std::memcpy(dst_k, src_k, head_dim * sizeof(float));
                std::memcpy(dst_v, src_v, value_head_dim * sizeof(float));
            }
        }
    }

    float scale_attr = node.getFloatAttr("scale", 0.0f);
    float scale = scale_attr != 0.0f ? scale_attr : (1.0f / std::sqrt(static_cast<float>(head_dim)));
    float softcap = node.getFloatAttr("softcap", 0.0f);

    std::vector<float> output_data(batch * q_seq * q_hidden, 0.f);
    std::vector<float> scores(total_seq, 0.f);

    for (size_t b = 0; b < batch; ++b) {
        size_t valid = std::min(valid_lengths[b], total_seq);
        if (valid == 0) valid = total_seq;
        for (size_t qh = 0; qh < q_heads; ++qh) {
            size_t kv_head = qh / group_size;
            const float* key_head = key_storage.data() + ((b * kv_heads + kv_head) * total_seq * head_dim);
            const float* value_head = value_storage.data() + ((b * kv_heads + kv_head) * total_seq * value_head_dim);
            for (size_t qs = 0; qs < q_seq; ++qs) {
                const float* q_vec = q_data + ((b * q_seq + qs) * q_hidden) + qh * head_dim;
                size_t causal_limit = past_len + qs + 1;
                size_t allowed = std::min(valid, causal_limit);
                if (allowed == 0) allowed = 1;

                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t t = 0; t < allowed; ++t) {
                    const float* k_vec = key_head + t * head_dim;
                    float dot = 0.f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += q_vec[d] * k_vec[d];
                    }
                    float scaled = dot * scale;
                    if (softcap > 0.f) {
                        scaled = softcap * std::tanh(scaled / softcap);
                    }
                    scores[t] = scaled;
                    if (scaled > max_score) max_score = scaled;
                }

                float denom = 0.f;
                for (size_t t = 0; t < allowed; ++t) {
                    float expv = std::exp(scores[t] - max_score);
                    scores[t] = expv;
                    denom += expv;
                }
                float inv_denom = denom > 0 ? 1.0f / denom : 0.0f;

                float* out_vec = output_data.data() + ((b * q_seq + qs) * q_hidden) + qh * head_dim;
                std::fill(out_vec, out_vec + head_dim, 0.f);

                for (size_t t = 0; t < allowed; ++t) {
                    float weight = scores[t] * inv_denom;
                    const float* v_vec = value_head + t * value_head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out_vec[d] += weight * v_vec[d];
                    }
                }
            }
        }
    }

    auto output = std::make_shared<Tensor>(Q->shape(), DataType::FLOAT32);
    std::memcpy(output->data<float>(), output_data.data(), output_data.size() * sizeof(float));

    auto present_key = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(batch),
                             static_cast<int64_t>(kv_heads),
                             static_cast<int64_t>(total_seq),
                             static_cast<int64_t>(head_dim)},
        DataType::FLOAT32);
    std::memcpy(present_key->data<float>(), key_storage.data(), key_storage.size() * sizeof(float));

    auto present_value = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(batch),
                             static_cast<int64_t>(kv_heads),
                             static_cast<int64_t>(total_seq),
                             static_cast<int64_t>(value_head_dim)},
        DataType::FLOAT32);
    std::memcpy(present_value->data<float>(), value_storage.data(), value_storage.size() * sizeof(float));

    if (!use_cpu_fallback_) {
        output->toGPU();
        present_key->toGPU();
        present_value->toGPU();
    }

    tensors_[outputs[0]] = output;
    if (outputs.size() > 1 && !outputs[1].empty()) {
        tensors_[outputs[1]] = present_key;
    }
    if (outputs.size() > 2 && !outputs[2].empty()) {
        tensors_[outputs[2]] = present_value;
    }
}
