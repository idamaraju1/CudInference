void GpuExecutor::executeRotaryEmbedding(const Node& node) {
    const auto& inputs = node.inputs();
    const auto& outputs = node.outputs();

    if (verbose_) {
        LOG_DEBUG("RotaryEmbedding: received ", inputs.size(), " inputs");
        for (size_t i = 0; i < inputs.size(); ++i) {
            LOG_DEBUG("  Input[", i, "]: ", inputs[i]);
        }
    }

    if (inputs.size() < 3 || outputs.empty()) {
        throw std::runtime_error("RotaryEmbedding expects at least 1 output");
    }

    auto data_tensor = getTensor(inputs[0]);
    if (data_tensor->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("RotaryEmbedding currently supports FLOAT32 activations only");
    }

    std::shared_ptr<Tensor> cos_tensor;
    std::shared_ptr<Tensor> sin_tensor;
    std::shared_ptr<Tensor> position_tensor;

    if (inputs.size() == 4) {
        position_tensor = getTensor(inputs[1]);
        cos_tensor = getTensor(inputs[2]);
        sin_tensor = getTensor(inputs[3]);
    } else if (inputs.size() == 3) {
        auto second = getTensor(inputs[1]);
        auto third = getTensor(inputs[2]);
        if (second->dtype() == DataType::INT64 || second->dtype() == DataType::INT32) {
            position_tensor = second;
            cos_tensor = third;
            throw std::runtime_error("RotaryEmbedding: missing sine cache tensor");
        } else {
            cos_tensor = second;
            sin_tensor = third;
        }
    } else if (inputs.size() == 5) {
        // Some exporter order: X, position_ids, cos, sin, (unused extra e.g., past_position)
        position_tensor = getTensor(inputs[1]);
        cos_tensor = getTensor(inputs[2]);
        sin_tensor = getTensor(inputs[3]);
        // ignore remaining tensors
    } else {
        throw std::runtime_error("RotaryEmbedding: unsupported input configuration");
    }

    if (!cos_tensor || !sin_tensor) {
        throw std::runtime_error("RotaryEmbedding requires cosine and sine cache tensors");
    }
    if (cos_tensor->dtype() != DataType::FLOAT32 || sin_tensor->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("RotaryEmbedding currently supports FLOAT32 cos/sin tensors only");
    }

    const auto& data_shape = data_tensor->shape();
    if (data_shape.size() < 3 || data_shape.size() > 4) {
        throw std::runtime_error("RotaryEmbedding input rank must be 3 or 4");
    }

    size_t batch = static_cast<size_t>(data_shape[0]);
    size_t sequence_length = static_cast<size_t>(data_shape.size() == 4 ? data_shape[2] : data_shape[1]);
    size_t hidden_total = static_cast<size_t>(data_shape.back());

    auto inferHeadSize = [&](const std::shared_ptr<Tensor>& cache) -> size_t {
        const auto& shape = cache->shape();
        if (shape.empty()) {
            throw std::runtime_error("RotaryEmbedding: cannot infer head size from empty cache shape");
        }
        int64_t last_dim = shape.back();
        if (last_dim <= 0) {
            throw std::runtime_error("RotaryEmbedding: invalid cache last dimension");
        }
        return static_cast<size_t>(last_dim) * 2;
    };

    size_t num_heads = 0;
    size_t head_size = 0;
    int64_t num_heads_attr = node.getIntAttr("num_heads", 0);

    if (data_shape.size() == 4) {
        num_heads = static_cast<size_t>(data_shape[1]);
        head_size = static_cast<size_t>(data_shape[3]);
    } else {
        if (num_heads_attr > 0) {
            num_heads = static_cast<size_t>(num_heads_attr);
            if (num_heads == 0 || hidden_total % num_heads != 0) {
                throw std::runtime_error("RotaryEmbedding: invalid num_heads attribute");
            }
            head_size = hidden_total / num_heads;
        } else {
            head_size = inferHeadSize(cos_tensor);
            if (head_size == 0 || hidden_total % head_size != 0) {
                throw std::runtime_error("RotaryEmbedding: unable to infer num_heads from cache shape");
            }
            num_heads = hidden_total / head_size;
        }
    }

    int64_t rotary_dim_attr = node.getIntAttr("rotary_embedding_dim", static_cast<int64_t>(head_size));
    if (rotary_dim_attr == 0) {
        rotary_dim_attr = static_cast<int64_t>(head_size);
    }
    if (rotary_dim_attr <= 0 || rotary_dim_attr > static_cast<int64_t>(head_size) || (rotary_dim_attr % 2) != 0) {
        throw std::runtime_error("RotaryEmbedding: invalid rotary_embedding_dim");
    }
    size_t rotary_dim = static_cast<size_t>(rotary_dim_attr);
    size_t rotary_half = rotary_dim / 2;
    bool interleaved = node.getIntAttr("interleaved", 0) != 0;

    auto position_shape = position_tensor ? position_tensor->shape() : std::vector<int64_t>{};
    std::vector<uint8_t> position_cache;
    std::vector<int64_t> position_converted;
    const int64_t* position_data = nullptr;
    if (position_tensor) {
        if (position_shape.empty() || position_shape.size() > 2) {
            throw std::runtime_error("RotaryEmbedding: position_ids must be 1D or 2D");
        }

        switch (position_tensor->dtype()) {
            case DataType::INT64: {
                position_data = getHostData<int64_t>(position_tensor, position_cache);
                break;
            }
            case DataType::INT32: {
                const int32_t* src = getHostData<int32_t>(position_tensor, position_cache);
                position_converted.resize(position_tensor->size());
                for (size_t i = 0; i < position_converted.size(); ++i) {
                    position_converted[i] = static_cast<int64_t>(src[i]);
                }
                position_data = position_converted.data();
                break;
            }
            default: {
                const float* src = getHostData<float>(position_tensor, position_cache);
                position_converted.resize(position_tensor->size());
                for (size_t i = 0; i < position_converted.size(); ++i) {
                    position_converted[i] = static_cast<int64_t>(std::llround(src[i]));
                }
                position_data = position_converted.data();
                break;
            }
        }
    }

    auto fetchPositionId = [&](size_t batch_idx, size_t seq_idx) -> int64_t {
        if (!position_data) return 0;
        if (position_shape.size() == 1) {
            if (seq_idx >= static_cast<size_t>(position_shape[0])) {
                throw std::runtime_error("RotaryEmbedding: position_ids sequence dimension mismatch");
            }
            return position_data[seq_idx];
        }
        size_t pos_batch = static_cast<size_t>(position_shape[0]);
        size_t pos_seq = static_cast<size_t>(position_shape[1]);
        size_t b_idx = (pos_batch == 1) ? 0 : batch_idx;
        size_t s_idx = (pos_seq == 1) ? 0 : seq_idx;
        if (b_idx >= pos_batch || s_idx >= pos_seq) {
            throw std::runtime_error("RotaryEmbedding: position_ids broadcast mismatch");
        }
        return position_data[b_idx * pos_seq + s_idx];
    };

    auto prepareCache = [&](const std::shared_ptr<Tensor>& cache,
                            const std::string& label) -> std::vector<float> {
        const auto& cache_shape = cache->shape();
        std::vector<uint8_t> raw_cache;
        const float* cache_data = getHostData<float>(cache, raw_cache);
        if (cache_shape.empty()) {
            throw std::runtime_error("RotaryEmbedding: cache tensor " + label + " has empty shape");
        }

        // Debug logging for autoregressive generation
        if (verbose_) {
            LOG_DEBUG("RotaryEmbedding ", label, " cache shape: [", cache_shape[0]);
            for (size_t i = 1; i < cache_shape.size(); ++i) {
                LOG_DEBUG(", ", cache_shape[i]);
            }
            LOG_DEBUG("], sequence_length: ", sequence_length, ", has position_ids: ", (position_data ? "yes" : "no"));
        }

        std::vector<float> selected(batch * sequence_length * rotary_half);
        size_t rank = cache_shape.size();

        if (position_data) {
            if (rank != 2) {
                throw std::runtime_error("RotaryEmbedding: expected 2D " + label +
                                         " when position_ids are provided");
            }
            size_t rows = static_cast<size_t>(cache_shape[0]);
            size_t cols = static_cast<size_t>(cache_shape[1]);
            if (cols < rotary_half) {
                throw std::runtime_error("RotaryEmbedding: cache " + label + " width is smaller than half rotary dim");
            }

            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < sequence_length; ++s) {
                    int64_t pos = fetchPositionId(b, s);
                    if (pos < 0 || pos >= static_cast<int64_t>(rows)) {
                        throw std::runtime_error("RotaryEmbedding: position id out of range for cache " + label);
                    }
                    const float* src = cache_data + static_cast<size_t>(pos) * cols;
                    float* dst = selected.data() + (b * sequence_length + s) * rotary_half;
                    std::memcpy(dst, src, rotary_half * sizeof(float));
                }
            }
        } else if (rank == 3) {
            size_t cache_batch = static_cast<size_t>(cache_shape[0]);
            size_t cache_seq = static_cast<size_t>(cache_shape[1]);
            size_t cols = static_cast<size_t>(cache_shape[2]);
            if (cols < rotary_half) {
                throw std::runtime_error("RotaryEmbedding: cache " + label + " width is smaller than half rotary dim");
            }

            for (size_t b = 0; b < batch; ++b) {
                size_t b_idx = cache_batch == 1 ? 0 : b;
                if (b_idx >= cache_batch) {
                    throw std::runtime_error("RotaryEmbedding: cache batch mismatch for " + label);
                }
                for (size_t s = 0; s < sequence_length; ++s) {
                    size_t s_idx = cache_seq == 1 ? 0 : s;
                    if (s_idx >= cache_seq) {
                        throw std::runtime_error("RotaryEmbedding: cache sequence mismatch for " + label);
                    }
                    const float* src = cache_data + (b_idx * cache_seq + s_idx) * cols;
                    float* dst = selected.data() + (b * sequence_length + s) * rotary_half;
                    std::memcpy(dst, src, rotary_half * sizeof(float));
                }
            }
        } else if (rank == 2) {
            size_t rows = static_cast<size_t>(cache_shape[0]);
            size_t cols = static_cast<size_t>(cache_shape[1]);
            if (cols < rotary_half) {
                throw std::runtime_error("RotaryEmbedding: cache " + label + " width is smaller than half rotary dim");
            }

            // For autoregressive generation without position_ids:
            // Clamp sequence indices to available cache rows to prevent out-of-bounds access
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < sequence_length; ++s) {
                    size_t row = s;

                    // Safety: clamp to available cache size
                    if (row >= rows) {
                        if (verbose_) {
                            LOG_DEBUG("RotaryEmbedding: sequence position ", s,
                                     " exceeds cache rows ", rows, ", clamping to last row");
                        }
                        row = rows - 1;  // Use last available position
                    }

                    const float* src = cache_data + row * cols;
                    float* dst = selected.data() + (b * sequence_length + s) * rotary_half;
                    std::memcpy(dst, src, rotary_half * sizeof(float));
                }
            }
        } else {
            throw std::runtime_error("RotaryEmbedding: unsupported cache rank for " + label);
        }

        return selected;
    };

    auto cos_values = prepareCache(cos_tensor, "cos_cache");
    auto sin_values = prepareCache(sin_tensor, "sin_cache");

    std::vector<uint8_t> data_cache;
    const float* raw_data_ptr = getHostData<float>(data_tensor, data_cache);
    std::vector<float> reordered_input;
    const float* rotation_input = raw_data_ptr;

    size_t token_count = batch * sequence_length;

    if (data_shape.size() == 4) {
        reordered_input.resize(data_tensor->size());
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < sequence_length; ++s) {
                for (size_t h = 0; h < num_heads; ++h) {
                    size_t dst_index = ((b * sequence_length + s) * num_heads + h) * head_size;
                    size_t src_index = ((b * num_heads + h) * sequence_length + s) * head_size;
                    std::memcpy(reordered_input.data() + dst_index,
                                raw_data_ptr + src_index,
                                head_size * sizeof(float));
                }
            }
        }
        rotation_input = reordered_input.data();
    }

    std::vector<float> rotated_bsh(data_tensor->size());
    auto rotateHead = [&](const float* src,
                          float* dst,
                          const float* cos_ptr,
                          const float* sin_ptr) {
        if (!interleaved) {
            const float* first = src;
            const float* second = src + rotary_half;
            for (size_t i = 0; i < rotary_half; ++i) {
                float cos_val = cos_ptr[i];
                float sin_val = sin_ptr[i];
                float x1 = first[i];
                float x2 = second[i];
                dst[i] = x1 * cos_val - x2 * sin_val;
                dst[i + rotary_half] = x2 * cos_val + x1 * sin_val;
            }
        } else {
            for (size_t i = 0; i < rotary_half; ++i) {
                size_t even_idx = i * 2;
                size_t odd_idx = even_idx + 1;
                float cos_val = cos_ptr[i];
                float sin_val = sin_ptr[i];
                float x_even = src[even_idx];
                float x_odd = src[odd_idx];
                dst[even_idx] = x_even * cos_val - x_odd * sin_val;
                dst[odd_idx] = x_odd * cos_val + x_even * sin_val;
            }
        }
        if (head_size > rotary_dim) {
            std::memcpy(dst + rotary_dim, src + rotary_dim, (head_size - rotary_dim) * sizeof(float));
        }
    };

    for (size_t token = 0; token < token_count; ++token) {
        const float* cos_ptr = cos_values.data() + token * rotary_half;
        const float* sin_ptr = sin_values.data() + token * rotary_half;
        for (size_t head = 0; head < num_heads; ++head) {
            size_t offset = (token * num_heads + head) * head_size;
            rotateHead(rotation_input + offset, rotated_bsh.data() + offset, cos_ptr, sin_ptr);
        }
    }

    std::vector<float> final_output;
    const float* copy_source = rotated_bsh.data();
    if (data_shape.size() == 4) {
        final_output.resize(rotated_bsh.size());
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < sequence_length; ++s) {
                for (size_t h = 0; h < num_heads; ++h) {
                    size_t src_index = ((b * sequence_length + s) * num_heads + h) * head_size;
                    size_t dst_index = ((b * num_heads + h) * sequence_length + s) * head_size;
                    std::memcpy(final_output.data() + dst_index,
                                rotated_bsh.data() + src_index,
                                head_size * sizeof(float));
                }
            }
        }
        copy_source = final_output.data();
    }

    auto output = allocateOutput(data_tensor->shape(), data_tensor->dtype());
    size_t bytes = data_tensor->size() * sizeof(float);
    if (use_cpu_fallback_) {
        if (output->device() == DeviceType::CUDA) {
            output->toCPU();
        }
        std::memcpy(output->data_ptr<float>(), copy_source, bytes);
    } else {
        CUDA_CHECK(cudaMemcpy(output->data_ptr<float>(), copy_source, bytes, cudaMemcpyHostToDevice));
    }

    tensors_[outputs[0]] = output;
}
