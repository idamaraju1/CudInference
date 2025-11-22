void GpuExecutor::executeRotaryEmbedding(const Node& node) {
    const auto& inputs = node.inputs();
    const auto& outputs = node.outputs();

    if (verbose_) {
        LOG_DEBUG("RotaryEmbedding: received ", inputs.size(), " inputs");
    }

    if (inputs.size() < 3 || outputs.empty()) {
        throw std::runtime_error("RotaryEmbedding expects at least 3 inputs and 1 output");
    }

    // Parse inputs (same as before)
    auto data_tensor = getTensor(inputs[0]);
    if (data_tensor->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("RotaryEmbedding currently supports FLOAT32 only");
    }

    std::shared_ptr<Tensor> cos_tensor;
    std::shared_ptr<Tensor> sin_tensor;
    std::shared_ptr<Tensor> position_tensor;

    // Parse input configuration (same logic as before)
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
            if (hasTensor("position_ids")) {
                position_tensor = getTensor("position_ids");
            }
        }
    } else if (inputs.size() == 5) {
        position_tensor = getTensor(inputs[1]);
        cos_tensor = getTensor(inputs[2]);
        sin_tensor = getTensor(inputs[3]);
    } else {
        throw std::runtime_error("RotaryEmbedding: unsupported input configuration");
    }

    if (!cos_tensor || !sin_tensor) {
        throw std::runtime_error("RotaryEmbedding requires cos and sin cache tensors");
    }

    // Parse dimensions
    const auto& data_shape = data_tensor->shape();
    if (data_shape.size() < 3 || data_shape.size() > 4) {
        throw std::runtime_error("RotaryEmbedding input rank must be 3 or 4");
    }

    size_t batch = static_cast<size_t>(data_shape[0]);
    size_t sequence_length = static_cast<size_t>(data_shape.size() == 4 ? data_shape[2] : data_shape[1]);
    size_t hidden_total = static_cast<size_t>(data_shape.back());

    size_t num_heads = 0;
    size_t head_size = 0;
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

    // Prepare position data
    auto position_shape = position_tensor ? position_tensor->shape() : std::vector<int64_t>{};
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

    // Helper to fetch position ID
    auto fetchPositionId = [&](size_t batch_idx, size_t seq_idx) -> int64_t {
        if (!position_data) return static_cast<int64_t>(seq_idx);
        if (position_shape.size() == 1) return position_data[seq_idx];
        size_t pos_batch = static_cast<size_t>(position_shape[0]);
        size_t pos_seq = static_cast<size_t>(position_shape[1]);
        size_t b_idx = (pos_batch == 1) ? 0 : batch_idx;
        size_t s_idx = (pos_seq == 1) ? 0 : seq_idx;
        return position_data[b_idx * pos_seq + s_idx];
    };

    // Prepare cos/sin caches (select appropriate values based on positions)
    std::vector<uint8_t> cos_cache, sin_cache;
    const float* cos_data = getHostData<float>(cos_tensor, cos_cache);
    const float* sin_data = getHostData<float>(sin_tensor, sin_cache);

    const auto& cos_shape = cos_tensor->shape();
    std::vector<float> selected_cos(batch * sequence_length * rotary_half);
    std::vector<float> selected_sin(batch * sequence_length * rotary_half);

    // Select appropriate cos/sin values based on position_ids
    if (cos_shape.size() == 2) {
        size_t max_positions = static_cast<size_t>(cos_shape[0]);
        size_t dim = static_cast<size_t>(cos_shape[1]);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < sequence_length; ++s) {
                int64_t pos = fetchPositionId(b, s);
                if (pos < 0 || pos >= static_cast<int64_t>(max_positions)) {
                    pos = std::min(std::max<int64_t>(0, pos), static_cast<int64_t>(max_positions - 1));
                }
                size_t src_offset = static_cast<size_t>(pos) * dim;
                size_t dst_offset = (b * sequence_length + s) * rotary_half;
                std::memcpy(&selected_cos[dst_offset], &cos_data[src_offset], rotary_half * sizeof(float));
                std::memcpy(&selected_sin[dst_offset], &sin_data[src_offset], rotary_half * sizeof(float));
            }
        }
    } else if (cos_shape.size() == 3) {
        // 3D cache [batch, seq, dim]
        size_t cache_batch = static_cast<size_t>(cos_shape[0]);
        size_t cache_seq = static_cast<size_t>(cos_shape[1]);
        size_t dim = static_cast<size_t>(cos_shape[2]);
        for (size_t b = 0; b < batch; ++b) {
            size_t b_idx = cache_batch == 1 ? 0 : b;
            for (size_t s = 0; s < sequence_length; ++s) {
                size_t s_idx = cache_seq == 1 ? 0 : s;
                size_t src_offset = (b_idx * cache_seq + s_idx) * dim;
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
        // BNSH -> BSNH reordering
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

    // Allocate output
    auto output = allocateOutput(data_tensor->shape(), DataType::FLOAT32);
    std::vector<float> rotated(data_tensor->size());

    int batch_seq = static_cast<int>(batch * sequence_length);

    // ========================================================================
    // GPU_PERSISTENT MODE: Keep everything on GPU
    // ========================================================================
    if (exec_mode_ == ExecutionMode::GPU_PERSISTENT) {
        // Ensure inputs are on GPU
        data_tensor->ensureOnGPU();
        cos_tensor->ensureOnGPU();
        sin_tensor->ensureOnGPU();

        // Get GPU pointers - assume cos/sin are already at correct positions
        // (position selection logic would need GPU implementation for full support)
        const float* d_input = data_tensor->deviceData<float>();
        const float* d_cos = cos_tensor->deviceData<float>();
        const float* d_sin = sin_tensor->deviceData<float>();

        // TODO: Handle BNSH -> BSNH conversion on GPU if needed
        // For now, assume input is in correct format or handle conversion later

        // Allocate output on GPU
        auto output = allocateOutput(data_tensor->shape(), DataType::FLOAT32);

        // Launch RoPE kernel with GPU pointers
        launchRotaryEmbedding(
            d_input, d_cos, d_sin,
            output->mutableDeviceData<float>(),
            batch_seq,
            static_cast<int>(num_heads),
            static_cast<int>(head_size),
            static_cast<int>(rotary_dim),
            interleaved,
            false,  // use_cpu = false
            num_cpu_threads_
        );

        // Output stays on GPU!
        tensors_[outputs[0]] = output;

        if (verbose_) {
            LOG_DEBUG("RotaryEmbedding GPU_PERSISTENT: batch_seq=", batch_seq,
                      ", num_heads=", num_heads, ", head_size=", head_size);
        }

        return;  // Done with GPU_PERSISTENT path
    }

    // ========================================================================
    // CPU_ONLY and GPU_COPY MODES (legacy paths)
    // ========================================================================
    if (use_cpu_fallback_) {
        // CPU path
        launchRotaryEmbedding(
            input_data, selected_cos.data(), selected_sin.data(), rotated.data(),
            batch_seq, static_cast<int>(num_heads), static_cast<int>(head_size),
            static_cast<int>(rotary_dim), interleaved, true, num_cpu_threads_
        );

        // Convert back to BNSH if needed
        if (data_shape.size() == 4) {
            std::vector<float> final_output(rotated.size());
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < sequence_length; ++s) {
                    for (size_t n = 0; n < num_heads; ++n) {
                        size_t src_idx = ((b * sequence_length + s) * num_heads + n) * head_size;
                        size_t dst_idx = ((b * num_heads + n) * sequence_length + s) * head_size;
                        std::memcpy(&final_output[dst_idx], &rotated[src_idx], head_size * sizeof(float));
                    }
                }
            }
            std::memcpy(output->data<float>(), final_output.data(), final_output.size() * sizeof(float));
        } else {
            std::memcpy(output->data<float>(), rotated.data(), rotated.size() * sizeof(float));
        }
    } else {
        // GPU path
        float* d_input;
        float* d_cos;
        float* d_sin;
        float* d_output_rope;

        size_t data_bytes = batch_seq * num_heads * head_size * sizeof(float);
        size_t cache_bytes = batch_seq * rotary_half * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
        CUDA_CHECK(cudaMalloc(&d_cos, cache_bytes));
        CUDA_CHECK(cudaMalloc(&d_sin, cache_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_rope, data_bytes));

        CUDA_CHECK(cudaMemcpy(d_input, input_data, data_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, selected_cos.data(), cache_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, selected_sin.data(), cache_bytes, cudaMemcpyHostToDevice));

        launchRotaryEmbedding(
            d_input, d_cos, d_sin, d_output_rope,
            batch_seq, static_cast<int>(num_heads), static_cast<int>(head_size),
            static_cast<int>(rotary_dim), interleaved, false, num_cpu_threads_
        );

        CUDA_CHECK(cudaMemcpy(rotated.data(), d_output_rope, data_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_cos));
        CUDA_CHECK(cudaFree(d_sin));
        CUDA_CHECK(cudaFree(d_output_rope));

        // Convert back to BNSH if needed
        if (data_shape.size() == 4) {
            std::vector<float> final_output(rotated.size());
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < sequence_length; ++s) {
                    for (size_t n = 0; n < num_heads; ++n) {
                        size_t src_idx = ((b * sequence_length + s) * num_heads + n) * head_size;
                        size_t dst_idx = ((b * num_heads + n) * sequence_length + s) * head_size;
                        std::memcpy(&final_output[dst_idx], &rotated[src_idx], head_size * sizeof(float));
                    }
                }
            }
            CUDA_CHECK(cudaMemcpy(output->data<float>(), final_output.data(),
                                 final_output.size() * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(output->data<float>(), rotated.data(),
                                 rotated.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    tensors_[outputs[0]] = output;
}
