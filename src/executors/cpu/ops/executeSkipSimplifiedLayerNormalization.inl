void CpuExecutor::executeSkipSimplifiedLayerNormalization(const Node& node) {
    if (node.inputs().size() != 3) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization expects input, skip, and gamma tensors");
    }

    auto input = getTensor(node.inputs()[0]);
    auto skip = getTensor(node.inputs()[1]);
    auto gamma = getTensor(node.inputs()[2]);

    if (input->dtype() != DataType::FLOAT32 || skip->dtype() != DataType::FLOAT32 ||
        gamma->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization currently supports FLOAT32 tensors only");
    }
    if (input->shape() != skip->shape()) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization: input and skip tensors must share the same shape");
    }
    if (input->ndim() < 2) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization expects rank >= 2 tensors");
    }

    size_t hidden = static_cast<size_t>(input->dim(input->ndim() - 1));
    if (gamma->ndim() != 1 || static_cast<size_t>(gamma->dim(0)) != hidden) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization: gamma must be 1D with length equal to the last dimension");
    }

    float epsilon = node.getFloatAttr("epsilon", 1e-5f);
    size_t rows = input->size() / hidden;

    auto sum_tensor = allocateOutput(input->shape(), DataType::FLOAT32);
    auto output_tensor = allocateOutput(input->shape(), DataType::FLOAT32);

    const float* input_data = input->data_ptr<float>();
    const float* skip_data = skip->data_ptr<float>();
    const float* gamma_data = gamma->data_ptr<float>();
    float* sum_data = sum_tensor->data_ptr<float>();
    float* output_data = output_tensor->data_ptr<float>();

    std::vector<float> mean(rows, 0.0f);
    std::vector<float> inv_std(rows, 0.0f);

    for (size_t row = 0; row < rows; ++row) {
        size_t base = row * hidden;
        float row_sum = 0.f;
        for (size_t col = 0; col < hidden; ++col) {
            float value = input_data[base + col] + skip_data[base + col];
            sum_data[base + col] = value;
            row_sum += value;
        }

        float mean_val = row_sum / static_cast<float>(hidden);
        float var_sum = 0.f;
        for (size_t col = 0; col < hidden; ++col) {
            float diff = sum_data[base + col] - mean_val;
            var_sum += diff * diff;
        }
        float variance = var_sum / static_cast<float>(hidden);
        float inv = 1.0f / std::sqrt(variance + epsilon);

        mean[row] = mean_val;
        inv_std[row] = inv;

        for (size_t col = 0; col < hidden; ++col) {
            float normalized = (sum_data[base + col] - mean_val) * inv;
            output_data[base + col] = normalized * gamma_data[col];
        }
    }

    auto mean_tensor = allocateOutput(
        std::vector<int64_t>{static_cast<int64_t>(rows)}, DataType::FLOAT32);
    auto invstd_tensor = allocateOutput(
        std::vector<int64_t>{static_cast<int64_t>(rows)}, DataType::FLOAT32);

    std::memcpy(mean_tensor->data_ptr<float>(), mean.data(), rows * sizeof(float));
    std::memcpy(invstd_tensor->data_ptr<float>(), inv_std.data(), rows * sizeof(float));

    const auto& outs = node.outputs();
    if (!outs.empty() && !outs[0].empty()) {
        tensors_[outs[0]] = output_tensor;
    }
    if (outs.size() > 1 && !outs[1].empty()) {
        tensors_[outs[1]] = mean_tensor;
    }
    if (outs.size() > 2 && !outs[2].empty()) {
        tensors_[outs[2]] = invstd_tensor;
    }
    if (outs.size() > 3 && !outs[3].empty()) {
        tensors_[outs[3]] = sum_tensor;
    }
}

