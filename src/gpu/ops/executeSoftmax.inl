void GpuExecutor::executeSoftmax(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Softmax expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    if (input->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("Softmax currently supports FLOAT32 only");
    }

    int64_t axis = node.getIntAttr("axis", 1);
    if (axis < 0) {
        axis += static_cast<int64_t>(input->ndim());
    }
    if (axis < 0 || axis >= static_cast<int64_t>(input->ndim())) {
        throw std::runtime_error("Softmax: axis out of range");
    }

    auto output = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32);
    std::vector<uint8_t> cache;
    const float* src = getHostData<float>(input, cache);
    float* dst = output->ptr_data<float>();

    int64_t axis_dim = input->dim(axis);
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= input->dim(i);
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < static_cast<int64_t>(input->ndim()); ++i) {
        inner *= input->dim(i);
    }

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (int64_t inner_idx = 0; inner_idx < inner; ++inner_idx) {
            int64_t base = outer_idx * axis_dim * inner + inner_idx;
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t k = 0; k < axis_dim; ++k) {
                int64_t idx = base + k * inner;
                max_val = std::max(max_val, src[idx]);
            }

            float sum = 0.0f;
            for (int64_t k = 0; k < axis_dim; ++k) {
                int64_t idx = base + k * inner;
                dst[idx] = std::exp(src[idx] - max_val);
                sum += dst[idx];
            }

            float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;
            for (int64_t k = 0; k < axis_dim; ++k) {
                int64_t idx = base + k * inner;
                dst[idx] *= inv_sum;
            }
        }
    }

    if (!use_cpu_fallback_) {
        output->toGPU();
    }

    tensors_[node.outputs()[0]] = output;
}
