void GpuExecutor::executeTrilu(const Node& node) {
    if (node.inputs().empty() || node.outputs().size() != 1) {
        throw std::runtime_error("Trilu expects at least 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    if (input->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("Trilu currently supports FLOAT32 only");
    }

    int64_t k = 0;
    if (node.inputs().size() >= 2) {
        auto k_tensor = getTensor(node.inputs()[1]);
        k = readScalarAsInt(k_tensor);
    }

    bool upper = node.getIntAttr("upper", 1) != 0;
    if (input->ndim() < 2) {
        throw std::runtime_error("Trilu expects input with rank >= 2");
    }

    auto output = std::make_shared<CpuTensor>(input->shape(), DataType::FLOAT32);

    // Handle empty tensors
    if (input->size() == 0) {
        if (!use_cpu_fallback_) {
            output->toGPU();
        }
        tensors_[node.outputs()[0]] = output;
        return;
    }

    std::vector<uint8_t> cache;
    const float* src = getHostData<float>(input, cache);
    float* dst = output->data_ptr<float>();

    int64_t rows = input->dim(input->ndim() - 2);
    int64_t cols = input->dim(input->ndim() - 1);
    int64_t matrix_size = rows * cols;
    int64_t batches = input->size() / matrix_size;

    for (int64_t b = 0; b < batches; ++b) {
        const float* src_matrix = src + b * matrix_size;
        float* dst_matrix = dst + b * matrix_size;
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                bool keep = upper ? ((j - i) >= k) : ((j - i) <= k);
                dst_matrix[i * cols + j] = keep ? src_matrix[i * cols + j] : 0.0f;
            }
        }
    }

    if (!use_cpu_fallback_) {
        output->toGPU();
    }

    tensors_[node.outputs()[0]] = output;
}
