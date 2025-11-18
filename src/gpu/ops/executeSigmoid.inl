void GpuExecutor::executeSigmoid(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Sigmoid expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    if (input->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("Sigmoid currently supports FLOAT32 only");
    }

    auto output = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32);
    std::vector<uint8_t> cache;
    const float* src = getHostData<float>(input, cache);
    float* dst = output->ptr_data<float>();

    for (size_t i = 0; i < input->size(); ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }

    if (!use_cpu_fallback_) {
        output->toGPU();
    }

    tensors_[node.outputs()[0]] = output;
}
