void CpuExecutor::executeCos(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Cos expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    if (input->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("Cos currently supports FLOAT32 only");
    }

    auto output = allocateOutput(input->shape(), DataType::FLOAT32);
    const float* src = input->data_ptr<float>();
    float* dst = output->data_ptr<float>();

    for (size_t i = 0; i < input->size(); ++i) {
        dst[i] = std::cos(src[i]);
    }

    tensors_[node.outputs()[0]] = output;
}

