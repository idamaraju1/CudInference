void GpuExecutor::executeNeg(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Neg expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    std::shared_ptr<TensorBase> output = std::make_shared<CpuTensor>(input->shape(), input->dtype());
    bool move_to_gpu = !use_cpu_fallback_;

    switch (input->dtype()) {
        case DataType::FLOAT32:
            applyUnary<float>(input, output, [](float v) { return -v; }, move_to_gpu);
            break;
        case DataType::INT32:
            applyUnary<int32_t>(input, output, [](int32_t v) { return -v; }, move_to_gpu);
            break;
        case DataType::INT64:
            applyUnary<int64_t>(input, output, [](int64_t v) { return -v; }, move_to_gpu);
            break;
        default:
            throw std::runtime_error("Neg: Unsupported data type " +
                                     dataTypeToString(input->dtype()));
    }

    tensors_[node.outputs()[0]] = output;
}
