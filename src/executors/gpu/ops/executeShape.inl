void GpuExecutor::executeShape(const Node& node) {
    // Shape: Returns the shape of the input tensor as a 1D tensor
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Shape expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    const auto& input_shape = input->shape();

    // Create output tensor: 1D tensor with length = ndim
    std::vector<int64_t> output_shape = {static_cast<int64_t>(input_shape.size())};
    auto output = std::make_shared<Tensor>(output_shape, DataType::INT64);

    int64_t* data_ptr = output->data<int64_t>();
    for (size_t i = 0; i < input_shape.size(); ++i) {
        data_ptr[i] = input_shape[i];
    }

    if (!use_cpu_fallback_) {
        output->toGPU();
    }

    tensors_[node.outputs()[0]] = output;
}
