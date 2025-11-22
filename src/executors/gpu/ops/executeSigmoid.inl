void GpuExecutor::executeSigmoid(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Sigmoid expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    if (input->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("Sigmoid currently supports FLOAT32 only");
    }

    auto output = allocateOutput(input->shape(), DataType::FLOAT32);

    // TODO - refactor: GPU Sigmoid kernel not yet implemented
    // For now, compute on CPU then copy to GPU
    std::vector<uint8_t> cache;
    const float* src = getHostData<float>(input, cache);
    std::vector<float> host_output(input->size());

    for (size_t i = 0; i < input->size(); ++i) {
        host_output[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }

    CUDA_CHECK(cudaMemcpy(output->data_ptr<float>(), host_output.data(),
                          input->size() * sizeof(float), cudaMemcpyHostToDevice));

    tensors_[node.outputs()[0]] = output;
}

