void GpuExecutor::executeNeg(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Neg expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    auto output = allocateOutput(input->shape(), input->dtype());

    // TODO - refactor: GPU Neg kernel not yet implemented
    // For now, compute on CPU then copy to GPU
    std::vector<uint8_t> cache;
    const float* src = getHostData<float>(input, cache);
    std::vector<float> host_output(input->size());
    
    switch (input->dtype()) {
        case DataType::FLOAT32:
            for (size_t i = 0; i < input->size(); ++i) {
                host_output[i] = -src[i];
            }
            break;
        case DataType::INT32: {
            const int32_t* src_int = getHostData<int32_t>(input, cache);
            for (size_t i = 0; i < input->size(); ++i) {
                host_output[i] = static_cast<float>(-src_int[i]);
            }
            break;
        }
        case DataType::INT64: {
            const int64_t* src_int = getHostData<int64_t>(input, cache);
            for (size_t i = 0; i < input->size(); ++i) {
                host_output[i] = static_cast<float>(-src_int[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("Neg: Unsupported data type " +
                                     dataTypeToString(input->dtype()));
    }

    CUDA_CHECK(cudaMemcpy(output->data_ptr<float>(), host_output.data(),
                          input->size() * sizeof(float), cudaMemcpyHostToDevice));

    tensors_[node.outputs()[0]] = output;
}

