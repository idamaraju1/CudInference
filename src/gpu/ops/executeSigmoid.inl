void GpuExecutor::executeSigmoid(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Sigmoid expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    if (input->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("Sigmoid currently supports FLOAT32 only");
    }

    auto output = allocateOutput(input->shape(), DataType::FLOAT32);
    int size = static_cast<int>(input->size());

    if (use_cpu_fallback_) {
        // CPU path
        std::vector<uint8_t> cache;
        const float* src = getHostData<float>(input, cache);
        float* dst = output->data<float>();

        if (num_cpu_threads_ > 1) {
            kernels::sigmoidCPUMultiThreaded(src, dst, size, num_cpu_threads_);
        } else {
            kernels::sigmoidCPU(src, dst, size);
        }
    } else {
        // GPU path
        kernels::launchSigmoid(input->data<float>(), output->data<float>(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = output;
}
