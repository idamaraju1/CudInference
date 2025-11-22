void GpuExecutor::executeSqrt(const Node& node) {
    // Sqrt: Y = sqrt(A)
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Sqrt expects 1 input and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto output = allocateOutput(A->shape());

    launchSqrtKernel(A->data_ptr<float>(), output->data_ptr<float>(), A->size(), use_cpu_fallback_, num_cpu_threads_);

    tensors_[node.outputs()[0]] = output;
}
