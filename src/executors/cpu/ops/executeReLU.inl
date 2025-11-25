void CpuExecutor::executeReLU(const Node& node) {
    // ReLU: Y = max(0, X)
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("ReLU expects 1 input and 1 output");
    }

    auto X = getTensor(node.inputs()[0]);
    auto Y = allocateOutput(X->shape());

    int size = X->size();
    LOG_DEBUG("  ReLU: size=", size);

    // Execute on CPU
    if (num_cpu_threads_ > 1) {
        kernels::reluCPUMultiThreaded(X->data<float>(), Y->data<float>(), size, num_cpu_threads_);
    } else {
        kernels::reluCPU(X->data<float>(), Y->data<float>(), size);
    }

    tensors_[node.outputs()[0]] = Y;
}
