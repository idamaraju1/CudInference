void GpuExecutor::executeSimplifiedLayerNormalization(const Node& node) {
    if (node.inputs().size() < 1 || node.inputs().size() > 3 || node.outputs().size() != 1) {
        throw std::runtime_error("SimplifiedLayerNormalization expects X[, gamma][, beta] and 1 output");
    }

    auto X = getTensor(node.inputs()[0]);
    auto Y = allocateOutput(X->shape(), X->dtype());
    size_t total = X->size();
    size_t N = X->shape().back();
    size_t M = total / N;
    float epsilon = node.getFloatAttr("epsilon", 1e-5f);

    const float* gamma = nullptr;
    const float* beta = nullptr;

    if (node.inputs().size() >= 2) {
        auto G = getTensor(node.inputs()[1]);
        if (G) {
            if (G->device() == DeviceType::CPU) G->toGPU();
            gamma = G->data_ptr<float>();
        }
    }
    if (node.inputs().size() >= 3) {
        auto B = getTensor(node.inputs()[2]);
        if (B) {
            if (B->device() == DeviceType::CPU) B->toGPU();
            beta = B->data_ptr<float>();
        }
    }

    // Ensure device pointers
    if (X->device() == DeviceType::CPU) X->toGPU();
    if (Y->device() == DeviceType::CPU) Y->toGPU();

    // Execute on GPU
    kernels::launchSimplifiedLayerNorm(
        X->data_ptr<float>(), gamma, beta, Y->data_ptr<float>(), (int)M, (int)N, epsilon, /*stream*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    tensors_[node.outputs()[0]] = Y;
}

