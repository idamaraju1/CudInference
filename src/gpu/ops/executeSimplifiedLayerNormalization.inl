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
            if (use_cpu_fallback_ && G->device() == DeviceType::CUDA) G->toCPU();
            gamma = G->data_ptr<float>();
        }
    }
    if (node.inputs().size() >= 3) {
        auto B = getTensor(node.inputs()[2]);
        if (B) {
            if (use_cpu_fallback_ && B->device() == DeviceType::CUDA) B->toCPU();
            beta = B->data_ptr<float>();
        }
    }

    // Dispatch CPU or GPU path
    // inside executeSimplifiedLayerNormalization after you’ve validated shapes
    const auto& ins = node.inputs();
    if (use_cpu_fallback_) {
        // Ensure host access
        if (X->device() == DeviceType::CUDA) X->toCPU();
        if (Y->device() == DeviceType::CUDA) Y->toCPU();
        const float* gammaPtr = nullptr;
        const float* betaPtr  = nullptr;
        if (ins.size() >= 2) { auto G = getTensor(ins[1]); if (G && G->device()==DeviceType::CUDA) G->toCPU(); gammaPtr = ins.size()>=2 ? getTensor(ins[1])->data_ptr<float>() : nullptr; }
        if (ins.size() >= 3) { auto B = getTensor(ins[2]); if (B && B->device()==DeviceType::CUDA) B->toCPU(); betaPtr  = ins.size()>=3 ? getTensor(ins[2])->data_ptr<float>() : nullptr; }

        if (num_cpu_threads_ > 1) {
            kernels::simplifiedLayerNormCPUMultiThreaded(
                X->data_ptr<float>(), gammaPtr, betaPtr, Y->data_ptr<float>(), (int)M, (int)N, epsilon, num_cpu_threads_);
        } else {
            kernels::simplifiedLayerNormCPU(
                X->data_ptr<float>(), gammaPtr, betaPtr, Y->data_ptr<float>(), (int)M, (int)N, epsilon);
        }
    } else {
        // Ensure device pointers
        if (X->device() == DeviceType::CPU) X->toGPU();
        if (Y->device() == DeviceType::CPU) Y->toGPU();

        const float* gammaDev = nullptr;
        const float* betaDev  = nullptr;
        if (ins.size() >= 2) { auto G = getTensor(ins[1]); if (G && G->device()==DeviceType::CPU) G->toGPU(); gammaDev = ins.size()>=2 ? getTensor(ins[1])->data_ptr<float>() : nullptr; }
        if (ins.size() >= 3) { auto B = getTensor(ins[2]); if (B && B->device()==DeviceType::CPU) B->toGPU(); betaDev  = ins.size()>=3 ? getTensor(ins[2])->data_ptr<float>() : nullptr; }

        kernels::launchSimplifiedLayerNorm(
            X->data_ptr<float>(), gammaDev, betaDev, Y->data_ptr<float>(), (int)M, (int)N, epsilon, /*stream*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = Y;
}
