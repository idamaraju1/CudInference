void GpuExecutor::executeSimplifiedLayerNormalization(const Node& node) {
    if (node.inputs().size() < 1 || node.inputs().size() > 3 || node.outputs().size() != 1) {
        throw std::runtime_error("SimplifiedLayerNormalization expects X[, gamma][, beta] and 1 output");
    }

    auto X = getTensor(node.inputs()[0]);
    if (X->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("SimplifiedLayerNormalization currently supports FLOAT32 tensors only");
    }

    auto Y = allocateOutput(X->shape(), X->dtype());
    size_t total = X->size();

    int64_t axis_attr = node.getIntAttr("axis", static_cast<int64_t>(X->ndim()) - 1);
    if (axis_attr < 0) {
        axis_attr += static_cast<int64_t>(X->ndim());
    }
    if (axis_attr < 0 || axis_attr >= static_cast<int64_t>(X->ndim())) {
        throw std::runtime_error("SimplifiedLayerNormalization: invalid axis attribute");
    }

    size_t N = 1;
    for (int64_t i = axis_attr; i < static_cast<int64_t>(X->ndim()); ++i) {
        N *= static_cast<size_t>(X->dim(static_cast<size_t>(i)));
    }
    if (N == 0 || total % N != 0) {
        throw std::runtime_error("SimplifiedLayerNormalization: invalid hidden size computed from axis");
    }
    size_t M = total / N;
    float epsilon = node.getFloatAttr("epsilon", 1e-5f);

    std::shared_ptr<Tensor> gamma_tensor = nullptr;
    std::shared_ptr<Tensor> beta_tensor = nullptr;
    if (node.inputs().size() >= 2) {
        gamma_tensor = getTensor(node.inputs()[1]);
    }
    if (node.inputs().size() >= 3) {
        beta_tensor = getTensor(node.inputs()[2]);
    }

    auto validateScale = [&](const std::shared_ptr<Tensor>& tensor,
                             const char* name) {
        if (!tensor) return;
        if (tensor->dtype() != DataType::FLOAT32) {
            throw std::runtime_error(std::string("SimplifiedLayerNormalization: ")
                                     + name + " must be FLOAT32");
        }
        size_t count = tensor->size();
        if (count != N) {
            throw std::runtime_error(std::string("SimplifiedLayerNormalization: ")
                                     + name + " size ("
                                     + std::to_string(count)
                                     + ") must equal normalized dimension ("
                                     + std::to_string(N) + ")");
        }
    };

    validateScale(gamma_tensor, "gamma");
    validateScale(beta_tensor, "beta");

    if (use_cpu_fallback_) {
        // Previous fallback only normalized via sum-of-squares and skipped mean
        // subtraction, which diverged sharply from ONNX Runtime. Reuse the same
        // accumulation math as the CUDA path so CPU traces line up.
        std::vector<uint8_t> x_cache;
        std::vector<uint8_t> gamma_cache;
        std::vector<uint8_t> beta_cache;
        const float* x_data = getHostData<float>(X, x_cache);
        const float* gamma_host = nullptr;
        const float* beta_host = nullptr;
        if (gamma_tensor) {
            gamma_host = getHostData<float>(gamma_tensor, gamma_cache);
        }
        if (beta_tensor) {
            beta_host = getHostData<float>(beta_tensor, beta_cache);
        }
        float* y_data = Y->data<float>();

        if (num_cpu_threads_ > 1) {
            kernels::simplifiedLayerNormCPUMultiThreaded(
                x_data, gamma_host, beta_host, y_data,
                static_cast<int>(M), static_cast<int>(N),
                epsilon, num_cpu_threads_);
        } else {
            kernels::simplifiedLayerNormCPU(
                x_data, gamma_host, beta_host, y_data,
                static_cast<int>(M), static_cast<int>(N),
                epsilon);
        }

        tensors_[node.outputs()[0]] = Y;
        return;
    } else {
        // Ensure device pointers
        if (X->device() == DeviceType::CPU) X->toGPU();
        if (Y->device() == DeviceType::CPU) Y->toGPU();

        const float* gammaDev = nullptr;
        const float* betaDev  = nullptr;
        if (gamma_tensor) {
            if (gamma_tensor->device() == DeviceType::CPU) gamma_tensor->toGPU();
            gammaDev = gamma_tensor->data<float>();
        }
        if (beta_tensor) {
            if (beta_tensor->device() == DeviceType::CPU) beta_tensor->toGPU();
            betaDev = beta_tensor->data<float>();
        }

        kernels::launchSimplifiedLayerNorm(
            X->data<float>(), gammaDev, betaDev, Y->data<float>(), (int)M, (int)N, epsilon, /*stream*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = Y;
}
