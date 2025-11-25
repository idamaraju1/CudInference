void GpuExecutor::executeGemm(const Node& node) {
    // GEMM: Y = alpha * A @ B + beta * C
    // Simplified: Y = A @ B + C (assuming alpha=1, beta=1)
    if (node.inputs().size() < 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Gemm expects at least 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Check for transpose attributes
    bool transA = node.getIntAttr("transA", 0) != 0;
    bool transB = node.getIntAttr("transB", 0) != 0;
    float alpha = node.getFloatAttr("alpha", 1.0f);
    float beta = node.getFloatAttr("beta", 1.0f);

    LOG_DEBUG("  Gemm: A", A->shapeStr(), " B", B->shapeStr(),
             " transA=", transA, " transB=", transB);

    // Only support alpha=1, beta=1 for now
    if (alpha != 1.0f || beta != 1.0f) {
        throw std::runtime_error("Gemm only supports alpha=1.0 and beta=1.0");
    }

    // Determine dimensions based on transpose flags
    // Gemm: Y = alpha * op(A) @ op(B) + beta * C
    // where op(X) = X if trans=0, X^T if trans=1
    int64_t M = transA ? A->dim(1) : A->dim(0);
    int64_t K = transA ? A->dim(0) : A->dim(1);
    int64_t K_B = transB ? B->dim(1) : B->dim(0);
    int64_t N = transB ? B->dim(0) : B->dim(1);

    LOG_DEBUG("  Result dimensions: M=", M, " K=", K, " N=", N);

    if (K != K_B) {
        throw std::runtime_error("Gemm dimension mismatch: K dimensions don't match");
    }

    // Handle transpose by creating transposed copies if needed
    std::shared_ptr<Tensor> A_op = A;
    std::shared_ptr<Tensor> B_op = B;

    if (transA) {
        // Create temporary CPU tensor for transpose
        auto A_temp = std::make_shared<Tensor>(A->shape());
        if (A->device() == DeviceType::CUDA) {
            // Copy from GPU to CPU
            CUDA_CHECK(cudaMemcpy(A_temp->data<float>(), A->data<float>(),
                                 A->size() * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(A_temp->data<float>(), A->data<float>(), A->size() * sizeof(float));
        }

        A_op = std::make_shared<Tensor>(std::vector<int64_t>{M, K});
        transposeMatrix(A_temp->data<float>(), A_op->data<float>(), A->dim(0), A->dim(1), use_cpu_fallback_);
        if (!use_cpu_fallback_) A_op->toGPU();
    }

    if (transB) {
        // Create temporary CPU tensor for transpose
        auto B_temp = std::make_shared<Tensor>(B->shape());
        if (B->device() == DeviceType::CUDA) {
            // Copy from GPU to CPU
            CUDA_CHECK(cudaMemcpy(B_temp->data<float>(), B->data<float>(),
                                 B->size() * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(B_temp->data<float>(), B->data<float>(), B->size() * sizeof(float));
        }

        B_op = std::make_shared<Tensor>(std::vector<int64_t>{K, N});
        transposeMatrix(B_temp->data<float>(), B_op->data<float>(), B->dim(0), B->dim(1), use_cpu_fallback_);
        if (!use_cpu_fallback_) B_op->toGPU();
    }

    auto Y = allocateOutput({M, N});

    // GPU_PERSISTENT MODE: Keep everything on GPU
    if (exec_mode_ == ExecutionMode::GPU_PERSISTENT) {
        A_op->ensureOnGPU();
        B_op->ensureOnGPU();

        const float* d_A = A_op->deviceData<float>();
        const float* d_B = B_op->deviceData<float>();
        float* d_Y = Y->mutableDeviceData<float>();

        // TODO: Use cuBLAS with transpose flags instead of pre-transposing
        // For now, assume A_op and B_op are already correctly oriented
        kernels::launchMatMul(d_A, d_B, d_Y, M, K, N);

        // Add bias if present
        if (node.inputs().size() >= 3) {
            auto C = getTensor(node.inputs()[2]);
            C->ensureOnGPU();

            const float* d_C = C->deviceData<float>();
            kernels::launchAdd(d_Y, d_C, d_Y, Y->size());
        }

        tensors_[node.outputs()[0]] = Y;
        return;
    }

    if (use_cpu_fallback_) {
        if (num_cpu_threads_ > 1) {
            kernels::matmulCPUMultiThreaded(A_op->data<float>(), B_op->data<float>(), Y->data<float>(),
                                           M, K, N, num_cpu_threads_);
        } else {
            kernels::matmulCPU(A_op->data<float>(), B_op->data<float>(), Y->data<float>(),
                              M, K, N);
        }
    } else {
        kernels::launchMatMul(A_op->data<float>(), B_op->data<float>(), Y->data<float>(),
                             M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Add bias if present
    if (node.inputs().size() >= 3) {
        auto C = getTensor(node.inputs()[2]);
        int size = Y->size();

        if (use_cpu_fallback_) {
            if (num_cpu_threads_ > 1) {
                kernels::addCPUMultiThreaded(Y->data<float>(), C->data<float>(), Y->data<float>(), size, num_cpu_threads_);
            } else {
                kernels::addCPU(Y->data<float>(), C->data<float>(), Y->data<float>(), size);
            }
        } else {
            kernels::launchAdd(Y->data<float>(), C->data<float>(), Y->data<float>(), size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    tensors_[node.outputs()[0]] = Y;
}
