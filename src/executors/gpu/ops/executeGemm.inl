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
    int64_t M = transA ? A->dim(1) : A->dim(0);
    int64_t K = transA ? A->dim(0) : A->dim(1);
    int64_t K_B = transB ? B->dim(1) : B->dim(0);
    int64_t N = transB ? B->dim(0) : B->dim(1);

    LOG_DEBUG("  Result dimensions: M=", M, " K=", K, " N=", N);

    if (K != K_B) {
        throw std::runtime_error("Gemm dimension mismatch: K dimensions don't match");
    }

    // Handle transpose by creating transposed copies if needed
    std::shared_ptr<TensorBase> A_op = A;
    std::shared_ptr<TensorBase> B_op = B;

    if (transA) {
        // Transpose on CPU then copy to GPU
        std::vector<uint8_t> cache;
        const float* a_data = getHostData<float>(A, cache);
        
        auto A_op_cpu = std::make_shared<CpuTensor>(std::vector<int64_t>{M, K}, A->dtype());
        transposeMatrix(a_data, A_op_cpu->data_ptr<float>(), A->dim(0), A->dim(1));
        A_op = A_op_cpu->toGPU();
    }

    if (transB) {
        // Transpose on CPU then copy to GPU
        std::vector<uint8_t> cache;
        const float* b_data = getHostData<float>(B, cache);
        
        auto B_op_cpu = std::make_shared<CpuTensor>(std::vector<int64_t>{K, N}, B->dtype());
        transposeMatrix(b_data, B_op_cpu->data_ptr<float>(), B->dim(0), B->dim(1));
        B_op = B_op_cpu->toGPU();
    }

    auto Y = allocateOutput({M, N});

    // Execute on GPU
    kernels::launchMatMul(A_op->data_ptr<float>(), B_op->data_ptr<float>(), Y->data_ptr<float>(),
                         M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Add bias if present
    if (node.inputs().size() >= 3) {
        auto C = getTensor(node.inputs()[2]);
        int size = Y->size();

        // Execute on GPU
        kernels::launchAdd(Y->data_ptr<float>(), C->data_ptr<float>(), Y->data_ptr<float>(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = Y;
}

