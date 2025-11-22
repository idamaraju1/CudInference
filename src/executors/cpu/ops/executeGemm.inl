void CpuExecutor::executeGemm(const Node& node) {
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
        // Create temporary CPU tensor for transpose
        auto A_temp = std::make_shared<CpuTensor>(A->shape(), A->dtype());
        std::memcpy(A_temp->data_ptr<float>(), A->data_ptr<float>(), A->size() * sizeof(float));

        A_op = std::make_shared<CpuTensor>(std::vector<int64_t>{M, K}, A->dtype());
        transposeMatrix(A_temp->data_ptr<float>(), A_op->data_ptr<float>(), A->dim(0), A->dim(1));
    }

    if (transB) {
        // Create temporary CPU tensor for transpose
        auto B_temp = std::make_shared<CpuTensor>(B->shape(), B->dtype());
        std::memcpy(B_temp->data_ptr<float>(), B->data_ptr<float>(), B->size() * sizeof(float));

        B_op = std::make_shared<CpuTensor>(std::vector<int64_t>{K, N}, B->dtype());
        transposeMatrix(B_temp->data_ptr<float>(), B_op->data_ptr<float>(), B->dim(0), B->dim(1));
    }

    auto Y = allocateOutput({M, N});

    // Execute on CPU
    kernels::matmulCPU(A_op->data_ptr<float>(), B_op->data_ptr<float>(), Y->data_ptr<float>(),
                      M, K, N);

    // Add bias if present
    if (node.inputs().size() >= 3) {
        auto C = getTensor(node.inputs()[2]);
        int size = Y->size();

        // Execute on CPU
        kernels::addCPU(Y->data_ptr<float>(), C->data_ptr<float>(), Y->data_ptr<float>(), size);
    }

    tensors_[node.outputs()[0]] = Y;
}

