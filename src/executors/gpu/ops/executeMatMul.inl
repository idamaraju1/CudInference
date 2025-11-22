void GpuExecutor::executeMatMul(const Node& node) {
    // MatMul: Y = A @ B
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("MatMul expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Fast path: strictly 2D matrices
    if (A->ndim() == 2 && B->ndim() == 2) {
        int64_t M = A->dim(0);
        int64_t K = A->dim(1);
        int64_t K2 = B->dim(0);
        int64_t N = B->dim(1);

        if (K != K2) {
            throw std::runtime_error("MatMul dimension mismatch: A(" +
                                   std::to_string(M) + "," + std::to_string(K) + ") @ B(" +
                                   std::to_string(K2) + "," + std::to_string(N) + ")");
        }

        auto Y = allocateOutput({M, N});
        LOG_DEBUG("  MatMul: (", M, ", ", K, ") @ (", K, ", ", N, ") -> (", M, ", ", N, ")");

        // Execute on GPU
        kernels::launchMatMul(A->data_ptr<float>(), B->data_ptr<float>(), Y->data_ptr<float>(),
                             M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        tensors_[node.outputs()[0]] = Y;
        return;
    }

    if (A->ndim() < 2 || B->ndim() < 2) {
        throw std::runtime_error("MatMul requires both inputs to have rank >= 2");
    }

    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();

    int64_t M = shapeA[shapeA.size() - 2];
    int64_t K = shapeA.back();
    int64_t K2 = shapeB[shapeB.size() - 2];
    int64_t N = shapeB.back();

    if (K != K2) {
        throw std::runtime_error("MatMul dimension mismatch: A K=" + std::to_string(K) +
                                 " vs B K=" + std::to_string(K2));
    }

    std::vector<int64_t> batchA(shapeA.begin(), shapeA.end() - 2);
    std::vector<int64_t> batchB(shapeB.begin(), shapeB.end() - 2);
    std::vector<int64_t> batch_shape;
    if (!computeBroadcastShape(batchA, batchB, batch_shape)) {
        throw std::runtime_error("MatMul: unable to broadcast batch dimensions");
    }

    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(M);
    output_shape.push_back(N);
    auto Y = allocateOutput(output_shape);

    // TODO - refactor: Batched MatMul not yet fully implemented for GPU-only executor
    // For now, fall back to CPU computation then copy to GPU
    std::vector<uint8_t> cacheA;
    std::vector<uint8_t> cacheB;
    const float* hostA = getHostData<float>(A, cacheA);
    const float* hostB = getHostData<float>(B, cacheB);

    std::vector<float> host_output(Y->size());
    float* output_ptr = host_output.data();
    size_t matrixA_size = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t matrixB_size = static_cast<size_t>(K) * static_cast<size_t>(N);
    size_t matrixY_size = static_cast<size_t>(M) * static_cast<size_t>(N);

    size_t batch_count = computeSizeFromShape(batch_shape);
    for (size_t batch = 0; batch < batch_count; ++batch) {
        const float* A_ptr = hostA + batch * matrixA_size;
        const float* B_ptr = hostB + batch * matrixB_size;
        float* Y_ptr = output_ptr + batch * matrixY_size;

        kernels::matmulCPU(A_ptr, B_ptr, Y_ptr, M, K, N);
    }

    size_t bytes = host_output.size() * sizeof(float);
    CUDA_CHECK(cudaMemcpy(Y->data_ptr<float>(), host_output.data(), bytes, cudaMemcpyHostToDevice));

    tensors_[node.outputs()[0]] = Y;
}

