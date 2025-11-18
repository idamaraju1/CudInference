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

        if (use_cpu_fallback_) {
            if (num_cpu_threads_ > 1) {
                kernels::matmulCPUMultiThreaded(A->ptr_data<float>(), B->ptr_data<float>(), Y->ptr_data<float>(),
                                               M, K, N, num_cpu_threads_);
            } else {
                kernels::matmulCPU(A->ptr_data<float>(), B->ptr_data<float>(), Y->ptr_data<float>(),
                                  M, K, N);
            }
        } else {
            kernels::launchMatMul(A->ptr_data<float>(), B->ptr_data<float>(), Y->ptr_data<float>(),
                                 M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

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

    std::vector<int64_t> target_shape_A = batch_shape;
    target_shape_A.push_back(M);
    target_shape_A.push_back(K);

    std::vector<int64_t> target_shape_B = batch_shape;
    target_shape_B.push_back(K);
    target_shape_B.push_back(N);

    size_t batch_count = computeSizeFromShape(batch_shape);
    if (batch_count == 0) {
        tensors_[node.outputs()[0]] = Y;
        return;
    }

    std::vector<uint8_t> cacheA;
    std::vector<uint8_t> cacheB;
    const float* hostA = getHostData<float>(A, cacheA);
    const float* hostB = getHostData<float>(B, cacheB);

    const float* expandedA = nullptr;
    const float* expandedB = nullptr;
    std::vector<float> bufferA;
    std::vector<float> bufferB;

    if (shapeA == target_shape_A) {
        expandedA = hostA;
    } else {
        broadcastToBuffer(hostA, shapeA, bufferA, target_shape_A);
        expandedA = bufferA.data();
    }

    if (shapeB == target_shape_B) {
        expandedB = hostB;
    } else {
        broadcastToBuffer(hostB, shapeB, bufferB, target_shape_B);
        expandedB = bufferB.data();
    }

    std::vector<float> host_output(Y->size());
    float* output_ptr = host_output.data();
    size_t matrixA_size = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t matrixB_size = static_cast<size_t>(K) * static_cast<size_t>(N);
    size_t matrixY_size = static_cast<size_t>(M) * static_cast<size_t>(N);

    for (size_t batch = 0; batch < batch_count; ++batch) {
        const float* A_ptr = expandedA + batch * matrixA_size;
        const float* B_ptr = expandedB + batch * matrixB_size;
        float* Y_ptr = output_ptr + batch * matrixY_size;

        if (num_cpu_threads_ > 1) {
            kernels::matmulCPUMultiThreaded(A_ptr, B_ptr, Y_ptr, M, K, N, num_cpu_threads_);
        } else {
            kernels::matmulCPU(A_ptr, B_ptr, Y_ptr, M, K, N);
        }
    }

    size_t bytes = host_output.size() * sizeof(float);
    if (use_cpu_fallback_) {
        std::memcpy(Y->ptr_data<float>(), host_output.data(), bytes);
    } else {
        CUDA_CHECK(cudaMemcpy(Y->ptr_data<float>(), host_output.data(), bytes, cudaMemcpyHostToDevice));
    }

    tensors_[node.outputs()[0]] = Y;
}
