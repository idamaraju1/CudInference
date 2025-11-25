void CpuExecutor::executeMatMul(const Node& node) {
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

        if (num_cpu_threads_ > 1) {
            kernels::matmulCPUMultiThreaded(A->data<float>(), B->data<float>(), Y->data<float>(),
                                           M, K, N, num_cpu_threads_);
        } else {
            kernels::matmulCPU(A->data<float>(), B->data<float>(), Y->data<float>(),
                              M, K, N);
        }

        tensors_[node.outputs()[0]] = Y;
        return;
    }

    throw std::runtime_error("MatMul: batched operations not yet supported in CPU executor");
}
