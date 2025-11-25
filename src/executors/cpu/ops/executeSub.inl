void CpuExecutor::executeSub(const Node& node) {
    // Sub: Y = A - B
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Sub expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Check for scalar broadcasting
    bool is_scalar_B = (B->size() == 1);

    if (is_scalar_B) {
        // B is scalar: Y = A - scalar
        float scalar = B->data<float>()[0];
        auto Y = allocateOutput(A->shape());

        int size = A->size();
        const float* A_data = A->data<float>();
        float* Y_data = Y->data<float>();

        if (num_cpu_threads_ > 1) {
            #pragma omp parallel for num_threads(num_cpu_threads_)
            for (int i = 0; i < size; ++i) {
                Y_data[i] = A_data[i] - scalar;
            }
        } else {
            for (int i = 0; i < size; ++i) {
                Y_data[i] = A_data[i] - scalar;
            }
        }

        tensors_[node.outputs()[0]] = Y;
        return;
    }

    // Element-wise subtraction (same shape)
    if (A->shape() == B->shape()) {
        auto Y = allocateOutput(A->shape());
        int size = A->size();

        if (num_cpu_threads_ > 1) {
            kernels::subCPUMultiThreaded(A->data<float>(), B->data<float>(), Y->data<float>(), size, num_cpu_threads_);
        } else {
            kernels::subCPU(A->data<float>(), B->data<float>(), Y->data<float>(), size);
        }

        tensors_[node.outputs()[0]] = Y;
        return;
    }

    throw std::runtime_error("Sub: complex broadcasting not yet supported in CPU executor");
}
