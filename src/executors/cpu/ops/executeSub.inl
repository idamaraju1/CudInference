void CpuExecutor::executeSub(const Node& node) {
    // Sub: C = A - B (element-wise)
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Sub expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Require same shape (no general broadcasting yet)
    if (A->shape() != B->shape()) {
        // Simple broadcasting: if B is a scalar (size 1), broadcast it
        if (B->size() == 1) {
            auto C = allocateOutput(A->shape());
            int size = A->size();

            float scalar = B->data_ptr<float>()[0];
            
            for (int i = 0; i < size; ++i) {
                C->data_ptr<float>()[i] = A->data_ptr<float>()[i] - scalar;
            }

            tensors_[node.outputs()[0]] = C;
            return;
        }

        throw std::runtime_error("Sub shape mismatch: " + A->shapeStr() +
                                 " vs " + B->shapeStr());
    }

    auto C = allocateOutput(A->shape());
    int size = A->size();
    LOG_DEBUG("  Sub: size=", size);

    // Execute on CPU
    kernels::subCPU(A->data_ptr<float>(), B->data_ptr<float>(), C->data_ptr<float>(), size);

    tensors_[node.outputs()[0]] = C;
}

