void GpuExecutor::executeTranspose(const Node& node) {
    // Transpose: Permute tensor dimensions
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Transpose expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);

    // Get permutation
    std::vector<int64_t> perm_int64 = node.getIntsAttr("perm");
    std::vector<int> perm;

    if (perm_int64.empty()) {
        // Default: reverse dimensions
        for (int i = input->ndim() - 1; i >= 0; --i) {
            perm.push_back(i);
        }
    } else {
        for (auto p : perm_int64) {
            perm.push_back(static_cast<int>(p));
        }
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    for (auto p : perm) {
        output_shape.push_back(input->dim(p));
    }

    auto output = allocateOutput(output_shape);

    launchTransposeKernel(input->data<float>(), output->data<float>(), input->shape(), perm, use_cpu_fallback_, num_cpu_threads_);

    tensors_[node.outputs()[0]] = output;
}
