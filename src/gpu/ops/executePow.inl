void GpuExecutor::executePow(const Node& node) {
    // Pow: Y = A ^ B
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Pow expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    bool B_is_scalar = (B->size() == 1);

    if (B_is_scalar || (A->shape() == B->shape())) {
        auto output = allocateOutput(A->shape());
        int64_t size = A->size();

        if (B_is_scalar) {
            // Move B to CPU to read the scalar value
            if (B->device() == DeviceType::CUDA) {
                B->toCPU();
            }
            float exponent = B->ptr_data<float>()[0];
            launchPowScalarKernel(A->ptr_data<float>(), exponent, output->ptr_data<float>(), size, use_cpu_fallback_, num_cpu_threads_);
        } else {
            launchPowKernel(A->ptr_data<float>(), B->ptr_data<float>(), output->ptr_data<float>(), size, use_cpu_fallback_, num_cpu_threads_);
        }

        tensors_[node.outputs()[0]] = output;
    } else {
        throw std::runtime_error("Pow: Broadcasting not fully supported yet");
    }
}
