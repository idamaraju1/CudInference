void GpuExecutor::executeAdd(const Node& node) {
    // Add: C = A + B (element-wise)
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Add expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // For simplicity, require same shape (no broadcasting for now)
    if (A->shape() != B->shape()) {
        // Simple broadcasting: if B is a scalar (size 1), broadcast it
        if (B->size() == 1) {
            auto C = allocateOutput(A->shape());
            int size = A->size();

            // Move B to CPU to read the scalar value
            if (B->device() == DeviceType::CUDA) {
                B->toCPU();
            }
            float scalar = B->data_ptr<float>()[0];

            if (use_cpu_fallback_) {
                for (int i = 0; i < size; ++i) {
                    C->data_ptr<float>()[i] = A->data_ptr<float>()[i] + scalar;
                }
            } else {
                if (B->device() == DeviceType::CPU) {
                    B->toGPU();
                }
                kernels::launchAddScalar(A->data_ptr<float>(), scalar, C->data_ptr<float>(), size);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            tensors_[node.outputs()[0]] = C;
            return;
        }

        throw std::runtime_error("Add shape mismatch: " + A->shapeStr() +
                               " vs " + B->shapeStr());
    }

    auto C = allocateOutput(A->shape());
    int size = A->size();
    LOG_DEBUG("  Add: size=", size);

    // Execute
    if (use_cpu_fallback_) {
        if (num_cpu_threads_ > 1) {
            kernels::addCPUMultiThreaded(A->data_ptr<float>(), B->data_ptr<float>(), C->data_ptr<float>(), size, num_cpu_threads_);
        } else {
            kernels::addCPU(A->data_ptr<float>(), B->data_ptr<float>(), C->data_ptr<float>(), size);
        }
    } else {
        kernels::launchAdd(A->data_ptr<float>(), B->data_ptr<float>(), C->data_ptr<float>(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = C;
}
