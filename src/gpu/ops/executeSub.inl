void GpuExecutor::executeSub(const Node& node) {
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

            // Move B to CPU to read the scalar value
            if (B->device() == DeviceType::CUDA) {
                B->toCPU();
            }
            float scalar = B->ptr_data<float>()[0];

            if (use_cpu_fallback_) {
                for (int i = 0; i < size; ++i) {
                    C->ptr_data<float>()[i] = A->ptr_data<float>()[i] - scalar;
                }
            } else {
                if (B->device() == DeviceType::CPU) {
                    B->toGPU();
                }
                kernels::launchSubScalar(A->ptr_data<float>(), scalar, C->ptr_data<float>(), size);
                CUDA_CHECK(cudaDeviceSynchronize());
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

    // Execute
    if (use_cpu_fallback_) {
        if (num_cpu_threads_ > 1) {
            kernels::subCPUMultiThreaded(A->ptr_data<float>(), B->ptr_data<float>(), C->ptr_data<float>(), size, num_cpu_threads_);
        } else {
            kernels::subCPU(A->ptr_data<float>(), B->ptr_data<float>(), C->ptr_data<float>(), size);
        }
    } else {
        kernels::launchSub(A->ptr_data<float>(), B->ptr_data<float>(), C->ptr_data<float>(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = C;
}
