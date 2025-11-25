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

            // GPU_PERSISTENT MODE: Scalar broadcast on GPU
            if (exec_mode_ == ExecutionMode::GPU_PERSISTENT) {
                A->ensureOnGPU();
                B->ensureOnGPU();

                const float* d_A = A->deviceData<float>();
                float* d_C = C->mutableDeviceData<float>();

                // Read scalar from GPU (need single value copy)
                float scalar;
                CUDA_CHECK(cudaMemcpy(&scalar, B->deviceData<float>(), sizeof(float), cudaMemcpyDeviceToHost));

                kernels::launchAddScalar(d_A, scalar, d_C, size);

                tensors_[node.outputs()[0]] = C;
                return;
            }

            // Move B to CPU to read the scalar value
            if (B->device() == DeviceType::CUDA) {
                B->toCPU();
            }
            float scalar = B->data<float>()[0];

            if (use_cpu_fallback_) {
                for (int i = 0; i < size; ++i) {
                    C->data<float>()[i] = A->data<float>()[i] + scalar;
                }
            } else {
                if (B->device() == DeviceType::CPU) {
                    B->toGPU();
                }
                kernels::launchAddScalar(A->data<float>(), scalar, C->data<float>(), size);
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

    // GPU_PERSISTENT MODE: Element-wise add on GPU
    if (exec_mode_ == ExecutionMode::GPU_PERSISTENT) {
        A->ensureOnGPU();
        B->ensureOnGPU();

        const float* d_A = A->deviceData<float>();
        const float* d_B = B->deviceData<float>();
        float* d_C = C->mutableDeviceData<float>();

        kernels::launchAdd(d_A, d_B, d_C, size);

        tensors_[node.outputs()[0]] = C;
        return;
    }

    // Execute
    if (use_cpu_fallback_) {
        if (num_cpu_threads_ > 1) {
            kernels::addCPUMultiThreaded(A->data<float>(), B->data<float>(), C->data<float>(), size, num_cpu_threads_);
        } else {
            kernels::addCPU(A->data<float>(), B->data<float>(), C->data<float>(), size);
        }
    } else {
        kernels::launchAdd(A->data<float>(), B->data<float>(), C->data<float>(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = C;
}
