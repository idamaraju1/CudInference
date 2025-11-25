void GpuExecutor::executeMul(const Node& node) {
    // Mul: Y = A * B (element-wise or with broadcasting)
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Mul expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Check if one is a scalar
    bool A_is_scalar = (A->size() == 1);
    bool B_is_scalar = (B->size() == 1);

    if (A_is_scalar || B_is_scalar || (A->shape() == B->shape())) {
        // Element-wise or scalar multiplication
        auto output_shape = A_is_scalar ? B->shape() : A->shape();
        auto output = allocateOutput(output_shape);
        int64_t size = output->size();

        // GPU_PERSISTENT MODE
        if (exec_mode_ == ExecutionMode::GPU_PERSISTENT) {
            if (A_is_scalar) {
                float scalar;
                A->ensureOnGPU();
                B->ensureOnGPU();
                CUDA_CHECK(cudaMemcpy(&scalar, A->deviceData<float>(), sizeof(float), cudaMemcpyDeviceToHost));
                launchMulScalarKernel(B->deviceData<float>(), scalar, output->mutableDeviceData<float>(), size, false, num_cpu_threads_);
            } else if (B_is_scalar) {
                float scalar;
                A->ensureOnGPU();
                B->ensureOnGPU();
                CUDA_CHECK(cudaMemcpy(&scalar, B->deviceData<float>(), sizeof(float), cudaMemcpyDeviceToHost));
                launchMulScalarKernel(A->deviceData<float>(), scalar, output->mutableDeviceData<float>(), size, false, num_cpu_threads_);
            } else {
                A->ensureOnGPU();
                B->ensureOnGPU();
                launchMulKernel(A->deviceData<float>(), B->deviceData<float>(), output->mutableDeviceData<float>(), size, false, num_cpu_threads_);
            }
            tensors_[node.outputs()[0]] = output;
            return;
        }

        if (A_is_scalar) {
            // Move A to CPU to read the scalar value
            if (A->device() == DeviceType::CUDA) {
                A->toCPU();
            }
            float scalar = A->data<float>()[0];
            launchMulScalarKernel(B->data<float>(), scalar, output->data<float>(), size, use_cpu_fallback_, num_cpu_threads_);
        } else if (B_is_scalar) {
            // Move B to CPU to read the scalar value
            if (B->device() == DeviceType::CUDA) {
                B->toCPU();
            }
            float scalar = B->data<float>()[0];
            launchMulScalarKernel(A->data<float>(), scalar, output->data<float>(), size, use_cpu_fallback_, num_cpu_threads_);
        } else {
            launchMulKernel(A->data<float>(), B->data<float>(), output->data<float>(), size, use_cpu_fallback_, num_cpu_threads_);
        }

        tensors_[node.outputs()[0]] = output;
        return;
    }

    // General NumPy-style broadcasting path
    std::vector<int64_t> output_shape;
    if (!computeBroadcastShape(A->shape(), B->shape(), output_shape)) {
        throw std::runtime_error("Mul: incompatible shapes for broadcasting");
    }

    auto output = allocateOutput(output_shape);
    size_t total = output->size();

    if (total == 0) {
        tensors_[node.outputs()[0]] = output;
        return;
    }

    std::vector<uint8_t> cacheA;
    std::vector<uint8_t> cacheB;
    const float* dataA = getHostData<float>(A, cacheA);
    const float* dataB = getHostData<float>(B, cacheB);

    const float* broadcastA = nullptr;
    const float* broadcastB = nullptr;
    std::vector<float> expandedA;
    std::vector<float> expandedB;

    if (A->shape() == output_shape) {
        broadcastA = dataA;
    } else {
        broadcastToBuffer(dataA, A->shape(), expandedA, output_shape);
        broadcastA = expandedA.data();
    }

    if (B->shape() == output_shape) {
        broadcastB = dataB;
    } else {
        broadcastToBuffer(dataB, B->shape(), expandedB, output_shape);
        broadcastB = expandedB.data();
    }

    if (use_cpu_fallback_) {
        float* dst = output->data<float>();
        for (size_t i = 0; i < total; ++i) {
            dst[i] = broadcastA[i] * broadcastB[i];
        }
    } else {
        std::vector<float> host_output(total);
        for (size_t i = 0; i < total; ++i) {
            host_output[i] = broadcastA[i] * broadcastB[i];
        }
        CUDA_CHECK(cudaMemcpy(output->data<float>(), host_output.data(),
                              total * sizeof(float), cudaMemcpyHostToDevice));
    }

    tensors_[node.outputs()[0]] = output;
}
