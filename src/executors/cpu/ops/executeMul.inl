void CpuExecutor::executeMul(const Node& node) {
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

        if (A_is_scalar) {
            float scalar = A->data_ptr<float>()[0];
            mulCPU(B->data_ptr<float>(), &scalar, output->data_ptr<float>(), size, 1);
        } else if (B_is_scalar) {
            float scalar = B->data_ptr<float>()[0];
            mulCPU(A->data_ptr<float>(), &scalar, output->data_ptr<float>(), size, 1);
        } else {
            mulCPU(A->data_ptr<float>(), B->data_ptr<float>(), output->data_ptr<float>(), size, 1);
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

    // Compute on CPU
    float* dst = output->data_ptr<float>();
    for (size_t i = 0; i < total; ++i) {
        dst[i] = broadcastA[i] * broadcastB[i];
    }

    tensors_[node.outputs()[0]] = output;
}

