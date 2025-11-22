void CpuExecutor::executeSimplifiedLayerNormalization(const Node& node) {
    if (node.inputs().size() < 1 || node.inputs().size() > 3 || node.outputs().size() != 1) {
        throw std::runtime_error("SimplifiedLayerNormalization expects X[, gamma][, beta] and 1 output");
    }

    auto X = getTensor(node.inputs()[0]);
    auto Y = allocateOutput(X->shape(), X->dtype());
    size_t total = X->size();
    size_t N = X->shape().back();
    size_t M = total / N;
    float epsilon = node.getFloatAttr("epsilon", 1e-5f);

    const float* gamma = nullptr;
    const float* beta = nullptr;

    if (node.inputs().size() >= 2) {
        auto G = getTensor(node.inputs()[1]);
        if (G) {
            gamma = G->data_ptr<float>();
        }
    }
    if (node.inputs().size() >= 3) {
        auto B = getTensor(node.inputs()[2]);
        if (B) {
            beta = B->data_ptr<float>();
        }
    }

    // Execute on CPU
    kernels::simplifiedLayerNormCPU(
        X->data_ptr<float>(), gamma, beta, Y->data_ptr<float>(), (int)M, (int)N, epsilon);

    tensors_[node.outputs()[0]] = Y;
}

