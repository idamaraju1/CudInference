void GpuExecutor::executeWhere(const Node& node) {
    if (node.inputs().size() != 3 || node.outputs().size() != 1) {
        throw std::runtime_error("Where expects 3 inputs and 1 output");
    }

    auto condition = getTensor(node.inputs()[0]);
    auto A = getTensor(node.inputs()[1]);
    auto B = getTensor(node.inputs()[2]);

    LOG_DEBUG("Where: condition shape = ", condition->shapeStr(), ", A shape = ", A->shapeStr(), ", B shape = ", B->shapeStr());

    // Debug: check inputs for NaN
    if (A->dtype() == DataType::FLOAT32 && A->size() > 0 && A->size() <= 10) {
        std::vector<uint8_t> cache;
        const float* a_data = getHostData<float>(A, cache);
        for (size_t i = 0; i < A->size(); ++i) {
            if (std::isnan(a_data[i]) || std::isinf(a_data[i])) {
                LOG_DEBUG("Where input A contains NaN/Inf at index ", i, ", value = ", a_data[i], ", tensor name = ", node.inputs()[1]);
            }
        }
    }
    if (B->dtype() == DataType::FLOAT32 && B->size() > 0 && B->size() <= 10) {
        std::vector<uint8_t> cache;
        const float* b_data = getHostData<float>(B, cache);
        for (size_t i = 0; i < B->size(); ++i) {
            if (std::isnan(b_data[i]) || std::isinf(b_data[i])) {
                LOG_DEBUG("Where input B contains NaN/Inf at index ", i, ", value = ", b_data[i], ", tensor name = ", node.inputs()[2]);
            }
        }
    }

    if (A->shape() != B->shape() || A->shape() != condition->shape()) {
        throw std::runtime_error("Where: condition, X, and Y must share the same shape");
    }

    auto cond_values = tensorToBoolVector(condition);
    DataType out_dtype = promoteDataType(A->dtype(), B->dtype());
    auto output = std::make_shared<CpuTensor>(A->shape(), out_dtype);

    auto selectAndStore = [&](auto dummy) {
        using T = decltype(dummy);
        std::vector<uint8_t> cacheA;
        std::vector<uint8_t> cacheB;
        std::vector<T> convA;
        std::vector<T> convB;
        const T* a_ptr = getDataAs<T>(A, cacheA, convA);
        const T* b_ptr = getDataAs<T>(B, cacheB, convB);
        T* dst = output->data_ptr<T>();
        for (size_t i = 0; i < cond_values.size(); ++i) {
            dst[i] = cond_values[i] ? a_ptr[i] : b_ptr[i];
        }
    };

    switch (out_dtype) {
        case DataType::FLOAT32:
            selectAndStore(float{});
            break;
        case DataType::INT64:
            selectAndStore(int64_t{});
            break;
        case DataType::INT32:
            selectAndStore(int32_t{});
            break;
        case DataType::UINT8:
            selectAndStore(uint8_t{});
            break;
        default:
            throw std::runtime_error("Where: Unsupported data type " +
                                     dataTypeToString(out_dtype));
    }

    // Debug: check for NaN/Inf in output
    if (out_dtype == DataType::FLOAT32 && output->size() > 0 && output->size() <= 10) {
        std::vector<uint8_t> cache;
        const float* out_data = getHostData<float>(output, cache);
        bool has_nan = false;
        for (size_t i = 0; i < output->size(); ++i) {
            if (std::isnan(out_data[i]) || std::isinf(out_data[i])) {
                LOG_DEBUG("Where output contains NaN/Inf at index ", i, ", value = ", out_data[i]);
                has_nan = true;
            }
        }
        if (has_nan) {
            LOG_DEBUG("Where output shape = ", output->shapeStr(), ", output name = ", node.outputs()[0]);
        }
    }

    if (!use_cpu_fallback_) {
        output->toGPU();
    }

    tensors_[node.outputs()[0]] = output;
}
