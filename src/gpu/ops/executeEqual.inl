void GpuExecutor::executeEqual(const Node& node) {
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Equal expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    if (A->shape() != B->shape()) {
        throw std::runtime_error("Equal: operands must have the same shape");
    }

    size_t count = A->size();
    auto output = allocateOutput(A->shape(), DataType::UINT8);

    // Always compute on CPU for now
    if (!use_cpu_fallback_) {
        output->toCPU();
    }

    uint8_t* out_ptr = output->data_ptr<uint8_t>();
    std::vector<uint8_t> cacheA;
    std::vector<uint8_t> cacheB;

    auto compareDifferentTypes = [&]() {
        std::vector<double> bufA;
        std::vector<double> bufB;

        switch (A->dtype()) {
            case DataType::FLOAT32:
                convertToDoubleBuffer(getHostData<float>(A, cacheA), count, bufA);
                break;
            case DataType::INT32:
                convertToDoubleBuffer(getHostData<int32_t>(A, cacheA), count, bufA);
                break;
            case DataType::INT64:
                convertToDoubleBuffer(getHostData<int64_t>(A, cacheA), count, bufA);
                break;
            case DataType::UINT8:
                convertToDoubleBuffer(getHostData<uint8_t>(A, cacheA), count, bufA);
                break;
            default:
                throw std::runtime_error("Equal: Unsupported data type " +
                                         dataTypeToString(A->dtype()));
        }

        switch (B->dtype()) {
            case DataType::FLOAT32:
                convertToDoubleBuffer(getHostData<float>(B, cacheB), count, bufB);
                break;
            case DataType::INT32:
                convertToDoubleBuffer(getHostData<int32_t>(B, cacheB), count, bufB);
                break;
            case DataType::INT64:
                convertToDoubleBuffer(getHostData<int64_t>(B, cacheB), count, bufB);
                break;
            case DataType::UINT8:
                convertToDoubleBuffer(getHostData<uint8_t>(B, cacheB), count, bufB);
                break;
            default:
                throw std::runtime_error("Equal: Unsupported data type " +
                                         dataTypeToString(B->dtype()));
        }

        for (size_t i = 0; i < count; ++i) {
            out_ptr[i] = static_cast<uint8_t>(bufA[i] == bufB[i]);
        }
    };

    if (A->dtype() == B->dtype()) {
        switch (A->dtype()) {
            case DataType::FLOAT32: {
                const float* a_ptr = getHostData<float>(A, cacheA);
                const float* b_ptr = getHostData<float>(B, cacheB);
                elementwiseCompare(a_ptr, b_ptr, out_ptr, count, [](float a, float b) { return a == b; });
                break;
            }
            case DataType::INT32: {
                const int32_t* a_ptr = getHostData<int32_t>(A, cacheA);
                const int32_t* b_ptr = getHostData<int32_t>(B, cacheB);
                elementwiseCompare(a_ptr, b_ptr, out_ptr, count, [](int32_t a, int32_t b) { return a == b; });
                break;
            }
            case DataType::INT64: {
                const int64_t* a_ptr = getHostData<int64_t>(A, cacheA);
                const int64_t* b_ptr = getHostData<int64_t>(B, cacheB);
                elementwiseCompare(a_ptr, b_ptr, out_ptr, count, [](int64_t a, int64_t b) { return a == b; });
                break;
            }
            case DataType::UINT8: {
                const uint8_t* a_ptr = getHostData<uint8_t>(A, cacheA);
                const uint8_t* b_ptr = getHostData<uint8_t>(B, cacheB);
                elementwiseCompare(a_ptr, b_ptr, out_ptr, count, [](uint8_t a, uint8_t b) { return a == b; });
                break;
            }
            default:
                throw std::runtime_error("Equal: Unsupported data type " +
                                         dataTypeToString(A->dtype()));
        }
    } else {
        compareDifferentTypes();
    }

    if (!use_cpu_fallback_) {
        output->toGPU();
    }

    tensors_[node.outputs()[0]] = output;
}
