void GpuExecutor::executeWhere(const Node& node) {
    if (node.inputs().size() != 3 || node.outputs().size() != 1) {
        throw std::runtime_error("Where expects 3 inputs and 1 output");
    }

    auto condition = getTensor(node.inputs()[0]);
    auto A = getTensor(node.inputs()[1]);
    auto B = getTensor(node.inputs()[2]);

    if (A->shape() != B->shape() || A->shape() != condition->shape()) {
        throw std::runtime_error("Where: condition, X, and Y must share the same shape");
    }

    auto cond_values = tensorToBoolVector(condition);
    DataType out_dtype = promoteDataType(A->dtype(), B->dtype());
    auto output_cpu = std::make_shared<CpuTensor>(A->shape(), out_dtype);

    // TODO - refactor: GPU Where kernel not yet implemented
    // For now, compute on CPU then copy to GPU
    auto selectAndStore = [&](auto dummy) {
        using T = decltype(dummy);
        std::vector<uint8_t> cacheA;
        std::vector<uint8_t> cacheB;
        std::vector<T> convA;
        std::vector<T> convB;
        const T* a_ptr = getDataAs<T>(A, cacheA, convA);
        const T* b_ptr = getDataAs<T>(B, cacheB, convB);
        T* dst = output_cpu->data_ptr<T>();
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

    auto output = output_cpu->toGPU();
    tensors_[node.outputs()[0]] = output;
}

