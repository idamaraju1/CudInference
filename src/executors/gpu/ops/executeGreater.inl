void GpuExecutor::executeGreater(const Node& node) {
    if (node.inputs().size() != 2 || node.outputs().size() != 1)
        throw std::runtime_error("Greater expects 2 inputs and 1 output");

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    std::vector<int64_t> output_shape;
    if (!computeBroadcastShape(A->shape(), B->shape(), output_shape))
        throw std::runtime_error("Greater: operands have incompatible shapes");

    auto output_cpu = std::make_shared<CpuTensor>(output_shape, DataType::UINT8);
    size_t out_size = computeSizeFromShape(output_shape);
    uint8_t* out_ptr = output_cpu->data_ptr<uint8_t>();

    DataType compare_dtype = promoteDataType(A->dtype(), B->dtype());

    // TODO - refactor: GPU Greater kernel not yet implemented
    // For now, compute on CPU then copy to GPU
    auto compareAndStore = [&](auto dummy) {
        using T = decltype(dummy);

        std::vector<uint8_t> cacheA, cacheB;
        std::vector<T> convA, convB;
        const T* a_ptr = getDataAs<T>(A, cacheA, convA);
        const T* b_ptr = getDataAs<T>(B, cacheB, convB);

        std::vector<T> broadA(out_size);
        std::vector<T> broadB(out_size);
        broadcastToBuffer(a_ptr, A->shape(), broadA, output_shape);
        broadcastToBuffer(b_ptr, B->shape(), broadB, output_shape);

        for (size_t i = 0; i < out_size; ++i)
            out_ptr[i] = static_cast<uint8_t>(broadA[i] > broadB[i]);
    };

    switch (compare_dtype) {
        case DataType::FLOAT32: compareAndStore(float{}); break;
        case DataType::INT64:   compareAndStore(int64_t{}); break;
        case DataType::INT32:   compareAndStore(int32_t{}); break;
        case DataType::UINT8:   compareAndStore(uint8_t{}); break;
        default:
            throw std::runtime_error("Greater: Unsupported data type " +
                                     dataTypeToString(compare_dtype));
    }

    auto output = output_cpu->toGPU();
    tensors_[node.outputs()[0]] = output;
}

