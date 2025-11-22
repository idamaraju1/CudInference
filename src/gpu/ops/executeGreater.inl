void GpuExecutor::executeGreater(const Node& node) {
    if (node.inputs().size() != 2 || node.outputs().size() != 1)
        throw std::runtime_error("Greater expects 2 inputs and 1 output");

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    std::vector<int64_t> output_shape;
    if (!computeBroadcastShape(A->shape(), B->shape(), output_shape))
        throw std::runtime_error("Greater: operands have incompatible shapes");

    auto output = std::make_shared<CpuTensor>(output_shape, DataType::UINT8);
    size_t out_size = computeSizeFromShape(output_shape);
    uint8_t* out_ptr = output->data_ptr<uint8_t>();

    DataType compare_dtype = promoteDataType(A->dtype(), B->dtype());

    auto compareAndStore = [&](auto dummy) {
        using T = decltype(dummy);

        // Get pointers safely (ensures cache vectors live long enough)
        std::vector<uint8_t> cacheA, cacheB;
        std::vector<T> convA, convB;
        const T* a_ptr = getDataAs<T>(A, cacheA, convA);
        const T* b_ptr = getDataAs<T>(B, cacheB, convB);

        // Prepare broadcasted buffers
        std::vector<T> broadA(out_size);
        std::vector<T> broadB(out_size);
        broadcastToBuffer(a_ptr, A->shape(), broadA, output_shape);
        broadcastToBuffer(b_ptr, B->shape(), broadB, output_shape);

        // Compare safely within bounds
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

    // Move to GPU *after* CPU writes are complete
    if (!use_cpu_fallback_)
        output->toGPU();

    tensors_[node.outputs()[0]] = std::move(output);
}
