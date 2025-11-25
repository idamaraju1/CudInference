void GpuExecutor::executeCast(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Cast expects 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);

    int64_t to_attr = node.getIntAttr("to", -1);
    if (to_attr == -1) {
        to_attr = node.getIntAttr("dtype", -1);
    }
    if (to_attr == -1) {
        throw std::runtime_error("Cast: missing 'to' attribute");
    }

    DataType target_dtype = mapONNXTypeToDataType(static_cast<int>(to_attr));
    if (target_dtype == DataType::FLOAT16) {
        throw std::runtime_error("Cast: FLOAT16 output is not supported yet");
    }

    auto output = std::make_shared<Tensor>(input->shape(), target_dtype);
    size_t count = input->size();
    std::vector<uint8_t> host_cache;

    switch (input->dtype()) {
        case DataType::FLOAT32: {
            const float* src = getHostData<float>(input, host_cache);
            dispatchCastToTarget(src, target_dtype, output, count);
            break;
        }
        case DataType::INT32: {
            const int32_t* src = getHostData<int32_t>(input, host_cache);
            dispatchCastToTarget(src, target_dtype, output, count);
            break;
        }
        case DataType::INT64: {
            const int64_t* src = getHostData<int64_t>(input, host_cache);
            dispatchCastToTarget(src, target_dtype, output, count);
            break;
        }
        case DataType::UINT8: {
            const uint8_t* src = getHostData<uint8_t>(input, host_cache);
            dispatchCastToTarget(src, target_dtype, output, count);
            break;
        }
        default:
            throw std::runtime_error("Cast: Unsupported input dtype " +
                                     dataTypeToString(input->dtype()));
    }

    if (!use_cpu_fallback_) {
        output->toGPU();
    }

    tensors_[node.outputs()[0]] = output;
}
