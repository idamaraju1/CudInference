void GpuExecutor::executeRange(const Node& node) {
    if (node.inputs().size() != 3 || node.outputs().size() != 1) {
        throw std::runtime_error("Range expects 3 inputs and 1 output");
    }

    auto start_tensor = getTensor(node.inputs()[0]);
    auto limit_tensor = getTensor(node.inputs()[1]);
    auto delta_tensor = getTensor(node.inputs()[2]);

    DataType output_dtype = start_tensor->dtype();
    if (output_dtype != DataType::FLOAT32 &&
        output_dtype != DataType::INT32 &&
        output_dtype != DataType::INT64) {
        throw std::runtime_error("Range: Unsupported start data type " +
                                 dataTypeToString(output_dtype));
    }

    double start_val = readScalarAsDouble(start_tensor);
    double limit_val = readScalarAsDouble(limit_tensor);
    double delta_val = readScalarAsDouble(delta_tensor);

    int64_t element_count = computeRangeElementCount(start_val, limit_val, delta_val);
    auto output_cpu = allocateOutput({element_count}, output_dtype);

    if (element_count > 0) {
        // TODO - refactor: GPU Range kernel not yet implemented
        // For now, compute on CPU then copy to GPU
        switch (output_dtype) {
            case DataType::FLOAT32:
                fillRangeValues<float>(output_cpu->data_ptr<float>(),
                                       element_count,
                                       static_cast<float>(start_val),
                                       static_cast<float>(delta_val));
                break;
            case DataType::INT32:
                fillRangeValues<int32_t>(output_cpu->data_ptr<int32_t>(),
                                         element_count,
                                         static_cast<int32_t>(start_val),
                                         static_cast<int32_t>(delta_val));
                break;
            case DataType::INT64:
                fillRangeValues<int64_t>(output_cpu->data_ptr<int64_t>(),
                                         element_count,
                                         static_cast<int64_t>(start_val),
                                         static_cast<int64_t>(delta_val));
                break;
            default:
                break;
        }

        auto output = output_cpu->toGPU();
        tensors_[node.outputs()[0]] = output;
    } else {
        tensors_[node.outputs()[0]] = output_cpu->toGPU();
    }
}

