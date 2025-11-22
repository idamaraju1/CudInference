void GpuExecutor::executeConstantOfShape(const Node& node) {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("ConstantOfShape expects 1 input and 1 output");
    }

    auto shape_tensor = getTensor(node.inputs()[0]);
    std::vector<int64_t> output_shape = tensorToShapeVector(shape_tensor);

    for (auto dim : output_shape) {
        if (dim < 0) {
            throw std::runtime_error("ConstantOfShape: negative dimensions are not supported");
        }
    }

    auto value_attr = node.getTensorAttr("value");
    DataType output_dtype = value_attr ? value_attr->dtype() : DataType::FLOAT32;

    size_t total_elems = 1;
    for (auto dim : output_shape) {
        total_elems *= dim;
    }

    // TODO - refactor: GPU ConstantOfShape kernel not yet implemented
    // For now, compute on CPU then copy to GPU
    auto output_cpu = std::make_shared<CpuTensor>(output_shape, output_dtype);

    auto fillDefault = [&](auto dummy) {
        using T = decltype(dummy);
        T value = static_cast<T>(0);
        if (value_attr) {
            if (value_attr->size() != 1) {
                throw std::runtime_error("ConstantOfShape: value attribute must be a scalar");
            }
            value = value_attr->data_ptr<T>()[0];
        }
        fillTensorWithValue(output_cpu->data_ptr<T>(), total_elems, value);
    };

    switch (output_dtype) {
        case DataType::FLOAT32:
            fillDefault(float{});
            break;
        case DataType::INT32:
            fillDefault(int32_t{});
            break;
        case DataType::INT64:
            fillDefault(int64_t{});
            break;
        case DataType::UINT8:
            fillDefault(uint8_t{});
            break;
        default:
            throw std::runtime_error("ConstantOfShape: Unsupported data type " +
                                     dataTypeToString(output_dtype));
    }

    auto output = output_cpu->toGPU();
    tensors_[node.outputs()[0]] = output;
}

