void GpuExecutor::executeExpand(const Node& node) {
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Expand expects 2 inputs and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    auto shape_tensor = getTensor(node.inputs()[1]);

    LOG_DEBUG("Expand: input tensor name = '", node.inputs()[0], "', shape tensor name = '", node.inputs()[1], "'");
    LOG_DEBUG("Expand: shape_tensor = ", shape_tensor->shapeStr(), ", size = ", shape_tensor->size(),
              ", dtype = ", dataTypeToString(shape_tensor->dtype()));

    // Debug: print shape tensor values
    if (shape_tensor->size() > 0 && shape_tensor->size() <= 10) {
        std::vector<uint8_t> cache;
        std::string vals_str = "shape_tensor values: [";
        if (shape_tensor->dtype() == DataType::FLOAT32) {
            const float* data = getHostData<float>(shape_tensor, cache);
            for (size_t i = 0; i < shape_tensor->size(); ++i) {
                if (i > 0) vals_str += ", ";
                vals_str += std::to_string(data[i]);
            }
        }
        vals_str += "]";
        LOG_DEBUG(vals_str);
    }

    auto target_shape = tensorToShapeVector(shape_tensor);
    int64_t input_rank = static_cast<int64_t>(input->ndim());
    int64_t output_rank = static_cast<int64_t>(target_shape.size());

    LOG_DEBUG("Expand: input shape = ", input->shapeStr(), " -> target shape = [",
              [&]() { std::string s; for (size_t i = 0; i < target_shape.size(); ++i) { if (i > 0) s += ", "; s += std::to_string(target_shape[i]); } return s; }(), "]");

    if (output_rank < input_rank) {
        throw std::runtime_error("Expand: target rank must be >= input rank");
    }

    // Compute the broadcasted output shape following NumPy-style rules.
    std::vector<int64_t> output_shape(output_rank, 1);
    int64_t offset = output_rank - input_rank;

    for (int64_t idx = 0; idx < output_rank; ++idx) {
        int64_t target_dim = target_shape[idx];
        int64_t input_dim = (idx < offset) ? 1 : input->dim(idx - offset);

        int64_t result_dim = 0;
        if (target_dim == 0) {
            if (input_dim != 0 && input_dim != 1) {
                throw std::runtime_error(
                    "Expand: incompatible dimension " + std::to_string(input_dim) +
                    " -> 0 at axis " + std::to_string(idx));
            }
            result_dim = 0;
        } else if (input_dim == 1) {
            result_dim = target_dim;
        } else if (target_dim == 1) {
            result_dim = input_dim;
        } else if (input_dim == target_dim) {
            result_dim = target_dim;
        } else {
            throw std::runtime_error("Expand: incompatible dimension at axis " +
                                     std::to_string(idx) + " (" +
                                     std::to_string(input_dim) + " vs " +
                                     std::to_string(target_dim) + ")");
        }

        output_shape[idx] = result_dim;
    }

    LOG_DEBUG("Expand: computed broadcast shape = [",
              [&]() { std::string s; for (size_t i = 0; i < output_shape.size(); ++i) { if (i > 0) s += ", "; s += std::to_string(output_shape[i]); } return s; }(), "]");

    switch (input->dtype()) {
        case DataType::FLOAT32: {
            std::vector<uint8_t> cache;
            const float* src = getHostData<float>(input, cache);
            auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32);
            broadcastCopy(src, output->ptr_data<float>(), input->shape(), output_shape);
            if (!use_cpu_fallback_) output->toGPU();
            tensors_[node.outputs()[0]] = output;
            break;
        }
        case DataType::INT32: {
            std::vector<uint8_t> cache;
            const int32_t* src = getHostData<int32_t>(input, cache);
            auto output = std::make_shared<Tensor>(output_shape, DataType::INT32);
            broadcastCopy(src, output->ptr_data<int32_t>(), input->shape(), output_shape);
            if (!use_cpu_fallback_) output->toGPU();
            tensors_[node.outputs()[0]] = output;
            break;
        }
        case DataType::INT64: {
            std::vector<uint8_t> cache;
            const int64_t* src = getHostData<int64_t>(input, cache);
            auto output = std::make_shared<Tensor>(output_shape, DataType::INT64);
            broadcastCopy(src, output->ptr_data<int64_t>(), input->shape(), output_shape);
            if (!use_cpu_fallback_) output->toGPU();
            tensors_[node.outputs()[0]] = output;
            break;
        }
        case DataType::UINT8: {
            std::vector<uint8_t> cache;
            const uint8_t* src = getHostData<uint8_t>(input, cache);
            auto output = std::make_shared<Tensor>(output_shape, DataType::UINT8);
            broadcastCopy(src, output->ptr_data<uint8_t>(), input->shape(), output_shape);
            if (!use_cpu_fallback_) output->toGPU();
            tensors_[node.outputs()[0]] = output;
            break;
        }
        default:
            throw std::runtime_error("Expand: Unsupported data type " +
                                     dataTypeToString(input->dtype()));
    }
}
