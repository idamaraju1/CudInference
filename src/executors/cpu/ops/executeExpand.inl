void CpuExecutor::executeExpand(const Node& node) {
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Expand expects 2 inputs and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    auto shape_tensor = getTensor(node.inputs()[1]);

    auto target_shape = tensorToShapeVector(shape_tensor);
    int64_t input_rank = static_cast<int64_t>(input->ndim());
    int64_t output_rank = static_cast<int64_t>(target_shape.size());

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

    switch (input->dtype()) {
        case DataType::FLOAT32: {
            const float* src = input->data_ptr<float>();
            auto output = allocateOutput(output_shape, DataType::FLOAT32);
            broadcastCopy(src, output->data_ptr<float>(), input->shape(), output_shape);
            tensors_[node.outputs()[0]] = output;
            break;
        }
        case DataType::INT32: {
            const int32_t* src = input->data_ptr<int32_t>();
            auto output = allocateOutput(output_shape, DataType::INT32);
            broadcastCopy(src, output->data_ptr<int32_t>(), input->shape(), output_shape);
            tensors_[node.outputs()[0]] = output;
            break;
        }
        case DataType::INT64: {
            const int64_t* src = input->data_ptr<int64_t>();
            auto output = allocateOutput(output_shape, DataType::INT64);
            broadcastCopy(src, output->data_ptr<int64_t>(), input->shape(), output_shape);
            tensors_[node.outputs()[0]] = output;
            break;
        }
        case DataType::UINT8: {
            const uint8_t* src = input->data_ptr<uint8_t>();
            auto output = allocateOutput(output_shape, DataType::UINT8);
            broadcastCopy(src, output->data_ptr<uint8_t>(), input->shape(), output_shape);
            tensors_[node.outputs()[0]] = output;
            break;
        }
        default:
            throw std::runtime_error("Expand: Unsupported data type " +
                                     dataTypeToString(input->dtype()));
    }
}

