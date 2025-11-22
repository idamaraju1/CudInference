void CpuExecutor::executeGather(const Node& node) {
    // Gather: output = data[indices] along specified axis
    // Inputs: data, indices
    // Attributes: axis (default 0)
    // Output: gathered tensor

    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Gather expects 2 inputs and 1 output");
    }

    auto data = getTensor(node.inputs()[0]);
    auto indices_tensor = getTensor(node.inputs()[1]);

    // Get axis attribute (default 0)
    int64_t axis = node.getIntAttr("axis", 0);

    // Handle negative axis
    if (axis < 0) {
        axis += data->ndim();
    }

    if (axis < 0 || axis >= static_cast<int64_t>(data->ndim())) {
        throw std::runtime_error("Gather axis out of range");
    }

    // Materialize indices as int64_t
    size_t indices_count = indices_tensor->size();
    std::vector<int64_t> host_indices(indices_count, 0);

    switch (indices_tensor->dtype()) {
        case DataType::INT64: {
            const int64_t* src = indices_tensor->data_ptr<int64_t>();
            std::copy(src, src + indices_count, host_indices.begin());
            break;
        }
        case DataType::INT32: {
            const int32_t* src = indices_tensor->data_ptr<int32_t>();
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(src[i]);
            }
            break;
        }
        case DataType::UINT8: {
            const uint8_t* src = indices_tensor->data_ptr<uint8_t>();
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(src[i]);
            }
            break;
        }
        default: {
            const float* src = indices_tensor->data_ptr<float>();
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(std::llround(src[i]));
            }
            break;
        }
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < axis; ++i) {
        output_shape.push_back(data->dim(i));
    }
    for (size_t i = 0; i < indices_tensor->ndim(); ++i) {
        output_shape.push_back(indices_tensor->dim(i));
    }
    for (size_t i = axis + 1; i < data->ndim(); ++i) {
        output_shape.push_back(data->dim(i));
    }

    // Allocate output
    auto output = allocateOutput(output_shape);

    // Compute dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= data->dim(i);
    }

    int64_t axis_dim_data = data->dim(axis);
    int64_t inner_size = 1;
    for (size_t i = axis + 1; i < data->ndim(); ++i) {
        inner_size *= data->dim(i);
    }

    int64_t axis_dim_indices = static_cast<int64_t>(host_indices.size());
    int64_t total_size = outer_size * axis_dim_indices * inner_size;

    if (outer_size == 0 || axis_dim_indices == 0 || inner_size == 0 || total_size == 0) {
        tensors_[node.outputs()[0]] = output;
        return;
    }

    // Execute on CPU
    gatherCPU(
        data->data_ptr<float>(),
        host_indices.data(),
        output->data_ptr<float>(),
        axis_dim_data,
        axis_dim_indices,
        outer_size,
        inner_size,
        total_size,
        1
    );

    tensors_[node.outputs()[0]] = output;
}

