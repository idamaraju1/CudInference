void CpuExecutor::executeReduceMean(const Node& node) {
    // ReduceMean: Compute mean along specified axes
    if (node.inputs().size() < 1 || node.outputs().size() != 1) {
        throw std::runtime_error("ReduceMean expects at least 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);

    // Get axes to reduce over
    std::vector<int64_t> axes = node.getIntsAttr("axes");
    if (axes.empty() && node.inputs().size() >= 2) {
        // Axes might be provided as second input tensor
        auto axes_tensor = getTensor(node.inputs()[1]);
        // Check data type and read accordingly
        if (axes_tensor->dtype() == DataType::INT64) {
            const int64_t* axes_data = axes_tensor->data_ptr<int64_t>();
            for (size_t i = 0; i < axes_tensor->size(); ++i) {
                axes.push_back(axes_data[i]);
            }
        } else {
            // Read as floats (computed tensors) and convert to int64
            const float* axes_data = axes_tensor->data_ptr<float>();
            for (size_t i = 0; i < axes_tensor->size(); ++i) {
                axes.push_back(static_cast<int64_t>(axes_data[i]));
            }
        }
    }

    if (axes.empty()) {
        // Reduce over all axes
        for (size_t i = 0; i < input->ndim(); ++i) {
            axes.push_back(i);
        }
    }

    // Handle negative axes
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += input->ndim();
        }
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    bool keepdims = node.getIntAttr("keepdims", 1) != 0;

    for (size_t i = 0; i < input->ndim(); ++i) {
        bool is_reduced = false;
        for (auto axis : axes) {
            if (static_cast<size_t>(axis) == i) {
                is_reduced = true;
                break;
            }
        }
        if (is_reduced && keepdims) {
            output_shape.push_back(1);
        } else if (!is_reduced) {
            output_shape.push_back(input->dim(i));
        }
    }

    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    auto output = allocateOutput(output_shape);

    // Execute on CPU
    reduceMeanCPU(input->data_ptr<float>(), output->data_ptr<float>(), input->shape(), axes, 1);

    tensors_[node.outputs()[0]] = output;
}

