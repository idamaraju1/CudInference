void GpuExecutor::executeSlice(const Node& node) {
    // Slice: Extract a slice from a tensor
    if (node.inputs().size() < 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Slice expects at least 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);

    // Get starts, ends, axes, steps
    std::vector<int64_t> starts, ends, axes, steps;

    // Try to get from inputs (ONNX opset >= 10)
    if (node.inputs().size() >= 3) {
        auto starts_tensor = getTensor(node.inputs()[1]);
        auto ends_tensor = getTensor(node.inputs()[2]);

        // Move to CPU if needed
        if (starts_tensor->device() == DeviceType::CUDA) starts_tensor->toCPU();
        if (ends_tensor->device() == DeviceType::CUDA) ends_tensor->toCPU();

        // Read starts and ends
        if (starts_tensor->dtype() == DataType::INT64) {
            const int64_t* starts_data = starts_tensor->ptr_data<int64_t>();
            const int64_t* ends_data = ends_tensor->ptr_data<int64_t>();
            for (size_t i = 0; i < starts_tensor->size(); ++i) {
                starts.push_back(starts_data[i]);
                ends.push_back(ends_data[i]);
            }
        } else {
            // Read as floats and convert
            const float* starts_data = starts_tensor->ptr_data<float>();
            const float* ends_data = ends_tensor->ptr_data<float>();
            for (size_t i = 0; i < starts_tensor->size(); ++i) {
                starts.push_back(static_cast<int64_t>(starts_data[i]));
                ends.push_back(static_cast<int64_t>(ends_data[i]));
            }
        }

        if (node.inputs().size() >= 4) {
            auto axes_tensor = getTensor(node.inputs()[3]);
            if (axes_tensor->device() == DeviceType::CUDA) axes_tensor->toCPU();

            if (axes_tensor->dtype() == DataType::INT64) {
                const int64_t* axes_data = axes_tensor->ptr_data<int64_t>();
                for (size_t i = 0; i < axes_tensor->size(); ++i) {
                    axes.push_back(axes_data[i]);
                }
            } else {
                const float* axes_data = axes_tensor->ptr_data<float>();
                for (size_t i = 0; i < axes_tensor->size(); ++i) {
                    axes.push_back(static_cast<int64_t>(axes_data[i]));
                }
            }
        }

        if (node.inputs().size() >= 5) {
            auto steps_tensor = getTensor(node.inputs()[4]);
            if (steps_tensor->device() == DeviceType::CUDA) steps_tensor->toCPU();

            if (steps_tensor->dtype() == DataType::INT64) {
                const int64_t* steps_data = steps_tensor->ptr_data<int64_t>();
                for (size_t i = 0; i < steps_tensor->size(); ++i) {
                    steps.push_back(steps_data[i]);
                }
            } else {
                const float* steps_data = steps_tensor->ptr_data<float>();
                for (size_t i = 0; i < steps_tensor->size(); ++i) {
                    steps.push_back(static_cast<int64_t>(steps_data[i]));
                }
            }
        }
    }

    // Fill in defaults
    if (axes.empty()) {
        for (size_t i = 0; i < starts.size(); ++i) {
            axes.push_back(i);
        }
    }

    if (steps.empty()) {
        steps.resize(starts.size(), 1);
    }

    // Build full starts, ends, steps for all dimensions
    std::vector<int64_t> full_starts(input->ndim(), 0);
    std::vector<int64_t> full_ends(input->ndim());
    std::vector<int64_t> full_steps(input->ndim(), 1);

    for (size_t i = 0; i < input->ndim(); ++i) {
        full_ends[i] = input->dim(i);
    }

    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = axes[i];
        if (axis < 0) axis += input->ndim();

        full_starts[axis] = starts[i];
        full_ends[axis] = ends[i];
        full_steps[axis] = steps[i];

        // Handle negative indices
        if (full_starts[axis] < 0) full_starts[axis] += input->dim(axis);
        if (full_ends[axis] < 0) full_ends[axis] += input->dim(axis);

        // Clamp to valid range
        full_starts[axis] = std::max<int64_t>(0, std::min(full_starts[axis], input->dim(axis)));
        full_ends[axis] = std::max<int64_t>(0, std::min(full_ends[axis], input->dim(axis)));
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < input->ndim(); ++i) {
        int64_t dim_size = (full_ends[i] - full_starts[i] + full_steps[i] - 1) / full_steps[i];
        output_shape.push_back(std::max<int64_t>(0, dim_size));
    }

    auto output = allocateOutput(output_shape);

    launchSliceKernel(input->ptr_data<float>(), output->ptr_data<float>(), input->shape(), full_starts, full_steps, output_shape, use_cpu_fallback_, num_cpu_threads_);

    tensors_[node.outputs()[0]] = output;
}
