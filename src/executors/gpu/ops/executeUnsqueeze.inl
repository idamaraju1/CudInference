void GpuExecutor::executeUnsqueeze(const Node& node) {
    // Unsqueeze: Add dimensions of size 1
    if (node.inputs().size() < 1 || node.outputs().size() != 1) {
        throw std::runtime_error("Unsqueeze expects at least 1 input and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);

    // Get axes to unsqueeze
    std::vector<int64_t> axes = node.getIntsAttr("axes");
    if (axes.empty() && node.inputs().size() >= 2) {
        // Axes might be provided as second input tensor
        auto axes_tensor = getTensor(node.inputs()[1]);
        // Move to CPU if needed
        if (axes_tensor->device() == DeviceType::CUDA) {
            axes_tensor->toCPU();
        }
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

    // Sort axes
    std::sort(axes.begin(), axes.end());

    // Compute output shape
    std::vector<int64_t> output_shape;
    size_t input_idx = 0;
    for (size_t i = 0; i < input->ndim() + axes.size(); ++i) {
        bool is_new_axis = false;
        for (auto axis : axes) {
            if (static_cast<size_t>(axis < 0 ? axis + input->ndim() + axes.size() : axis) == i) {
                is_new_axis = true;
                break;
            }
        }

        if (is_new_axis) {
            output_shape.push_back(1);
        } else {
            output_shape.push_back(input->dim(input_idx++));
        }
    }

    auto output = allocateOutput(output_shape, input->dtype());
    size_t bytes = input->size() * dataTypeSize(input->dtype());

    // Execute on GPU (just a memory copy)
    if (input->device() == DeviceType::CUDA) {
        launchUnsqueezeKernel(input->data_ptr<float>(), output->data_ptr<float>(), input->size());
    } else {
        CUDA_CHECK(cudaMemcpy(output->data_ptr<float>(), input->data_ptr<float>(), bytes, cudaMemcpyHostToDevice));
    }

    tensors_[node.outputs()[0]] = output;
}

