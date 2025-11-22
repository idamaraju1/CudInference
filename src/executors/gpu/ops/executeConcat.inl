void GpuExecutor::executeConcat(const Node& node) {
    // Concat: Concatenate tensors along an axis
    if (node.inputs().size() < 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Concat expects at least 2 inputs and 1 output");
    }

    // Get axis
    int64_t axis = node.getIntAttr("axis", 0);

    // Get all inputs
    std::vector<std::shared_ptr<TensorBase>> inputs;
    std::vector<std::vector<int64_t>> input_shapes;

    for (const auto& input_name : node.inputs()) {
        auto input = getTensor(input_name);
        inputs.push_back(input);
        input_shapes.push_back(input->shape());
    }

    // Handle negative axis
    if (axis < 0) {
        axis += inputs[0]->ndim();
    }

    // Compute output shape
    std::vector<int64_t> output_shape = inputs[0]->shape();
    for (size_t i = 1; i < inputs.size(); ++i) {
        output_shape[axis] += inputs[i]->dim(axis);
    }

    LOG_DEBUG("Concat: axis = ", axis, ", num_inputs = ", inputs.size(), ", output shape = [",
              [&]() { std::string s; for (size_t i = 0; i < output_shape.size(); ++i) { if (i > 0) s += ", "; s += std::to_string(output_shape[i]); } return s; }(), "]");

    DataType dtype = inputs[0]->dtype();
    for (size_t idx = 1; idx < inputs.size(); ++idx) {
        dtype = promoteDataType(dtype, inputs[idx]->dtype());
    }

    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        auto& input = inputs[idx];
        if (input->ndim() != output_shape.size()) {
            throw std::runtime_error("Concat: input ranks must match");
        }
        for (size_t dim = 0; dim < output_shape.size(); ++dim) {
            if (static_cast<int64_t>(dim) == axis) continue;
            if (input->dim(dim) != output_shape[dim]) {
                throw std::runtime_error("Concat: non-axis dimensions must match");
            }
        }
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= output_shape[i];
    }

    int64_t inner = 1;
    for (size_t i = axis + 1; i < output_shape.size(); ++i) {
        inner *= output_shape[i];
    }

    int64_t total_axis = output_shape[axis];

    if (outer == 0 || inner == 0 || total_axis == 0) {
        auto output = allocateOutput(output_shape, dtype);
        tensors_[node.outputs()[0]] = output;
        return;
    }

    auto concatTyped = [&](auto dummy) {
        using T = decltype(dummy);
        auto output = allocateOutput(output_shape, dtype);
        T* dst = output->data_ptr<T>();

        // Collect GPU pointers
        std::vector<const float*> gpu_ptrs;
        for (auto& input : inputs) {
            if (input->device() == DeviceType::CUDA) {
                gpu_ptrs.push_back(input->data_ptr<float>());
            } else {
                // TODO - refactor: Need to handle CPU tensors in GPU concat
                throw std::runtime_error("Concat: All inputs must be on GPU");
            }
        }

        // Execute on GPU
        std::vector<const float*> input_ptrs;
        for (auto& input : inputs) {
            input_ptrs.push_back(input->data_ptr<float>());
        }
        launchConcatKernel(input_ptrs, output->data_ptr<float>(), input_shapes, axis, output_shape);
    };

    switch (dtype) {
        case DataType::FLOAT32:
            concatTyped(float{});
            break;
        case DataType::INT32:
            // TODO - refactor: GPU concat for INT32 not implemented
            throw std::runtime_error("Concat: INT32 dtype not yet supported on GPU");
        case DataType::INT64:
            // TODO - refactor: GPU concat for INT64 not implemented
            throw std::runtime_error("Concat: INT64 dtype not yet supported on GPU");
        case DataType::UINT8:
            // TODO - refactor: GPU concat for UINT8 not implemented
            throw std::runtime_error("Concat: UINT8 dtype not yet supported on GPU");
        default:
            throw std::runtime_error("Concat: Unsupported data type " +
                                     dataTypeToString(dtype));
    }
}

