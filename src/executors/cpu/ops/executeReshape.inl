void CpuExecutor::executeReshape(const Node& node) {
    // Reshape: Change tensor shape without changing data
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Reshape expects 2 inputs and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);
    auto shape_tensor = getTensor(node.inputs()[1]);

    // Extract new shape
    std::vector<int64_t> new_shape;
    // Check data type and read accordingly
    if (shape_tensor->dtype() == DataType::INT64) {
        const int64_t* shape_data = shape_tensor->data_ptr<int64_t>();
        for (size_t i = 0; i < shape_tensor->size(); ++i) {
            new_shape.push_back(shape_data[i]);
        }
    } else {
        // Read as floats (computed tensors like Shape output) and convert to int64
        const float* shape_data = shape_tensor->data_ptr<float>();
        for (size_t i = 0; i < shape_tensor->size(); ++i) {
            new_shape.push_back(static_cast<int64_t>(shape_data[i]));
        }
    }

    // Handle -1 in shape (infer dimension)
    int64_t infer_idx = -1;
    int64_t known_size = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_idx != -1) {
                throw std::runtime_error("Reshape: Only one dimension can be -1");
            }
            infer_idx = i;
        } else if (new_shape[i] == 0) {
            // 0 means copy from input shape
            new_shape[i] = input->dim(i);
            known_size *= new_shape[i];
        } else {
            known_size *= new_shape[i];
        }
    }

    if (infer_idx != -1) {
        new_shape[infer_idx] = input->size() / known_size;
    }

    auto output = allocateOutput(new_shape);

    // Execute on CPU (just a memory copy)
    std::memcpy(output->data_ptr<float>(), input->data_ptr<float>(), input->size() * sizeof(float));

    tensors_[node.outputs()[0]] = output;
}

