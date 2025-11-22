void GpuExecutor::executeReduceSum(const Node& node) {
    if (node.inputs().empty() || node.inputs().size() > 2 || node.outputs().size() != 1) {
        throw std::runtime_error("ReduceSum expects 1 or 2 inputs and 1 output");
    }

    auto input = getTensor(node.inputs()[0]);

    // Determine axes
    std::vector<int64_t> axes = node.getIntsAttr("axes");
    if (axes.empty() && node.inputs().size() >= 2) {
        auto axes_tensor = getTensor(node.inputs()[1]);
        if (axes_tensor->device() == DeviceType::CUDA) {
            axes_tensor->toCPU();
        }
        if (axes_tensor->dtype() == DataType::INT64) {
            const int64_t* axis_data = axes_tensor->data<int64_t>();
            for (size_t i = 0; i < axes_tensor->size(); ++i) {
                axes.push_back(axis_data[i]);
            }
        } else {
            const float* axis_data = axes_tensor->data<float>();
            for (size_t i = 0; i < axes_tensor->size(); ++i) {
                axes.push_back(static_cast<int64_t>(axis_data[i]));
            }
        }
    }

    if (axes.empty()) {
        axes.resize(input->ndim());
        std::iota(axes.begin(), axes.end(), 0);
    }

    // Normalize axes
    int64_t ndim = static_cast<int64_t>(input->ndim());
    std::vector<bool> reduce_mask(ndim, false);
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            throw std::runtime_error("ReduceSum: axis out of range");
        }
        reduce_mask[axis] = true;
    }

    bool keepdims = node.getIntAttr("keepdims", 1) != 0;

    std::vector<int64_t> output_shape;
    for (int64_t dim = 0; dim < ndim; ++dim) {
        if (reduce_mask[dim]) {
            if (keepdims) {
                output_shape.push_back(1);
            }
        } else {
            output_shape.push_back(input->dim(dim));
        }
    }

    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    size_t output_size = computeSizeFromShape(output_shape);
    std::vector<float> host_output(output_size, 0.0f);

    std::vector<int64_t> input_strides = computeStrides(input->shape());
    std::vector<int64_t> output_strides = computeStrides(output_shape);

    std::vector<uint8_t> cache;

    size_t total_input = input->size();

    // Handle different data types
    if (input->dtype() == DataType::INT64) {
        const int64_t* host_data = getHostData<int64_t>(input, cache);
        for (size_t idx = 0; idx < total_input; ++idx) {
            size_t remainder = idx;
            std::vector<int64_t> coords(ndim);
            for (int64_t dim = 0; dim < ndim; ++dim) {
                coords[dim] = remainder / input_strides[dim];
                remainder %= input_strides[dim];
            }

            std::vector<int64_t> out_coords;
            out_coords.reserve(output_shape.size());
            for (int64_t dim = 0; dim < ndim; ++dim) {
                if (reduce_mask[dim]) {
                    if (keepdims) {
                        out_coords.push_back(0);
                    }
                } else {
                    out_coords.push_back(coords[dim]);
                }
            }

            int64_t out_idx = 0;
            for (size_t dim = 0; dim < out_coords.size(); ++dim) {
                out_idx += out_coords[dim] * output_strides[dim];
            }

            host_output[out_idx] += static_cast<float>(host_data[idx]);
        }
    } else {
        const float* host_data = getHostData<float>(input, cache);
        for (size_t idx = 0; idx < total_input; ++idx) {
            size_t remainder = idx;
            std::vector<int64_t> coords(ndim);
            for (int64_t dim = 0; dim < ndim; ++dim) {
                coords[dim] = remainder / input_strides[dim];
                remainder %= input_strides[dim];
            }

            std::vector<int64_t> out_coords;
            out_coords.reserve(output_shape.size());
            for (int64_t dim = 0; dim < ndim; ++dim) {
                if (reduce_mask[dim]) {
                    if (keepdims) {
                        out_coords.push_back(0);
                    }
                } else {
                    out_coords.push_back(coords[dim]);
                }
            }

            int64_t out_idx = 0;
            for (size_t dim = 0; dim < out_coords.size(); ++dim) {
                out_idx += out_coords[dim] * output_strides[dim];
            }

            host_output[out_idx] += host_data[idx];
        }
    }

    auto output = allocateOutput(output_shape);
    size_t bytes = output_size * sizeof(float);
    if (use_cpu_fallback_) {
        std::memcpy(output->data<float>(), host_output.data(), bytes);
    } else {
        CUDA_CHECK(cudaMemcpy(output->data<float>(), host_output.data(), bytes, cudaMemcpyHostToDevice));
    }

    tensors_[node.outputs()[0]] = output;
}
