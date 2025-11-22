void GpuExecutor::executeGather(const Node& node) {
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

    // Materialize indices as int64_t on host for uniform handling
    size_t indices_count = indices_tensor->size();
    std::vector<int64_t> host_indices(indices_count, 0);

    switch (indices_tensor->dtype()) {
        case DataType::INT64: {
            std::vector<uint8_t> cache;
            const int64_t* src = getHostData<int64_t>(indices_tensor, cache);
            std::copy(src, src + indices_count, host_indices.begin());
            break;
        }
        case DataType::INT32: {
            std::vector<uint8_t> cache;
            const int32_t* src = getHostData<int32_t>(indices_tensor, cache);
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(src[i]);
            }
            break;
        }
        case DataType::UINT8: {
            std::vector<uint8_t> cache;
            const uint8_t* src = getHostData<uint8_t>(indices_tensor, cache);
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(src[i]);
            }
            break;
        }
        default: {
            std::vector<uint8_t> cache;
            const float* src = getHostData<float>(indices_tensor, cache);
            for (size_t i = 0; i < indices_count; ++i) {
                host_indices[i] = static_cast<int64_t>(std::llround(src[i]));
            }
            break;
        }
    }

    // Compute output shape
    // output_shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
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

    // Compute dimensions for the gather kernel
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

    LOG_DEBUG("  Gather: axis=", axis, ", outer=", outer_size,
              ", axis_dim_data=", axis_dim_data, ", axis_dim_indices=", axis_dim_indices,
              ", inner=", inner_size);

    int64_t total_size = outer_size * axis_dim_indices * inner_size;
    if (outer_size == 0 || axis_dim_indices == 0 || inner_size == 0 || total_size == 0) {
        tensors_[node.outputs()[0]] = output;
        return;
    }

    if (use_cpu_fallback_) {
        std::vector<uint8_t> data_cache;
        const float* host_data = getHostData<float>(data, data_cache);

        launchGatherKernel(
            host_data,
            host_indices.data(),
            output->data<float>(),
            axis_dim_data,
            axis_dim_indices,
            outer_size,
            inner_size,
            true,
            num_cpu_threads_
        );
    } else {
        // GPU path: use device pointers directly if data is on GPU
        const float* d_data;
        std::vector<uint8_t> data_cache;
        bool need_free_data = false;

        if (data->device() == DeviceType::CUDA) {
            d_data = data->data<float>();
        } else {
            const float* host_data = getHostData<float>(data, data_cache);
            float* temp_data;
            size_t data_bytes = data->size() * sizeof(float);
            CUDA_CHECK(cudaMalloc(&temp_data, data_bytes));
            CUDA_CHECK(cudaMemcpy(temp_data, host_data, data_bytes, cudaMemcpyHostToDevice));
            d_data = temp_data;
            need_free_data = true;
        }

        // Allocate and copy indices to GPU
        int64_t* d_indices;
        size_t indices_bytes = host_indices.size() * sizeof(int64_t);
        CUDA_CHECK(cudaMalloc(&d_indices, indices_bytes));
        CUDA_CHECK(cudaMemcpy(d_indices, host_indices.data(), indices_bytes, cudaMemcpyHostToDevice));

        launchGatherKernel(
            d_data,
            d_indices,
            output->data<float>(),
            axis_dim_data,
            axis_dim_indices,
            outer_size,
            inner_size,
            false,  // use GPU kernel
            num_cpu_threads_
        );

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_indices));
        if (need_free_data) {
            CUDA_CHECK(cudaFree(const_cast<float*>(d_data)));
        }
    }

    tensors_[node.outputs()[0]] = output;
}
