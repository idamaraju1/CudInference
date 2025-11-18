void GpuExecutor::executeScatterND(const Node& node) {
    if (node.inputs().size() != 3 || node.outputs().size() != 1) {
        throw std::runtime_error("ScatterND expects 3 inputs and 1 output");
    }

    auto data = getTensor(node.inputs()[0]);
    auto indices = getTensor(node.inputs()[1]);
    auto updates = getTensor(node.inputs()[2]);

    if (data->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("ScatterND currently supports FLOAT32 only");
    }

    if (indices->dtype() != DataType::INT64 && indices->dtype() != DataType::INT32) {
        throw std::runtime_error("ScatterND indices must be INT64 or INT32");
    }

    auto output = std::make_shared<Tensor>(data->shape(), DataType::FLOAT32);
    std::vector<uint8_t> data_cache;
    const float* data_ptr = getHostData<float>(data, data_cache);
    std::memcpy(output->ptr_data<float>(), data_ptr, data->size() * sizeof(float));

    std::vector<uint8_t> indices_cache;
    std::vector<int64_t> indices_values(indices->size());
    if (indices->dtype() == DataType::INT64) {
        const int64_t* idx_ptr = getHostData<int64_t>(indices, indices_cache);
        std::copy(idx_ptr, idx_ptr + indices->size(), indices_values.begin());
    } else {
        const int32_t* idx_ptr = getHostData<int32_t>(indices, indices_cache);
        for (size_t i = 0; i < indices->size(); ++i) {
            indices_values[i] = static_cast<int64_t>(idx_ptr[i]);
        }
    }

    std::vector<uint8_t> updates_cache;
    const float* updates_ptr = getHostData<float>(updates, updates_cache);

    int64_t rank = static_cast<int64_t>(data->ndim());
    if (indices->ndim() == 0) {
        throw std::runtime_error("ScatterND: indices must have rank >= 1");
    }
    int64_t index_depth = indices->dim(indices->ndim() - 1);
    if (index_depth > rank) {
        throw std::runtime_error("ScatterND: index depth exceeds data rank");
    }

    int64_t num_updates = indices->size() / index_depth;
    int64_t slice_size = 1;
    for (int64_t i = index_depth; i < rank; ++i) {
        slice_size *= data->dim(i);
    }

    auto strides = computeStrides(data->shape());
    float* out_ptr = output->ptr_data<float>();

    for (int64_t update_idx = 0; update_idx < num_updates; ++update_idx) {
        int64_t base_offset = 0;
        for (int64_t dim = 0; dim < index_depth; ++dim) {
            int64_t dim_size = data->dim(dim);
            int64_t idx_value = indices_values[update_idx * index_depth + dim];
            if (idx_value < 0) {
                idx_value += dim_size;
            }
            if (idx_value < 0 || idx_value >= dim_size) {
                throw std::runtime_error("ScatterND: index out of bounds");
            }
            base_offset += idx_value * strides[dim];
        }

        const float* src_slice = updates_ptr + update_idx * slice_size;
        float* dst_slice = out_ptr + base_offset * slice_size;
        std::memcpy(dst_slice, src_slice, slice_size * sizeof(float));
    }

    if (!use_cpu_fallback_) {
        output->toGPU();
    }

    tensors_[node.outputs()[0]] = output;
}
