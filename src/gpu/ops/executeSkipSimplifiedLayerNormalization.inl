void GpuExecutor::executeSkipSimplifiedLayerNormalization(const Node& node) {
    if (node.inputs().size() < 3 || node.inputs().size() > 4) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization expects input, skip, gamma[, beta]");
    }

    auto input = getTensor(node.inputs()[0]);
    auto skip = getTensor(node.inputs()[1]);
    auto gamma = getTensor(node.inputs()[2]);
    std::shared_ptr<Tensor> beta = nullptr;
    if (node.inputs().size() == 4) {
        beta = getTensor(node.inputs()[3]);
    }

    if (input->dtype() != DataType::FLOAT32 || skip->dtype() != DataType::FLOAT32 ||
        gamma->dtype() != DataType::FLOAT32 ||
        (beta && beta->dtype() != DataType::FLOAT32)) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization currently supports FLOAT32 tensors only");
    }
    if (input->shape() != skip->shape()) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization: input and skip tensors must share the same shape");
    }
    if (input->ndim() < 2) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization expects rank >= 2 tensors");
    }

    int64_t axis_attr = node.getIntAttr("axis", static_cast<int64_t>(input->ndim()) - 1);
    if (axis_attr < 0) axis_attr += static_cast<int64_t>(input->ndim());
    if (axis_attr < 0 || axis_attr >= static_cast<int64_t>(input->ndim())) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization: invalid axis attribute");
    }

    size_t hidden = 1;
    for (int64_t i = axis_attr; i < static_cast<int64_t>(input->ndim()); ++i) {
        hidden *= static_cast<size_t>(input->dim(static_cast<size_t>(i)));
    }
    if (hidden == 0 || input->size() % hidden != 0) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization: invalid hidden dimension derived from axis");
    }
    auto validateScale = [&](const std::shared_ptr<Tensor>& tensor,
                             const char* name) {
        if (!tensor) return;
        if (tensor->size() != hidden) {
            throw std::runtime_error(
                std::string("SkipSimplifiedLayerNormalization: ") + name +
                " must have " + std::to_string(hidden) +
                " elements to match the normalized axis");
        }
    };
    validateScale(gamma, "gamma");
    validateScale(beta, "beta");

    float epsilon = node.getFloatAttr("epsilon", 1e-5f);
    size_t rows = input->size() / hidden;

    auto sum_tensor = allocateOutput(input->shape());
    auto output_tensor = allocateOutput(input->shape());

    auto computeStats = [&](const float* data,
                            std::vector<float>& mean,
                            std::vector<float>& inv_std) {
        for (size_t row = 0; row < rows; ++row) {
            const float* row_ptr = data + row * hidden;
            double sum = 0.0;
            for (size_t col = 0; col < hidden; ++col) {
                sum += static_cast<double>(row_ptr[col]);
            }
            double mean_val = sum / static_cast<double>(hidden);
            double var_acc = 0.0;
            for (size_t col = 0; col < hidden; ++col) {
                double diff = static_cast<double>(row_ptr[col]) - mean_val;
                var_acc += diff * diff;
            }
            double variance = var_acc / static_cast<double>(hidden);
            mean[row] = static_cast<float>(mean_val);
            inv_std[row] = 1.0f / std::sqrt(static_cast<float>(variance) + epsilon);
        }
    };

    const auto& outs = node.outputs();
    bool need_mean = outs.size() > 1 && !outs[1].empty();
    bool need_inv = outs.size() > 2 && !outs[2].empty();
    bool need_stats = need_mean || need_inv;

    if (use_cpu_fallback_) {
        std::vector<uint8_t> input_cache;
        std::vector<uint8_t> skip_cache;
        std::vector<uint8_t> gamma_cache;
        std::vector<uint8_t> beta_cache;
        const float* input_data = getHostData<float>(input, input_cache);
        const float* skip_data = getHostData<float>(skip, skip_cache);
        const float* gamma_data = getHostData<float>(gamma, gamma_cache);
        const float* beta_data = beta ? getHostData<float>(beta, beta_cache) : nullptr;

        float* sum_data = sum_tensor->data<float>();
        float* output_data = output_tensor->data<float>();

        if (num_cpu_threads_ > 1) {
            kernels::addCPUMultiThreaded(input_data, skip_data, sum_data,
                                         static_cast<int>(input->size()),
                                         num_cpu_threads_);
        } else {
            kernels::addCPU(input_data, skip_data, sum_data,
                            static_cast<int>(input->size()));
        }

        if (num_cpu_threads_ > 1) {
            kernels::simplifiedLayerNormCPUMultiThreaded(
                sum_data, gamma_data, beta_data, output_data,
                static_cast<int>(rows), static_cast<int>(hidden),
                epsilon, num_cpu_threads_);
        } else {
            kernels::simplifiedLayerNormCPU(
                sum_data, gamma_data, beta_data, output_data,
                static_cast<int>(rows), static_cast<int>(hidden),
                epsilon);
        }
    } else {
        if (input->device() == DeviceType::CPU) input->toGPU();
        if (skip->device() == DeviceType::CPU) skip->toGPU();
        if (gamma->device() == DeviceType::CPU) gamma->toGPU();
        if (beta && beta->device() == DeviceType::CPU) beta->toGPU();

        kernels::launchAdd(input->data<float>(), skip->data<float>(),
                           sum_tensor->data<float>(),
                           static_cast<int>(input->size()));

        const float* gamma_dev = gamma->data<float>();
        const float* beta_dev = beta ? beta->data<float>() : nullptr;

        kernels::launchSimplifiedLayerNorm(
            sum_tensor->data<float>(),
            gamma_dev,
            beta_dev,
            output_tensor->data<float>(),
            static_cast<int>(rows),
            static_cast<int>(hidden),
            epsilon,
            0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::shared_ptr<Tensor> mean_tensor = nullptr;
    std::shared_ptr<Tensor> invstd_tensor = nullptr;
    if (need_stats) {
        std::vector<uint8_t> sum_cache;
        const float* host_sum = getHostData<float>(sum_tensor, sum_cache);
        std::vector<float> mean(rows, 0.0f);
        std::vector<float> inv_std(rows, 0.0f);
        computeStats(host_sum, mean, inv_std);

        if (need_mean) {
            mean_tensor = std::make_shared<Tensor>(
                std::vector<int64_t>{static_cast<int64_t>(rows)},
                DataType::FLOAT32);
            std::memcpy(mean_tensor->data<float>(), mean.data(),
                        rows * sizeof(float));
            if (!use_cpu_fallback_) mean_tensor->toGPU();
        }
        if (need_inv) {
            invstd_tensor = std::make_shared<Tensor>(
                std::vector<int64_t>{static_cast<int64_t>(rows)},
                DataType::FLOAT32);
            std::memcpy(invstd_tensor->data<float>(), inv_std.data(),
                        rows * sizeof(float));
            if (!use_cpu_fallback_) invstd_tensor->toGPU();
        }
    }

    if (!use_cpu_fallback_) {
        if (sum_tensor->device() == DeviceType::CPU) sum_tensor->toGPU();
        if (output_tensor->device() == DeviceType::CPU) output_tensor->toGPU();
    }

    if (!outs.empty() && !outs[0].empty()) {
        tensors_[outs[0]] = output_tensor;
    }
    if (need_mean) {
        tensors_[outs[1]] = mean_tensor;
    }
    if (need_inv) {
        tensors_[outs[2]] = invstd_tensor;
    }
    if (outs.size() > 3 && !outs[3].empty()) {
        tensors_[outs[3]] = sum_tensor;
    }
}
