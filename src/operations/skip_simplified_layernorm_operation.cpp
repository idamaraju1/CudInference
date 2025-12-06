#include "skip_simplified_layernorm_operation.hpp"
#include "operation_registry.hpp"
#include "operation_utils.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"

namespace onnx_runner {
using namespace operation_utils;

void SkipSimplifiedLayerNormOperation::execute(const Node& node, ExecutionContext& ctx) {
    if (node.inputs().size() < 3 || node.inputs().size() > 4) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization expects input, skip, gamma[, beta]");
    }

    auto input = getTensor(node.inputs()[0], ctx);
    auto skip = getTensor(node.inputs()[1], ctx);
    auto gamma = getTensor(node.inputs()[2], ctx);
    std::shared_ptr<Tensor> beta = (node.inputs().size() == 4) ? getTensor(node.inputs()[3], ctx) : nullptr;

    if (input->dtype() != DataType::FLOAT32 || skip->dtype() != DataType::FLOAT32 ||
        gamma->dtype() != DataType::FLOAT32 || (beta && beta->dtype() != DataType::FLOAT32)) {
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

    if (gamma->size() != hidden) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization: gamma must match normalized dimension");
    }

    if (beta && beta->size() != hidden) {
        throw std::runtime_error("SkipSimplifiedLayerNormalization: beta must match normalized dimension");
    }

    float epsilon = node.getFloatAttr("epsilon", 1e-5f);
    size_t rows = input->size() / hidden;

    const auto& outs = node.outputs();
    bool need_sum = outs.size() > 3 && !outs[3].empty();
    auto sum_tensor = need_sum ? allocateOutput(input->shape(), ctx) : nullptr;
    auto output_tensor = allocateOutput(input->shape(), ctx);

    // Allocate GPU memory for output tensors when running on GPU
    if (!ctx.use_cpu) {
        output_tensor->allocateGPU();
        if (sum_tensor) {
            sum_tensor->allocateGPU();
        }
    }

    if (ctx.use_cpu) {
        std::vector<uint8_t> input_cache, skip_cache, gamma_cache, beta_cache;
        const float* input_data = getHostData<float>(input, input_cache);
        const float* skip_data = getHostData<float>(skip, skip_cache);
        const float* gamma_data = getHostData<float>(gamma, gamma_cache);
        const float* beta_data = beta ? getHostData<float>(beta, beta_cache) : nullptr;

        float* sum_data = sum_tensor ? sum_tensor->data<float>() : nullptr;
        float* output_data = output_tensor->data<float>();

        if (ctx.num_cpu_threads > 1) {
            kernels::skipSimplifiedLayerNormCPUMultiThreaded(
                input_data, skip_data, gamma_data, beta_data, output_data, sum_data,
                static_cast<int>(rows), static_cast<int>(hidden), epsilon, ctx.num_cpu_threads);
        } else {
            kernels::skipSimplifiedLayerNormCPU(
                input_data, skip_data, gamma_data, beta_data, output_data, sum_data,
                static_cast<int>(rows), static_cast<int>(hidden), epsilon);
        }
    } else {
        if (input->device() == DeviceType::CPU) input->toGPU();
        if (skip->device() == DeviceType::CPU) skip->toGPU();
        if (gamma->device() == DeviceType::CPU) gamma->toGPU();
        if (beta && beta->device() == DeviceType::CPU) beta->toGPU();

        const float* gamma_dev = gamma->data<float>();
        const float* beta_dev = beta ? beta->data<float>() : nullptr;
        float* residual_dev = sum_tensor ? sum_tensor->data<float>() : nullptr;

        kernels::launchSkipSimplifiedLayerNorm(
            input->data<float>(), skip->data<float>(), gamma_dev, beta_dev,
            output_tensor->data<float>(), residual_dev,
            static_cast<int>(rows), static_cast<int>(hidden), epsilon, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (!outs.empty() && !outs[0].empty()) {
        storeOutput(outs[0], output_tensor, ctx);
    }
    if (need_sum) {
        storeOutput(outs[3], sum_tensor, ctx);
    }
}

REGISTER_OPERATION(OpType::SKIPSIMPLIFIEDLAYERNORM, SkipSimplifiedLayerNormOperation)
} // namespace onnx_runner
