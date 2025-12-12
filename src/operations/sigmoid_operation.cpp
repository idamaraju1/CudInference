#include "sigmoid_operation.hpp"
#include "operation_registry.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"
#include "../utils/tensor.hpp"

namespace onnx_runner {

void SigmoidOperation::execute(const Node& node, ExecutionContext& ctx) {
    // Validate inputs and outputs
    validateInputOutputCount(node, 1, 1);

    // Get input tensor
    auto input = getTensor(node.inputs()[0], ctx);

    if (input->dtype() != DataType::FLOAT32) {
        throw std::runtime_error("Sigmoid currently supports FLOAT32 only");
    }

    auto output = allocateOutput(input->shape(), ctx, DataType::FLOAT32);
    int size = static_cast<int>(input->size());

    logDebug(ctx, "  Sigmoid: size=", size);

    // GPU_PERSISTENT mode: keep everything on GPU
    if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
        input->ensureOnGPU();

        const float* d_input = input->deviceData<float>();
        float* d_output = output->mutableDeviceData<float>();

        kernels::launchSigmoid(d_input, d_output, size);
        storeOutput(node.outputs()[0], output, ctx);
        return;
    }

    // CPU execution
    if (ctx.use_cpu) {
        const float* src = input->data<float>();
        float* dst = output->data<float>();

        if (ctx.num_cpu_threads > 1) {
            kernels::sigmoidCPUMultiThreaded(src, dst, size, ctx.num_cpu_threads);
        } else {
            kernels::sigmoidCPU(src, dst, size);
        }
        storeOutput(node.outputs()[0], output, ctx);
        return;
    }

    // GPU execution with copy mode
    kernels::launchSigmoid(input->data<float>(), output->data<float>(), size);
    CUDA_CHECK(cudaDeviceSynchronize());

    storeOutput(node.outputs()[0], output, ctx);
}

// Register the operation
REGISTER_OPERATION(OpType::SIGMOID, SigmoidOperation)

} // namespace onnx_runner
