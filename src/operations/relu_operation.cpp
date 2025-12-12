#include "relu_operation.hpp"
#include "operation_registry.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"
#include "../utils/tensor.hpp"

namespace onnx_runner {

void ReluOperation::execute(const Node& node, ExecutionContext& ctx) {
    // Validate inputs and outputs
    validateInputOutputCount(node, 1, 1);

    // Get input tensor
    auto X = getTensor(node.inputs()[0], ctx);
    auto Y = allocateOutput(X->shape(), ctx);

    int size = X->size();
    logDebug(ctx, "  ReLU: size=", size);

    // GPU_PERSISTENT mode: keep everything on GPU
    if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
        X->ensureOnGPU();

        const float* d_X = X->deviceData<float>();
        float* d_Y = Y->mutableDeviceData<float>();

        kernels::launchReLU(d_X, d_Y, size);
        storeOutput(node.outputs()[0], Y, ctx);
        return;
    }

    // CPU execution
    if (ctx.use_cpu) {
        const float* X_data = X->data<float>();
        float* Y_data = Y->data<float>();

        if (ctx.num_cpu_threads > 1) {
            kernels::reluCPUMultiThreaded(X_data, Y_data, size, ctx.num_cpu_threads);
        } else {
            kernels::reluCPU(X_data, Y_data, size);
        }
        storeOutput(node.outputs()[0], Y, ctx);
        return;
    }

    // GPU execution with copy mode
    kernels::launchReLU(X->data<float>(), Y->data<float>(), size);
    CUDA_CHECK(cudaDeviceSynchronize());

    storeOutput(node.outputs()[0], Y, ctx);
}

// Register the operation
REGISTER_OPERATION(OpType::RELU, ReluOperation)

} // namespace onnx_runner
