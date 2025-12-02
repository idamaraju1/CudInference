#include "mul_operation.hpp"
#include "operation_registry.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"
#include "../utils/tensor.hpp"

namespace onnx_runner {

void MulOperation::execute(const Node& node, ExecutionContext& ctx) {
    // Validate inputs and outputs
    validateInputOutputCount(node, 2, 1);

    // Get input tensors
    auto A = getTensor(node.inputs()[0], ctx);
    auto B = getTensor(node.inputs()[1], ctx);

    // Check if one is a scalar
    bool A_is_scalar = (A->size() == 1);
    bool B_is_scalar = (B->size() == 1);

    if (A_is_scalar || B_is_scalar || (A->shape() == B->shape())) {
        // Element-wise or scalar multiplication
        auto output_shape = A_is_scalar ? B->shape() : A->shape();
        auto output = allocateOutput(output_shape, ctx);

        if (A_is_scalar) {
            // A is scalar: output = scalar * B
            float scalar;
            if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
                A->ensureOnGPU();
                CUDA_CHECK(cudaMemcpy(&scalar, A->deviceData<float>(), sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                if (!ctx.use_cpu && A->device() == DeviceType::CUDA) {
                    A->toCPU();
                }
                scalar = A->data<float>()[0];
            }
            executeScalarBroadcast(B, scalar, output, ctx);
        } else if (B_is_scalar) {
            // B is scalar: output = A * scalar
            float scalar;
            if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
                B->ensureOnGPU();
                CUDA_CHECK(cudaMemcpy(&scalar, B->deviceData<float>(), sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                if (!ctx.use_cpu && B->device() == DeviceType::CUDA) {
                    B->toCPU();
                }
                scalar = B->data<float>()[0];
            }
            executeScalarBroadcast(A, scalar, output, ctx);
        } else {
            // Element-wise multiplication
            executeElementwise(A, B, output, ctx);
        }

        storeOutput(node.outputs()[0], output, ctx);
        return;
    }

    // TODO: General NumPy-style broadcasting
    throw std::runtime_error("Mul: complex broadcasting not yet supported in operation abstraction");
}

void MulOperation::executeElementwise(
    std::shared_ptr<Tensor> A,
    std::shared_ptr<Tensor> B,
    std::shared_ptr<Tensor> output,
    ExecutionContext& ctx
) {
    int64_t size = output->size();
    logDebug(ctx, "  Mul (elementwise): size=", size);

    // GPU_PERSISTENT mode
    if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
        A->ensureOnGPU();
        B->ensureOnGPU();
        launchMulKernel(A->deviceData<float>(), B->deviceData<float>(),
                       output->mutableDeviceData<float>(), size, false, ctx.num_cpu_threads);
        return;
    }

    // CPU or GPU execution
    launchMulKernel(A->data<float>(), B->data<float>(), output->data<float>(),
                   size, ctx.use_cpu, ctx.num_cpu_threads);

    if (!ctx.use_cpu) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void MulOperation::executeScalarBroadcast(
    std::shared_ptr<Tensor> tensor,
    float scalar,
    std::shared_ptr<Tensor> output,
    ExecutionContext& ctx
) {
    int64_t size = output->size();
    logDebug(ctx, "  Mul (scalar broadcast): size=", size);

    // GPU_PERSISTENT mode
    if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
        tensor->ensureOnGPU();
        launchMulScalarKernel(tensor->deviceData<float>(), scalar,
                            output->mutableDeviceData<float>(), size, false, ctx.num_cpu_threads);
        return;
    }

    // CPU or GPU execution
    launchMulScalarKernel(tensor->data<float>(), scalar, output->data<float>(),
                         size, ctx.use_cpu, ctx.num_cpu_threads);

    if (!ctx.use_cpu) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Register the operation
REGISTER_OPERATION(OpType::MUL, MulOperation)

} // namespace onnx_runner
