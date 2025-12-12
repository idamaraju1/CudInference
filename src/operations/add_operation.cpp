#include "add_operation.hpp"
#include "operation_registry.hpp"
#include "../utils/tensor.hpp"

namespace onnx_runner {

void AddOperation::execute(const Node& node, ExecutionContext& ctx) {
    // Validate inputs and outputs
    validateInputOutputCount(node, 2, 1);

    // Get input tensors
    auto A = getTensor(node.inputs()[0], ctx);
    auto B = getTensor(node.inputs()[1], ctx);

    // Check for scalar broadcasting
    if (B->size() == 1) {
        // B is scalar: C = A + scalar
        auto C = allocateOutput(A->shape(), ctx);
        executeScalarBroadcast(A, B, C, ctx);
        storeOutput(node.outputs()[0], C, ctx);
        return;
    }

    if (A->size() == 1) {
        // A is scalar: C = scalar + B (commutative)
        auto C = allocateOutput(B->shape(), ctx);
        executeScalarBroadcast(B, A, C, ctx);
        storeOutput(node.outputs()[0], C, ctx);
        return;
    }

    // Element-wise addition (same shape required)
    if (A->shape() != B->shape()) {
        throw std::runtime_error(
            "Add shape mismatch: " + A->shapeStr() + " vs " + B->shapeStr() +
            " (complex broadcasting not yet supported)"
        );
    }

    auto C = allocateOutput(A->shape(), ctx);
    executeElementwise(A, B, C, ctx);
    storeOutput(node.outputs()[0], C, ctx);
}

void AddOperation::executeElementwise(
    std::shared_ptr<Tensor> A,
    std::shared_ptr<Tensor> B,
    std::shared_ptr<Tensor> C,
    ExecutionContext& ctx
) {
    int size = A->size();
    logDebug(ctx, "  Add (elementwise): size=", size);

    // GPU_PERSISTENT mode: keep everything on GPU
    if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
        A->ensureOnGPU();
        B->ensureOnGPU();

        const float* d_A = A->deviceData<float>();
        const float* d_B = B->deviceData<float>();
        float* d_C = C->mutableDeviceData<float>();

        kernels::launchAdd(d_A, d_B, d_C, size);
        return;
    }

    // CPU execution
    if (ctx.use_cpu) {
        const float* A_data = A->data<float>();
        const float* B_data = B->data<float>();
        float* C_data = C->data<float>();

        if (ctx.num_cpu_threads > 1) {
            kernels::addCPUMultiThreaded(A_data, B_data, C_data, size, ctx.num_cpu_threads);
        } else {
            kernels::addCPU(A_data, B_data, C_data, size);
        }
        return;
    }

    // GPU execution with copy mode
    kernels::launchAdd(A->data<float>(), B->data<float>(), C->data<float>(), size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void AddOperation::executeScalarBroadcast(
    std::shared_ptr<Tensor> A,
    std::shared_ptr<Tensor> B,
    std::shared_ptr<Tensor> C,
    ExecutionContext& ctx
) {
    int size = A->size();
    logDebug(ctx, "  Add (scalar broadcast): size=", size);

    // GPU_PERSISTENT mode: scalar broadcast on GPU
    if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
        A->ensureOnGPU();
        B->ensureOnGPU();

        const float* d_A = A->deviceData<float>();
        float* d_C = C->mutableDeviceData<float>();

        // Read scalar from GPU
        float scalar;
        CUDA_CHECK(cudaMemcpy(&scalar, B->deviceData<float>(), sizeof(float), cudaMemcpyDeviceToHost));

        kernels::launchAddScalar(d_A, scalar, d_C, size);
        return;
    }

    // Get scalar value (ensure B is on CPU)
    if (!ctx.use_cpu && B->device() == DeviceType::CUDA) {
        B->toCPU();
    }
    float scalar = B->data<float>()[0];

    // CPU execution
    if (ctx.use_cpu) {
        const float* A_data = A->data<float>();
        float* C_data = C->data<float>();

        if (ctx.num_cpu_threads > 1) {
            #pragma omp parallel for num_threads(ctx.num_cpu_threads)
            for (int i = 0; i < size; ++i) {
                C_data[i] = A_data[i] + scalar;
            }
        } else {
            for (int i = 0; i < size; ++i) {
                C_data[i] = A_data[i] + scalar;
            }
        }
        return;
    }

    // GPU execution with copy mode
    kernels::launchAddScalar(A->data<float>(), scalar, C->data<float>(), size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Register the operation
REGISTER_OPERATION(OpType::ADD, AddOperation)

} // namespace onnx_runner
