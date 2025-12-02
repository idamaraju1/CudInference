#include "gemm_operation.hpp"
#include "operation_registry.hpp"
#include "operation_utils.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"
#include "../executors/cpu/kernels/cpu_kernels.h"

namespace onnx_runner {

using namespace operation_utils;

void GemmOperation::execute(const Node& node, ExecutionContext& ctx) {
    // GEMM: Y = alpha * A @ B + beta * C
    // Simplified: Y = A @ B + C (assuming alpha=1, beta=1)
    if (node.inputs().size() < 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Gemm expects at least 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0], ctx);
    auto B = getTensor(node.inputs()[1], ctx);

    // Check for transpose attributes
    bool transA = node.getIntAttr("transA", 0) != 0;
    bool transB = node.getIntAttr("transB", 0) != 0;
    float alpha = node.getFloatAttr("alpha", 1.0f);
    float beta = node.getFloatAttr("beta", 1.0f);

    logDebug(ctx, "  Gemm: A", A->shapeStr(), " B", B->shapeStr(),
             " transA=", transA, " transB=", transB);

    // Only support alpha=1, beta=1 for now
    if (alpha != 1.0f || beta != 1.0f) {
        throw std::runtime_error("Gemm only supports alpha=1.0 and beta=1.0");
    }

    // Determine dimensions based on transpose flags
    int64_t M = transA ? A->dim(1) : A->dim(0);
    int64_t K = transA ? A->dim(0) : A->dim(1);
    int64_t K_B = transB ? B->dim(1) : B->dim(0);
    int64_t N = transB ? B->dim(0) : B->dim(1);

    logDebug(ctx, "  Result dimensions: M=", M, " K=", K, " N=", N);

    if (K != K_B) {
        throw std::runtime_error("Gemm dimension mismatch: K dimensions don't match");
    }

    // Handle transpose by creating transposed copies if needed
    std::shared_ptr<Tensor> A_op = A;
    std::shared_ptr<Tensor> B_op = B;

    if (transA) {
        auto A_temp = std::make_shared<Tensor>(A->shape());
        if (A->device() == DeviceType::CUDA) {
            CUDA_CHECK(cudaMemcpy(A_temp->data<float>(), A->data<float>(),
                                 A->size() * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(A_temp->data<float>(), A->data<float>(), A->size() * sizeof(float));
        }

        A_op = std::make_shared<Tensor>(std::vector<int64_t>{M, K});
        transposeMatrix(A_temp->data<float>(), A_op->data<float>(), A->dim(0), A->dim(1));
        if (!ctx.use_cpu) A_op->toGPU();
    }

    if (transB) {
        auto B_temp = std::make_shared<Tensor>(B->shape());
        if (B->device() == DeviceType::CUDA) {
            CUDA_CHECK(cudaMemcpy(B_temp->data<float>(), B->data<float>(),
                                 B->size() * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(B_temp->data<float>(), B->data<float>(), B->size() * sizeof(float));
        }

        B_op = std::make_shared<Tensor>(std::vector<int64_t>{K, N});
        transposeMatrix(B_temp->data<float>(), B_op->data<float>(), B->dim(0), B->dim(1));
        if (!ctx.use_cpu) B_op->toGPU();
    }

    auto Y = allocateOutput({M, N}, ctx);

    // GPU_PERSISTENT MODE
    if (ctx.gpu_mode == ExecutionContext::GPUMode::PERSISTENT) {
        A_op->ensureOnGPU();
        B_op->ensureOnGPU();

        const float* d_A = A_op->deviceData<float>();
        const float* d_B = B_op->deviceData<float>();
        float* d_Y = Y->mutableDeviceData<float>();

        kernels::launchMatMul(d_A, d_B, d_Y, M, K, N);

        if (node.inputs().size() >= 3) {
            auto C = getTensor(node.inputs()[2], ctx);
            C->ensureOnGPU();
            const float* d_C = C->deviceData<float>();
            kernels::launchAdd(d_Y, d_C, d_Y, Y->size());
        }

        storeOutput(node.outputs()[0], Y, ctx);
        return;
    }

    if (ctx.use_cpu) {
        if (ctx.num_cpu_threads > 1) {
            kernels::matmulCPUMultiThreaded(A_op->data<float>(), B_op->data<float>(), Y->data<float>(),
                                           M, K, N, ctx.num_cpu_threads);
        } else {
            kernels::matmulCPU(A_op->data<float>(), B_op->data<float>(), Y->data<float>(),
                              M, K, N);
        }
    } else {
        kernels::launchMatMul(A_op->data<float>(), B_op->data<float>(), Y->data<float>(),
                             M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Add bias if present (broadcast across rows)
    if (node.inputs().size() >= 3) {
        auto C = getTensor(node.inputs()[2], ctx);

        // Bias C has shape [N], Y has shape [M, N]
        // We need to broadcast C across each row of Y
        const float* bias = C->data<float>();
        float* y_data = Y->data<float>();

        if (ctx.use_cpu) {
            // CPU: Add bias to each row
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    y_data[i * N + j] += bias[j];
                }
            }
        } else {
            // GPU: Add bias to each row
            for (int64_t i = 0; i < M; ++i) {
                kernels::launchAdd(y_data + i * N, bias, y_data + i * N, N);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    storeOutput(node.outputs()[0], Y, ctx);
}

// Register the operation
REGISTER_OPERATION(OpType::GEMM, GemmOperation)

} // namespace onnx_runner
