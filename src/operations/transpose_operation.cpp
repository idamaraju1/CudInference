#include "transpose_operation.hpp"
#include "operation_registry.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"

namespace onnx_runner {

void TransposeOperation::execute(const Node& node, ExecutionContext& ctx) {
    // Validate inputs and outputs
    validateInputOutputCount(node, 1, 1);

    // Get input tensor
    auto input = getTensor(node.inputs()[0], ctx);

    // Get permutation
    std::vector<int64_t> perm_int64 = node.getIntsAttr("perm");
    std::vector<int> perm;

    if (perm_int64.empty()) {
        // Default: reverse dimensions
        for (int i = input->ndim() - 1; i >= 0; --i) {
            perm.push_back(i);
        }
    } else {
        for (auto p : perm_int64) {
            perm.push_back(static_cast<int>(p));
        }
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    for (auto p : perm) {
        output_shape.push_back(input->dim(p));
    }

    auto output = allocateOutput(output_shape, ctx);

    logDebug(ctx, "  Transpose: input=", input->shapeStr(), " output=", output->shapeStr());

    // Execute transpose
    launchTransposeKernel(input->data<float>(), output->data<float>(),
                         input->shape(), perm, ctx.use_cpu, ctx.num_cpu_threads);

    if (!ctx.use_cpu) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    storeOutput(node.outputs()[0], output, ctx);
}

// Register the operation
REGISTER_OPERATION(OpType::TRANSPOSE, TransposeOperation)

} // namespace onnx_runner
