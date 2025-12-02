#include "shape_operation.hpp"
#include "operation_registry.hpp"

namespace onnx_runner {

void ShapeOperation::execute(const Node& node, ExecutionContext& ctx) {
    // Validate inputs and outputs
    validateInputOutputCount(node, 1, 1);

    // Get input tensor
    auto input = getTensor(node.inputs()[0], ctx);
    const auto& input_shape = input->shape();

    // Create output tensor: 1D tensor with length = ndim
    std::vector<int64_t> output_shape = {static_cast<int64_t>(input_shape.size())};
    auto output = std::make_shared<Tensor>(output_shape, DataType::INT64);

    // Fill with shape values
    int64_t* data_ptr = output->data<int64_t>();
    for (size_t i = 0; i < input_shape.size(); ++i) {
        data_ptr[i] = input_shape[i];
    }

    // Transfer to GPU if needed (not in CPU mode and not in PERSISTENT mode where it stays on CPU)
    if (!ctx.use_cpu && ctx.gpu_mode != ExecutionContext::GPUMode::PERSISTENT) {
        output->toGPU();
    }

    storeOutput(node.outputs()[0], output, ctx);
}

// Register the operation
REGISTER_OPERATION(OpType::SHAPE, ShapeOperation)

} // namespace onnx_runner
