#include "node.hpp"
#include <algorithm>

namespace onnx_runner {

OpType stringToOpType(const std::string& op_name) {
    // Arithmetic / linear algebra
    if (op_name == "MatMul") return OpType::MATMUL;
    if (op_name == "Gemm") return OpType::GEMM;
    if (op_name == "Add") return OpType::ADD;
    if (op_name == "Sub") return OpType::SUB;
    if (op_name == "Mul") return OpType::MUL;
    if (op_name == "Relu") return OpType::RELU;
    // Tensor manipulation
    if (op_name == "Transpose") return OpType::TRANSPOSE;
    if (op_name == "Gather") return OpType::GATHER;
    if (op_name == "Shape") return OpType::SHAPE;
    if (op_name == "Cast") return OpType::CAST;
    // Activations
    if (op_name == "Sigmoid") return OpType::SIGMOID;
    // Reductions
    if (op_name == "ReduceSum") return OpType::REDUCESUM;
    // Advanced operations
    if (op_name == "RotaryEmbedding") return OpType::ROTARYEMBEDDING;
    if (op_name == "GroupQueryAttention") return OpType::GROUPQUERYATTENTION;
    if (op_name == "SimplifiedLayerNormalization") return OpType::SIMPLIFIEDLAYERNORM;
    if (op_name == "SkipSimplifiedLayerNormalization") return OpType::SKIPSIMPLIFIEDLAYERNORM;
    return OpType::UNKNOWN;
}

std::string opTypeToString(OpType op_type) {
    switch (op_type) {
        case OpType::MATMUL: return "MatMul";
        case OpType::GEMM: return "Gemm";
        case OpType::ADD: return "Add";
        case OpType::SUB: return "Sub";
        case OpType::MUL: return "Mul";
        case OpType::RELU: return "Relu";
        case OpType::TRANSPOSE: return "Transpose";
        case OpType::GATHER: return "Gather";
        case OpType::SHAPE: return "Shape";
        case OpType::CAST: return "Cast";
        case OpType::SIGMOID: return "Sigmoid";
        case OpType::REDUCESUM: return "ReduceSum";
        case OpType::ROTARYEMBEDDING: return "RotaryEmbedding";
        case OpType::GROUPQUERYATTENTION: return "GroupQueryAttention";
        case OpType::SIMPLIFIEDLAYERNORM: return "SimplifiedLayerNormalization";
        case OpType::SKIPSIMPLIFIEDLAYERNORM: return "SkipSimplifiedLayerNormalization";
        default: return "Unknown";
    }
}

} // namespace onnx_runner
