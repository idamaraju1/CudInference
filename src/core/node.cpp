#include "node.hpp"
#include <algorithm>

namespace onnx_runner {

OpType stringToOpType(const std::string& op_name) {
    if (op_name == "MatMul") return OpType::MATMUL;
    if (op_name == "Relu") return OpType::RELU;
    if (op_name == "Add") return OpType::ADD;
    if (op_name == "Gemm") return OpType::GEMM;
    if (op_name == "Conv") return OpType::CONV;
    if (op_name == "MaxPool") return OpType::MAXPOOL;
    if (op_name == "Flatten") return OpType::FLATTEN;
    if (op_name == "Reshape") return OpType::RESHAPE;
    if (op_name == "Softmax") return OpType::SOFTMAX;
    if (op_name == "BatchNormalization") return OpType::BATCHNORM;
    return OpType::UNKNOWN;
}

std::string opTypeToString(OpType op_type) {
    switch (op_type) {
        case OpType::MATMUL: return "MatMul";
        case OpType::RELU: return "Relu";
        case OpType::ADD: return "Add";
        case OpType::GEMM: return "Gemm";
        case OpType::CONV: return "Conv";
        case OpType::MAXPOOL: return "MaxPool";
        case OpType::FLATTEN: return "Flatten";
        case OpType::RESHAPE: return "Reshape";
        case OpType::SOFTMAX: return "Softmax";
        case OpType::BATCHNORM: return "BatchNormalization";
        default: return "Unknown";
    }
}

} // namespace onnx_runner
