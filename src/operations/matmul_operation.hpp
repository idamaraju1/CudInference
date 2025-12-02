#pragma once

#include "operation.hpp"

namespace onnx_runner {

class MatMulOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "MatMul"; }
};

} // namespace onnx_runner
