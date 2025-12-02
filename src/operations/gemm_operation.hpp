#pragma once

#include "operation.hpp"

namespace onnx_runner {

class GemmOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "Gemm"; }
};

} // namespace onnx_runner
