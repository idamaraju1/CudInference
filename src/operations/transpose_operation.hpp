#pragma once

#include "operation.hpp"

namespace onnx_runner {

class TransposeOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "Transpose"; }
};

} // namespace onnx_runner
