#pragma once

#include "operation.hpp"

namespace onnx_runner {

class ShapeOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "Shape"; }
};

} // namespace onnx_runner
