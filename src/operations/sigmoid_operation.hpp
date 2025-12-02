#pragma once

#include "operation.hpp"

namespace onnx_runner {

class SigmoidOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "Sigmoid"; }
};

} // namespace onnx_runner
