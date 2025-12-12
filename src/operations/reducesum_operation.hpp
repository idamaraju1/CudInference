#pragma once

#include "operation.hpp"

namespace onnx_runner {

class ReduceSumOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "ReduceSum"; }
};

} // namespace onnx_runner
