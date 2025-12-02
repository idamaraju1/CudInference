#pragma once

#include "operation.hpp"

namespace onnx_runner {

class GatherOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "Gather"; }
};

} // namespace onnx_runner
