#pragma once

#include "operation.hpp"

namespace onnx_runner {

class GroupQueryAttentionOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "GroupQueryAttention"; }
};

} // namespace onnx_runner
