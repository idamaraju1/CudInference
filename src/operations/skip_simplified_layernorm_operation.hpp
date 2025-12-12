#pragma once

#include "operation.hpp"

namespace onnx_runner {

class SkipSimplifiedLayerNormOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "SkipSimplifiedLayerNormalization"; }
};

} // namespace onnx_runner
