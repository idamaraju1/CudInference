#pragma once

#include "operation.hpp"

namespace onnx_runner {

class SimplifiedLayerNormOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "SimplifiedLayerNormalization"; }
};

} // namespace onnx_runner
