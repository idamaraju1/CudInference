#pragma once

#include "operation.hpp"

namespace onnx_runner {

class RotaryEmbeddingOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "RotaryEmbedding"; }
};

} // namespace onnx_runner
