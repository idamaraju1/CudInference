#pragma once

#include "operation.hpp"

namespace onnx_runner {

class MulOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "Mul"; }

private:
    void executeElementwise(
        std::shared_ptr<Tensor> A,
        std::shared_ptr<Tensor> B,
        std::shared_ptr<Tensor> output,
        ExecutionContext& ctx
    );

    void executeScalarBroadcast(
        std::shared_ptr<Tensor> tensor,
        float scalar,
        std::shared_ptr<Tensor> output,
        ExecutionContext& ctx
    );
};

} // namespace onnx_runner
