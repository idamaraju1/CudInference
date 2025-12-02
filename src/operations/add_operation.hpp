#pragma once

#include "operation.hpp"
#include "../executors/gpu/kernels/gpu_kernels.cuh"

namespace onnx_runner {

class AddOperation : public Operation {
public:
    void execute(const Node& node, ExecutionContext& ctx) override;
    const char* name() const override { return "Add"; }

private:
    // Execute element-wise add
    void executeElementwise(
        std::shared_ptr<Tensor> A,
        std::shared_ptr<Tensor> B,
        std::shared_ptr<Tensor> C,
        ExecutionContext& ctx
    );

    // Execute scalar broadcasting (B is scalar)
    void executeScalarBroadcast(
        std::shared_ptr<Tensor> A,
        std::shared_ptr<Tensor> B,
        std::shared_ptr<Tensor> C,
        ExecutionContext& ctx
    );
};

} // namespace onnx_runner
