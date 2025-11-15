#pragma once

#include "executor.hpp"
#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include "../utils/logger.hpp"
#include <map>
#include <string>
#include <memory>

namespace onnx_runner {

/**
 * CPU executor implementation.
 * Executes ONNX computation graphs using CPU operations.
 */
class CpuExecutor : public Executor {
public:
    CpuExecutor() = default;

    // Implement Executor interface
    std::map<std::string, std::shared_ptr<Tensor>>
    execute(const Graph& graph,
            const std::map<std::string, std::shared_ptr<Tensor>>& inputs) override;

    void setVerbose(bool verbose) override { verbose_ = verbose; }
    DeviceType deviceType() const override { return DeviceType::CPU; }
    std::string name() const override { return "CPU"; }

private:
    std::shared_ptr<Tensor> allocateOutput(
        const std::vector<int64_t>& shape,
        DataType dtype = DataType::FLOAT32) override;

    void executeNode(const Node& node) override;

    bool verbose_ = false;
};

} // namespace onnx_runner
