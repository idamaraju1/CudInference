#pragma once

#include "graph.hpp"
#include <string>
#include <memory>

namespace onnx_runner {

// ModelParser reads an ONNX file and constructs a Graph object
class ModelParser {
public:
    ModelParser() = default;

    // Parse an ONNX model file and return a Graph
    // This is the main entry point for loading models
    std::shared_ptr<Graph> parse(const std::string& model_path);

private:
    // Helper to convert ONNX data type to our DataType enum
    DataType onnxDataTypeToDataType(int onnx_type);

    // Helper to extract tensor data from ONNX TensorProto
    std::shared_ptr<Tensor> parseTensorProto(const void* tensor_proto);

    // Helper to convert ONNX SparseTensorProto to dense Tensor
    std::shared_ptr<Tensor> parseSparseTensorProto(const void* sparse_tensor_proto);
};

} // namespace onnx_runner
