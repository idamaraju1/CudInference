#include "model_parser.hpp"
#include "../utils/logger.hpp"
#include "onnx.pb.h"
#include <fstream>
#include <stdexcept>

namespace onnx_runner {

std::shared_ptr<Graph> ModelParser::parse(const std::string& model_path) {
    LOG_INFO("Parsing ONNX model: ", model_path);

    // Read the file
    std::ifstream input(model_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open model file: " + model_path);
    }

    // Parse the protobuf
    onnx::ModelProto model;
    if (!model.ParseFromIstream(&input)) {
        throw std::runtime_error("Failed to parse ONNX model");
    }

    LOG_INFO("Model IR version: ", model.ir_version());
    LOG_INFO("Model producer: ", model.producer_name(), " ", model.producer_version());

    // Get the graph from the model
    const onnx::GraphProto& onnx_graph = model.graph();
    LOG_INFO("Graph name: ", onnx_graph.name());

    auto graph = std::make_shared<Graph>();

    // Parse initializers (constant tensors like weights and biases)
    LOG_INFO("Parsing ", onnx_graph.initializer_size(), " initializers...");
    for (int i = 0; i < onnx_graph.initializer_size(); ++i) {
        const onnx::TensorProto& tensor_proto = onnx_graph.initializer(i);
        std::string name = tensor_proto.name();

        auto tensor = parseTensorProto(&tensor_proto);
        graph->addInitializer(name, tensor);

        LOG_DEBUG("  Initializer: ", name, " ", tensor->shapeStr());
    }

    // Parse sparse initializers (sparse weights/parameters stored in COO format)
    LOG_INFO("Parsing ", onnx_graph.sparse_initializer_size(), " sparse initializers...");
    for (int i = 0; i < onnx_graph.sparse_initializer_size(); ++i) {
        const onnx::SparseTensorProto& sparse_proto = onnx_graph.sparse_initializer(i);
        std::string name = sparse_proto.values().name();

        // Check for name conflicts with regular initializers
        if (graph->isInitializer(name)) {
            throw std::runtime_error("Sparse initializer name conflicts with regular initializer: " + name);
        }

        auto tensor = parseSparseTensorProto(&sparse_proto);
        graph->addInitializer(name, tensor);

        LOG_DEBUG("  Sparse Initializer: ", name, " ", tensor->shapeStr());
    }

    // Parse graph inputs
    LOG_INFO("Parsing ", onnx_graph.input_size(), " inputs...");
    for (int i = 0; i < onnx_graph.input_size(); ++i) {
        const onnx::ValueInfoProto& input = onnx_graph.input(i);
        std::string name = input.name();

        // Skip if it's an initializer (initializers are also listed as inputs in ONNX)
        if (!graph->isInitializer(name)) {
            // Extract shape from type info
            std::vector<int64_t> shape;
            if (input.has_type() && input.type().has_tensor_type()) {
                const auto& tensor_type = input.type().tensor_type();
                if (tensor_type.has_shape()) {
                    for (int j = 0; j < tensor_type.shape().dim_size(); ++j) {
                        const auto& dim = tensor_type.shape().dim(j);
                        if (dim.has_dim_value()) {
                            shape.push_back(dim.dim_value());
                        } else {
                            // Dynamic dimension - use 1 as default
                            shape.push_back(1);
                        }
                    }
                }
            }

            if (!shape.empty()) {
                graph->addInput(name, shape);
                LOG_DEBUG("  Input: ", name, " shape: ", Tensor(shape).shapeStr());
            } else {
                graph->addInput(name);
                LOG_DEBUG("  Input: ", name, " (no shape info)");
            }
        }
    }

    // Parse graph outputs
    LOG_INFO("Parsing ", onnx_graph.output_size(), " outputs...");
    for (int i = 0; i < onnx_graph.output_size(); ++i) {
        const onnx::ValueInfoProto& output = onnx_graph.output(i);
        graph->addOutput(output.name());
        LOG_DEBUG("  Output: ", output.name());
    }

    // Parse nodes
    LOG_INFO("Parsing ", onnx_graph.node_size(), " nodes...");
    for (int i = 0; i < onnx_graph.node_size(); ++i) {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);

        // Get node name (use output name if node name is empty)
        std::string node_name = onnx_node.name();
        if (node_name.empty() && onnx_node.output_size() > 0) {
            node_name = onnx_node.output(0);
        }

        // Convert op_type
        OpType op_type = stringToOpType(onnx_node.op_type());
        if (op_type == OpType::UNKNOWN) {
            LOG_WARN("  Unknown op type: ", onnx_node.op_type(), " in node ", node_name);
        }

        auto node = std::make_shared<Node>(node_name, op_type);

        // Add inputs
        for (int j = 0; j < onnx_node.input_size(); ++j) {
            node->addInput(onnx_node.input(j));
        }

        // Add outputs
        for (int j = 0; j < onnx_node.output_size(); ++j) {
            node->addOutput(onnx_node.output(j));
        }

        // Parse attributes
        for (int j = 0; j < onnx_node.attribute_size(); ++j) {
            const onnx::AttributeProto& attr = onnx_node.attribute(j);
            std::string attr_name = attr.name();

            switch (attr.type()) {
                case onnx::AttributeProto::INT:
                    node->setAttribute(attr_name, AttributeValue::fromInt(attr.i()));
                    break;
                case onnx::AttributeProto::FLOAT:
                    node->setAttribute(attr_name, AttributeValue::fromFloat(attr.f()));
                    break;
                case onnx::AttributeProto::STRING:
                    node->setAttribute(attr_name, AttributeValue::fromString(attr.s()));
                    break;
                case onnx::AttributeProto::INTS: {
                    std::vector<int64_t> values;
                    for (int k = 0; k < attr.ints_size(); ++k) {
                        values.push_back(attr.ints(k));
                    }
                    node->setAttribute(attr_name, AttributeValue::fromInts(values));
                    break;
                }
                case onnx::AttributeProto::FLOATS: {
                    std::vector<float> values;
                    for (int k = 0; k < attr.floats_size(); ++k) {
                        values.push_back(attr.floats(k));
                    }
                    node->setAttribute(attr_name, AttributeValue::fromFloats(values));
                    break;
                }
                case onnx::AttributeProto::TENSOR: {
                    auto tensor_attr = parseTensorProto(&attr.t());
                    node->setAttribute(attr_name, AttributeValue::fromTensor(tensor_attr));
                    break;
                }
                default:
                    LOG_WARN("  Unsupported attribute type for ", attr_name);
                    break;
            }
        }

        graph->addNode(node);
        LOG_DEBUG("  Node: ", opTypeToString(op_type), " (", node_name, ")");
    }

    LOG_INFO("Model parsed successfully!");
    return graph;
}

DataType ModelParser::onnxDataTypeToDataType(int onnx_type) {
    // ONNX data types from onnx.proto
    switch (onnx_type) {
        case 1:  // FLOAT
            return DataType::FLOAT32;
        case 6:  // INT32
            return DataType::INT32;
        case 7:  // INT64
            return DataType::INT64;
        case 10: // FLOAT16
            return DataType::FLOAT16;
        case 2:  // UINT8
            return DataType::UINT8;
        default:
            LOG_WARN("Unknown ONNX data type: ", onnx_type, ", defaulting to FLOAT32");
            return DataType::FLOAT32;
    }
}

std::shared_ptr<Tensor> ModelParser::parseTensorProto(const void* proto_ptr) {
    const onnx::TensorProto* tensor_proto = static_cast<const onnx::TensorProto*>(proto_ptr);

    // Extract shape
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor_proto->dims_size(); ++i) {
        shape.push_back(tensor_proto->dims(i));
    }

    // Get data type
    DataType dtype = onnxDataTypeToDataType(tensor_proto->data_type());

    // Create tensor
    auto tensor = std::make_shared<Tensor>(shape, dtype);

    // Extract data
    // ONNX can store data in multiple formats - we handle the most common ones
    if (dtype == DataType::FLOAT32) {
        float* data = tensor->data<float>();
        size_t size = tensor->size();

        LOG_DEBUG("    float_data_size=", tensor_proto->float_data_size(),
                  ", has_raw_data=", tensor_proto->has_raw_data(),
                  ", tensor size=", size);

        if (tensor_proto->float_data_size() > 0) {
            // Data is stored in float_data field
            LOG_DEBUG("    Using float_data, size=", tensor_proto->float_data_size());
            for (size_t i = 0; i < size && i < (size_t)tensor_proto->float_data_size(); ++i) {
                data[i] = tensor_proto->float_data(i);
            }
        } else if (tensor_proto->has_raw_data()) {
            // Data is stored in raw_data field
            const std::string& raw_data = tensor_proto->raw_data();
            size_t bytes = std::min(size * sizeof(float), raw_data.size());
            LOG_DEBUG("    Using raw_data, bytes=", bytes, ", size=", size, ", raw_data.size()=", raw_data.size());
            std::memcpy(data, raw_data.data(), bytes);
            LOG_DEBUG("    First value after copy: ", data[0]);
        } else {
            LOG_DEBUG("    No data found!");
        }
    } else if (dtype == DataType::INT64) {
        int64_t* data = tensor->data<int64_t>();
        size_t size = tensor->size();

        if (tensor_proto->int64_data_size() > 0) {
            for (size_t i = 0; i < size && i < (size_t)tensor_proto->int64_data_size(); ++i) {
                data[i] = tensor_proto->int64_data(i);
            }
        } else if (tensor_proto->has_raw_data()) {
            const std::string& raw_data = tensor_proto->raw_data();
            size_t bytes = std::min(size * sizeof(int64_t), raw_data.size());
            std::memcpy(data, raw_data.data(), bytes);
        }
    }
    // Add more data types as needed

    return tensor;
}

std::shared_ptr<Tensor> ModelParser::parseSparseTensorProto(const void* proto_ptr) {
    const onnx::SparseTensorProto* sparse_proto = static_cast<const onnx::SparseTensorProto*>(proto_ptr);

    // Extract dense shape from dims
    std::vector<int64_t> shape;
    for (int i = 0; i < sparse_proto->dims_size(); ++i) {
        shape.push_back(sparse_proto->dims(i));
    }

    // Calculate total size
    int64_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    LOG_DEBUG("  Sparse tensor shape: ", Tensor(shape).shapeStr(), ", total elements: ", total_size);

    // Parse the values tensor (contains NNZ non-zero elements)
    if (!sparse_proto->has_values()) {
        throw std::runtime_error("SparseTensorProto missing values field");
    }
    auto values_tensor = parseTensorProto(&sparse_proto->values());
    int64_t nnz = values_tensor->size(); // Number of non-zero elements
    LOG_DEBUG("  Number of non-zero elements (NNZ): ", nnz);

    // Parse the indices tensor
    if (!sparse_proto->has_indices()) {
        throw std::runtime_error("SparseTensorProto missing indices field");
    }
    auto indices_tensor = parseTensorProto(&sparse_proto->indices());

    // Get data type from values tensor
    DataType dtype = values_tensor->dtype();

    // Currently only support FLOAT32
    if (dtype != DataType::FLOAT32) {
        throw std::runtime_error("Sparse tensors currently only support FLOAT32 data type");
    }

    // Create dense tensor filled with zeros
    auto dense_tensor = std::make_shared<Tensor>(shape, dtype);
    float* dense_data = dense_tensor->data<float>();
    std::memset(dense_data, 0, total_size * sizeof(float));

    // Get values data
    const float* values_data = values_tensor->data<float>();

    // Determine index format and populate dense tensor
    const auto& indices_shape = indices_tensor->shape();

    if (indices_shape.size() == 2 && indices_shape[0] == nnz) {
        // 2D indices format: [NNZ, rank]
        // Each row contains the multi-dimensional index for a value
        int64_t rank = indices_shape[1];
        LOG_DEBUG("  Using 2D indices format: [", nnz, ", ", rank, "]");

        if (rank != (int64_t)shape.size()) {
            throw std::runtime_error("Sparse tensor rank mismatch");
        }

        const int64_t* indices_data = indices_tensor->data<int64_t>();

        for (int64_t i = 0; i < nnz; ++i) {
            // Compute linear index from multi-dimensional index
            int64_t linear_idx = 0;
            int64_t stride = 1;
            for (int64_t j = rank - 1; j >= 0; --j) {
                int64_t idx = indices_data[i * rank + j];
                if (idx < 0 || idx >= shape[j]) {
                    throw std::runtime_error("Sparse tensor index out of bounds");
                }
                linear_idx += idx * stride;
                stride *= shape[j];
            }

            dense_data[linear_idx] = values_data[i];
        }

    } else if (indices_shape.size() == 1 && indices_shape[0] == nnz) {
        // 1D linearized indices format: [NNZ]
        LOG_DEBUG("  Using 1D linearized indices format: [", nnz, "]");

        const int64_t* indices_data = indices_tensor->data<int64_t>();

        for (int64_t i = 0; i < nnz; ++i) {
            int64_t linear_idx = indices_data[i];
            if (linear_idx < 0 || linear_idx >= total_size) {
                throw std::runtime_error("Sparse tensor linearized index out of bounds");
            }
            dense_data[linear_idx] = values_data[i];
        }

    } else {
        throw std::runtime_error("Invalid sparse tensor indices shape");
    }

    LOG_DEBUG("  Converted sparse tensor to dense format");
    return dense_tensor;
}

} // namespace onnx_runner
