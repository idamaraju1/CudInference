#include "model_parser.hpp"
#include "../utils/logger.hpp"
#include "onnx.pb.h"
#include <fstream>
#include <stdexcept>
#include <execution>
#include <mutex>
#include <vector>
#include <omp.h>

namespace onnx_runner {

std::shared_ptr<Graph> ModelParserMultiThreaded::parse(const std::string& model_path) {
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

    // Mutex to protect graph insertion and logging in parallel sections
    std::mutex graph_mutex;

    /* Parse initializers (Parallelized) */
    LOG_INFO("Parsing ", onnx_graph.initializer_size(), " initializers...");

    // use openMP to parallelize
    #pragma omp parallel for
    for (int i = 0; i < onnx_graph.initializer_size(); ++i) {
        const onnx::TensorProto& tensor_proto = onnx_graph.initializer(i);
        std::string name = tensor_proto.name();

        // Heavy work (parses protobuf + copies tensor data)
        auto tensor = parseTensorProto(&tensor_proto);

        // Protect graph mutation and logging
        #pragma omp critical
        {
            graph->addInitializer(name, tensor);
            LOG_DEBUG("  Initializer: ", name, " ", tensor->shapeStr());
        }
    }

    /* parse graph inputs (not parallelized) */
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

    /* Parse graph outputs: not parallelized */
    LOG_INFO("Parsing ", onnx_graph.output_size(), " outputs...");
    for (int i = 0; i < onnx_graph.output_size(); ++i) {
        const onnx::ValueInfoProto& output = onnx_graph.output(i);
        graph->addOutput(output.name());
        LOG_DEBUG("  Output: ", output.name());
    }

    /* Parse Nodes: Parallelized */
    LOG_INFO("Parsing ", onnx_graph.node_size(), " nodes...");

    #pragma omp parallel for
    for (int i = 0; i < onnx_graph.node_size(); ++i) {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);

        std::string node_name = onnx_node.name();
        if (node_name.empty() && onnx_node.output_size() > 0) {
            node_name = onnx_node.output(0);
        }

        OpType op_type = stringToOpType(onnx_node.op_type());

        // Create and populate the Node *locally*
        auto node = std::make_shared<Node>(node_name, op_type);

        // Inputs
        for (int j = 0; j < onnx_node.input_size(); ++j) {
            node->addInput(onnx_node.input(j));
        }

        // Outputs
        for (int j = 0; j < onnx_node.output_size(); ++j) {
            node->addOutput(onnx_node.output(j));
        }

        // Attributes
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
                    values.reserve(attr.ints_size());
                    for (int k = 0; k < attr.ints_size(); ++k) {
                        values.push_back(attr.ints(k));
                    }
                    node->setAttribute(attr_name, AttributeValue::fromInts(values));
                    break;
                }
                default:
                    // We’ll log this inside the critical section
                    break;
            }
        }

        // Commit to graph + logging under a critical region
        #pragma omp critical
        {
            if (op_type == OpType::UNKNOWN) {
                LOG_WARN("  Unknown op type: ", onnx_node.op_type(), " in node ", node_name);
            }
            graph->addNode(node);
            LOG_DEBUG("  Node: ", opTypeToString(op_type), " (", node_name, ")");
        }
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

} // namespace onnx_runner
