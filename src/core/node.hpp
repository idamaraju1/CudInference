#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace onnx_runner {

// Forward declaration
class Tensor;

// Supported operation types
enum class OpType {
    UNKNOWN,
    MATMUL,
    RELU,
    ADD,
    GEMM,
    CONV,
    MAXPOOL,
    FLATTEN,
    RESHAPE,
    SOFTMAX,
    BATCHNORM
};

// Convert string to OpType
OpType stringToOpType(const std::string& op_name);
std::string opTypeToString(OpType op_type);

// Attribute value (simplified - supports int, float, string, and lists)
struct AttributeValue {
    enum class Type { INT, FLOAT, STRING, INTS, FLOATS, STRINGS } type;

    int64_t i;
    float f;
    std::string s;
    std::vector<int64_t> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;

    static AttributeValue fromInt(int64_t value) {
        AttributeValue attr;
        attr.type = Type::INT;
        attr.i = value;
        return attr;
    }

    static AttributeValue fromFloat(float value) {
        AttributeValue attr;
        attr.type = Type::FLOAT;
        attr.f = value;
        return attr;
    }

    static AttributeValue fromString(const std::string& value) {
        AttributeValue attr;
        attr.type = Type::STRING;
        attr.s = value;
        return attr;
    }

    static AttributeValue fromInts(const std::vector<int64_t>& values) {
        AttributeValue attr;
        attr.type = Type::INTS;
        attr.ints = values;
        return attr;
    }
};

// Node represents a single operation in the computation graph
class Node {
public:
    Node(const std::string& name, OpType op_type)
        : name_(name), op_type_(op_type) {}

    // Accessors
    const std::string& name() const { return name_; }
    OpType opType() const { return op_type_; }
    const std::vector<std::string>& inputs() const { return inputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }
    const std::map<std::string, AttributeValue>& attributes() const { return attributes_; }

    // Mutators
    void addInput(const std::string& input) { inputs_.push_back(input); }
    void addOutput(const std::string& output) { outputs_.push_back(output); }
    void setAttribute(const std::string& name, const AttributeValue& value) {
        attributes_[name] = value;
    }

    // Get attribute with default
    int64_t getIntAttr(const std::string& name, int64_t default_val = 0) const {
        auto it = attributes_.find(name);
        if (it != attributes_.end() && it->second.type == AttributeValue::Type::INT) {
            return it->second.i;
        }
        return default_val;
    }

    float getFloatAttr(const std::string& name, float default_val = 0.0f) const {
        auto it = attributes_.find(name);
        if (it != attributes_.end() && it->second.type == AttributeValue::Type::FLOAT) {
            return it->second.f;
        }
        return default_val;
    }

    std::string getStringAttr(const std::string& name, const std::string& default_val = "") const {
        auto it = attributes_.find(name);
        if (it != attributes_.end() && it->second.type == AttributeValue::Type::STRING) {
            return it->second.s;
        }
        return default_val;
    }

    std::vector<int64_t> getIntsAttr(const std::string& name) const {
        auto it = attributes_.find(name);
        if (it != attributes_.end() && it->second.type == AttributeValue::Type::INTS) {
            return it->second.ints;
        }
        return {};
    }

private:
    std::string name_;
    OpType op_type_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::map<std::string, AttributeValue> attributes_;
};

} // namespace onnx_runner
