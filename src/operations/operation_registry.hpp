#pragma once

#include "operation.hpp"
#include "../core/node.hpp"
#include <map>
#include <memory>
#include <stdexcept>

namespace onnx_runner {

// Registry for all supported operations
class OperationRegistry {
public:
    // Get the singleton instance
    static OperationRegistry& getInstance() {
        static OperationRegistry instance;
        return instance;
    }

    // Register an operation
    void registerOperation(OpType type, std::shared_ptr<Operation> op) {
        operations_[type] = op;
    }

    // Get an operation by type
    std::shared_ptr<Operation> getOperation(OpType type) const {
        auto it = operations_.find(type);
        if (it == operations_.end()) {
            throw std::runtime_error(
                "Operation not registered: " + opTypeToString(type)
            );
        }
        return it->second;
    }

    // Check if an operation is registered
    bool hasOperation(OpType type) const {
        return operations_.find(type) != operations_.end();
    }

private:
    OperationRegistry() = default;
    std::map<OpType, std::shared_ptr<Operation>> operations_;
};

// Helper macro to register operations
#define REGISTER_OPERATION(OpTypeEnum, OperationClass) \
    namespace { \
        struct OperationClass##Registrar { \
            OperationClass##Registrar() { \
                OperationRegistry::getInstance().registerOperation( \
                    OpTypeEnum, \
                    std::make_shared<OperationClass>() \
                ); \
            } \
        }; \
        static OperationClass##Registrar global_##OperationClass##_registrar; \
    }

} // namespace onnx_runner
