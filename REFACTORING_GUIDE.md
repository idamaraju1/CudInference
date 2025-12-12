# OnnxRunner Refactoring and Improvement Guide

This document outlines recommended improvements to enhance code quality, maintainability, and extensibility of the OnnxRunner project.

## Table of Contents

1. [Architecture & Design](#architecture--design)
2. [Code Organization](#code-organization)
3. [Type Safety & Correctness](#type-safety--correctness)
4. [Testing & Validation](#testing--validation)
5. [Configuration & Flexibility](#configuration--flexibility)
6. [Documentation](#documentation)
7. [Performance & Observability](#performance--observability)
8. [Specific Code Quality Issues](#specific-code-quality-issues)
9. [Implementation Priorities](#implementation-priorities)

---

## Architecture & Design

### 1. Operation Abstraction Layer

**Current State**: Operations are implemented as methods in `GpuExecutor` (executeMatMul, executeReLU, etc.) with separate `.inl` files. This creates a monolithic executor class with 138+ lines of method declarations.

**Problem**:
- Tight coupling between executor and operations
- Difficult to test operations in isolation
- Adding new operations requires modifying executor class
- Violates Single Responsibility Principle

**Recommended Solution**: Operation Registry Pattern

**Implementation Steps**:

1. **Create base operation interface** (`src/gpu/operations/operation.hpp`):
```cpp
namespace onnx_runner {
namespace operations {

// Execution context passed to all operations
struct ExecutionContext {
    bool use_cpu;
    int num_cpu_threads;
    ExecutionMode mode;
    bool verbose;
    // Add profiling, memory pool, etc.
};

// Base operation interface
class Operation {
public:
    virtual ~Operation() = default;

    /**
     * Execute the operation
     * @param node ONNX node with inputs/outputs/attributes
     * @param tensors Tensor storage map (both inputs and outputs)
     * @param ctx Execution context
     */
    virtual void execute(
        const Node& node,
        std::map<std::string, std::shared_ptr<Tensor>>& tensors,
        const ExecutionContext& ctx
    ) = 0;

    // Optional: validate before execution
    virtual void validate(const Node& node) const {}

    // Optional: estimate memory/compute cost
    virtual size_t estimateMemory(const Node& node) const { return 0; }
};

} // namespace operations
} // namespace onnx_runner
```

2. **Create operation registry** (`src/gpu/operations/operation_registry.hpp`):
```cpp
class OperationRegistry {
private:
    std::unordered_map<OpType, std::unique_ptr<Operation>> operations_;

public:
    static OperationRegistry& instance() {
        static OperationRegistry registry;
        return registry;
    }

    void registerOperation(OpType type, std::unique_ptr<Operation> op) {
        operations_[type] = std::move(op);
    }

    Operation* getOperation(OpType type) {
        auto it = operations_.find(type);
        if (it == operations_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    bool hasOperation(OpType type) const {
        return operations_.find(type) != operations_.end();
    }
};
```

3. **Implement concrete operations** (`src/gpu/operations/matmul_op.{hpp,cpp}`):
```cpp
// matmul_op.hpp
class MatMulOperation : public Operation {
public:
    void execute(
        const Node& node,
        std::map<std::string, std::shared_ptr<Tensor>>& tensors,
        const ExecutionContext& ctx
    ) override;

private:
    void execute2D(/*...*/);
    void executeBatched(/*...*/);
};

// matmul_op.cpp
void MatMulOperation::execute(const Node& node, /*...*/) {
    // Move implementation from executeMatMul.inl here
    // Access tensors via tensors[name]
    // Use ctx for execution mode decisions
}
```

4. **Refactor GpuExecutor**:
```cpp
// gpu_executor.cpp
void GpuExecutor::executeNode(const Node& node) {
    Operation* op = OperationRegistry::instance().getOperation(node.opType());

    if (!op) {
        throw UnsupportedOperationError(
            "No implementation for " + opTypeToString(node.opType())
        );
    }

    ExecutionContext ctx{
        use_cpu_fallback_,
        num_cpu_threads_,
        exec_mode_,
        verbose_
    };

    op->execute(node, tensors_, ctx);
}
```

5. **Register operations at startup**:
```cpp
// src/gpu/operations/register_operations.cpp
void registerAllOperations() {
    auto& registry = OperationRegistry::instance();

    registry.registerOperation(OpType::MATMUL,
        std::make_unique<MatMulOperation>());
    registry.registerOperation(OpType::RELU,
        std::make_unique<ReLUOperation>());
    registry.registerOperation(OpType::ADD,
        std::make_unique<AddOperation>());
    // ... register all operations
}

// Call in main() or static initializer
```

**Benefits**:
- Operations are isolated and testable
- Executor becomes a simple dispatcher
- Easy to add new operations without modifying executor
- Can load operations dynamically (future: plugin system)
- Clear separation of concerns

**Estimated Effort**: 1-2 days

---

### 2. Error Handling Hierarchy

**Current State**: All errors throw `std::runtime_error` with string messages.

**Problem**:
- Cannot catch specific error types programmatically
- No structured error information
- Difficult to provide context-specific error recovery
- Error messages inconsistent across codebase

**Recommended Solution**: Typed exception hierarchy

**Implementation Steps**:

1. **Create exception hierarchy** (`src/utils/exceptions.hpp`):
```cpp
namespace onnx_runner {

// Base exception class
class OnnxRunnerException : public std::exception {
protected:
    std::string message_;
    std::string file_;
    int line_;

public:
    OnnxRunnerException(
        const std::string& message,
        const char* file = nullptr,
        int line = 0
    ) : message_(message), file_(file ? file : ""), line_(line) {}

    const char* what() const noexcept override {
        return message_.c_str();
    }

    std::string fullMessage() const {
        std::ostringstream oss;
        oss << message_;
        if (!file_.empty()) {
            oss << " [" << file_ << ":" << line_ << "]";
        }
        return oss.str();
    }
};

// Parsing errors
class ParseError : public OnnxRunnerException {
public:
    using OnnxRunnerException::OnnxRunnerException;
};

class UnsupportedFormatError : public ParseError {
public:
    using ParseError::ParseError;
};

// Execution errors
class ExecutionError : public OnnxRunnerException {
public:
    using OnnxRunnerException::OnnxRunnerException;
};

class DimensionMismatchError : public ExecutionError {
private:
    std::vector<int64_t> expected_;
    std::vector<int64_t> actual_;
    std::string operation_;

public:
    DimensionMismatchError(
        const std::string& operation,
        const std::vector<int64_t>& expected,
        const std::vector<int64_t>& actual
    ) : ExecutionError(formatMessage(operation, expected, actual)),
        expected_(expected),
        actual_(actual),
        operation_(operation) {}

    const std::vector<int64_t>& expected() const { return expected_; }
    const std::vector<int64_t>& actual() const { return actual_; }
    const std::string& operation() const { return operation_; }

private:
    static std::string formatMessage(
        const std::string& op,
        const std::vector<int64_t>& exp,
        const std::vector<int64_t>& act
    ) {
        std::ostringstream oss;
        oss << op << ": dimension mismatch. Expected [";
        for (size_t i = 0; i < exp.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << exp[i];
        }
        oss << "], got [";
        for (size_t i = 0; i < act.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << act[i];
        }
        oss << "]";
        return oss.str();
    }
};

class UnsupportedOperationError : public ExecutionError {
private:
    OpType op_type_;

public:
    UnsupportedOperationError(const std::string& op_name)
        : ExecutionError("Unsupported operation: " + op_name),
          op_type_(OpType::UNKNOWN) {}

    UnsupportedOperationError(OpType op_type)
        : ExecutionError("Unsupported operation: " + opTypeToString(op_type)),
          op_type_(op_type) {}

    OpType opType() const { return op_type_; }
};

class TypeError : public ExecutionError {
private:
    DataType expected_;
    DataType actual_;

public:
    TypeError(DataType expected, DataType actual)
        : ExecutionError(formatMessage(expected, actual)),
          expected_(expected),
          actual_(actual) {}

    DataType expected() const { return expected_; }
    DataType actual() const { return actual_; }

private:
    static std::string formatMessage(DataType exp, DataType act);
};

// GPU/CUDA errors
class GpuError : public ExecutionError {
private:
    cudaError_t cuda_error_;

public:
    GpuError(cudaError_t error, const char* file, int line)
        : ExecutionError(
            std::string("CUDA error: ") + cudaGetErrorString(error),
            file,
            line
          ),
          cuda_error_(error) {}

    cudaError_t cudaError() const { return cuda_error_; }
};

// Memory errors
class OutOfMemoryError : public ExecutionError {
private:
    size_t requested_bytes_;

public:
    OutOfMemoryError(size_t bytes)
        : ExecutionError("Out of memory: requested " +
                        std::to_string(bytes) + " bytes"),
          requested_bytes_(bytes) {}

    size_t requestedBytes() const { return requested_bytes_; }
};

// Configuration errors
class ConfigurationError : public OnnxRunnerException {
public:
    using OnnxRunnerException::OnnxRunnerException;
};

} // namespace onnx_runner
```

2. **Update CUDA_CHECK macro** (`src/utils/tensor.hpp`):
```cpp
// Replace old macro
#undef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw ::onnx_runner::GpuError(error, __FILE__, __LINE__); \
        } \
    } while(0)
```

3. **Update existing error throws**:

Before:
```cpp
throw std::runtime_error("MatMul dimension mismatch: ...");
```

After:
```cpp
throw DimensionMismatchError("MatMul", expected_shape, actual_shape);
```

Before:
```cpp
throw std::runtime_error("Unsupported operation: " + op_name);
```

After:
```cpp
throw UnsupportedOperationError(op_name);
```

4. **Usage examples**:
```cpp
// Caller can handle specific errors
try {
    executor.execute(graph, inputs);
} catch (const DimensionMismatchError& e) {
    LOG_ERROR("Dimension mismatch in ", e.operation());
    LOG_ERROR("  Expected: ", shapeToString(e.expected()));
    LOG_ERROR("  Actual: ", shapeToString(e.actual()));
    // Potentially reshape and retry
} catch (const UnsupportedOperationError& e) {
    LOG_ERROR("Operation not implemented: ", opTypeToString(e.opType()));
    // Fallback to CPU or skip
} catch (const GpuError& e) {
    LOG_ERROR(e.fullMessage());
    // Reset GPU state, fallback to CPU
} catch (const OnnxRunnerException& e) {
    LOG_ERROR("General error: ", e.what());
}
```

**Benefits**:
- Structured error information
- Type-safe error handling
- Better error messages
- Easier debugging
- Can add error recovery logic

**Estimated Effort**: 1-2 hours + incremental replacement

---

## Code Organization

### 3. Consolidate `.inl` Files

**Current State**: Operation implementations split across 15+ `.inl` files included into `gpu_executor.cpp`.

**Problem**:
- Non-standard file organization
- Hard to navigate (IDE doesn't show .inl in structure)
- Unclear compilation dependencies
- Violates expected C++ project structure

**Recommended Solution**: Move to standard .cpp files

**Implementation Steps**:

1. **Create operations directory**:
```bash
mkdir -p src/gpu/operations
```

2. **For each .inl file, create corresponding .hpp/.cpp**:

Example for `executeMatMul.inl`:
```bash
# Create header
cat > src/gpu/operations/matmul_op.hpp << 'EOF'
#pragma once
#include "operation.hpp"

namespace onnx_runner {
namespace operations {

class MatMulOperation : public Operation {
public:
    void execute(
        const Node& node,
        std::map<std::string, std::shared_ptr<Tensor>>& tensors,
        const ExecutionContext& ctx
    ) override;

private:
    void execute2D(/* ... */);
    void executeBatched(/* ... */);
};

} // namespace operations
} // namespace onnx_runner
EOF

# Move and rename .inl to .cpp
mv src/gpu/ops/executeMatMul.inl src/gpu/operations/matmul_op.cpp

# Update includes in .cpp
sed -i 's|void GpuExecutor::executeMatMul|void MatMulOperation::execute|' \
    src/gpu/operations/matmul_op.cpp
```

3. **Update CMakeLists.txt**:
```cmake
set(OPERATION_SOURCES
    src/gpu/operations/matmul_op.cpp
    src/gpu/operations/relu_op.cpp
    src/gpu/operations/add_op.cpp
    src/gpu/operations/gemm_op.cpp
    src/gpu/operations/gather_op.cpp
    src/gpu/operations/transpose_op.cpp
    # ... all operations
)

add_executable(onnx_gpu_engine
    src/main.cpp
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
    ${OPERATION_SOURCES}  # Add this
    ${CUDA_SOURCES}
)
```

4. **Remove .inl includes from gpu_executor.cpp**:
```cpp
// Remove all these:
// #include "ops/executeMatMul.inl"
// #include "ops/executeReLU.inl"
// ...

// Replace with operation registry usage (see #1)
```

**Benefits**:
- Standard C++ project structure
- Better IDE support
- Clear compilation units
- Easier to understand dependencies

**Estimated Effort**: 3-4 hours (mostly mechanical)

---

### 4. Separate Concerns in Tensor Class

**Current State**: `Tensor` class has 251 lines handling:
- Shape management
- Memory allocation (CPU/GPU)
- Data transfer
- Type information
- Data access
- Utility operations (reshape, fill, copy)

**Problem**: Violates Single Responsibility Principle, hard to test individual aspects, difficult to optimize memory management separately.

**Recommended Solution**: Split into focused classes

**Implementation Steps**:

1. **Create TensorShape class** (`src/utils/tensor_shape.hpp`):
```cpp
class TensorShape {
private:
    std::vector<int64_t> dims_;
    mutable std::vector<int64_t> strides_;  // Lazily computed
    mutable bool strides_valid_ = false;

public:
    TensorShape() = default;
    explicit TensorShape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

    // Shape accessors
    size_t ndim() const { return dims_.size(); }
    int64_t dim(size_t idx) const { return dims_[idx]; }
    const std::vector<int64_t>& dims() const { return dims_; }

    // Size computation
    size_t totalElements() const {
        if (dims_.empty()) return 1;
        return std::accumulate(dims_.begin(), dims_.end(),
                              1LL, std::multiplies<int64_t>());
    }

    // Strides (for advanced indexing)
    const std::vector<int64_t>& strides() const {
        if (!strides_valid_) {
            computeStrides();
        }
        return strides_;
    }

    // Validation
    bool isCompatibleWith(const TensorShape& other) const;
    bool canBroadcastTo(const TensorShape& target) const;

    // Manipulation
    TensorShape reshape(const std::vector<int64_t>& new_dims) const;
    TensorShape squeeze(int dim = -1) const;
    TensorShape unsqueeze(int dim) const;

    // String representation
    std::string toString() const;

private:
    void computeStrides() const;
};
```

2. **Create TensorStorage interface** (`src/utils/tensor_storage.hpp`):
```cpp
// Abstract storage interface
class TensorStorage {
public:
    virtual ~TensorStorage() = default;

    virtual void* data() = 0;
    virtual const void* data() const = 0;
    virtual size_t sizeBytes() const = 0;
    virtual DeviceType device() const = 0;

    virtual void copyFrom(const TensorStorage& other) = 0;
    virtual void copyTo(TensorStorage& other) const = 0;
};

// CPU storage implementation
class CpuTensorStorage : public TensorStorage {
private:
    std::vector<uint8_t> data_;

public:
    explicit CpuTensorStorage(size_t bytes) : data_(bytes) {}

    void* data() override { return data_.data(); }
    const void* data() const override { return data_.data(); }
    size_t sizeBytes() const override { return data_.size(); }
    DeviceType device() const override { return DeviceType::CPU; }

    void copyFrom(const TensorStorage& other) override;
    void copyTo(TensorStorage& other) const override;
};

// GPU storage implementation
class GpuTensorStorage : public TensorStorage {
private:
    void* gpu_ptr_;
    size_t size_;

public:
    explicit GpuTensorStorage(size_t bytes);
    ~GpuTensorStorage() override;

    void* data() override { return gpu_ptr_; }
    const void* data() const override { return gpu_ptr_; }
    size_t sizeBytes() const override { return size_; }
    DeviceType device() const override { return DeviceType::CUDA; }

    void copyFrom(const TensorStorage& other) override;
    void copyTo(TensorStorage& other) const override;
};
```

3. **Refactor Tensor class** (`src/utils/tensor.hpp`):
```cpp
class Tensor {
private:
    TensorShape shape_;
    DataType dtype_;
    std::shared_ptr<TensorStorage> storage_;

public:
    // Constructors
    Tensor(const TensorShape& shape, DataType dtype = DataType::FLOAT32);
    Tensor(const TensorShape& shape, const std::vector<float>& data);

    // Shape accessors (delegate to shape_)
    const TensorShape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim(); }
    int64_t dim(size_t idx) const { return shape_.dim(idx); }
    size_t size() const { return shape_.totalElements(); }

    // Data type
    DataType dtype() const { return dtype_; }

    // Device management
    DeviceType device() const { return storage_->device(); }
    bool isOnGPU() const { return device() == DeviceType::CUDA; }
    bool isOnCPU() const { return device() == DeviceType::CPU; }

    void toGPU();
    void toCPU();
    void ensureOnGPU();

    // Data access
    template<typename T>
    T* data() { return static_cast<T*>(storage_->data()); }

    template<typename T>
    const T* data() const { return static_cast<const T*>(storage_->data()); }

    // Copy operations
    void copyFrom(const Tensor& other);

    // Shape manipulation (returns new tensor or modifies in place)
    void reshape(const TensorShape& new_shape);
    std::shared_ptr<Tensor> view(const TensorShape& new_shape) const;

    // Utility
    std::string shapeStr() const { return shape_.toString(); }
};
```

**Benefits**:
- Each class has a single, clear responsibility
- Easier to test shape logic separately from memory management
- Can optimize storage strategies independently
- Easier to add new storage backends (e.g., shared memory, mapped memory)
- More maintainable

**Estimated Effort**: 2-3 days

---

## Type Safety & Correctness

### 5. Runtime Type Checking for Tensor Data Access

**Current State**: Template `data<T>()` method performs unchecked cast.

**Problem**: Type mismatches go undetected until runtime crash or silent corruption.

**Recommended Solution**: Add runtime type validation

**Implementation**:

1. **Add type mapping helper** (`src/utils/tensor.hpp`):
```cpp
template<typename T>
constexpr DataType dataTypeFor() {
    if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::INT32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DataType::INT64;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return DataType::UINT8;
    } else if constexpr (std::is_same_v<T, half>) {  // CUDA half type
        return DataType::FLOAT16;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type for tensor");
    }
}
```

2. **Update data() methods** (`src/utils/tensor.hpp`):
```cpp
template<typename T>
T* data() {
    DataType expected = dataTypeFor<T>();
    if (dtype_ != expected) {
        throw TypeError(expected, dtype_);
    }
    return static_cast<T*>(storage_->data());
}

template<typename T>
const T* data() const {
    DataType expected = dataTypeFor<T>();
    if (dtype_ != expected) {
        throw TypeError(expected, dtype_);
    }
    return static_cast<const T*>(storage_->data());
}

// For cases where type is known to be correct (after validation)
template<typename T>
T* uncheckedData() {
    return static_cast<T*>(storage_->data());
}
```

**Benefits**:
- Catches type errors early
- Clear error messages
- Minimal performance impact (only on access, not in hot loops)
- Option to use unchecked version in performance-critical code

**Estimated Effort**: 2-3 hours

---

### 6. RAII for CUDA Resources

**Current State**:
- Thread-local cuBLAS handle with manual management (matmul.cu:9-20)
- CUDA events need manual destroy (gpu_executor.hpp:22-23)

**Problem**:
- Resource leaks if exceptions occur
- Manual cleanup is error-prone
- No clear ownership semantics

**Recommended Solution**: Wrap all CUDA resources in RAII classes

**Implementation**:

1. **Create CUDA utilities** (`src/utils/cuda_utils.hpp`):
```cpp
namespace onnx_runner {
namespace cuda {

// RAII wrapper for cuBLAS handle
class CublasHandle {
private:
    cublasHandle_t handle_;

public:
    CublasHandle() {
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw GpuError(cudaErrorLaunchFailure, __FILE__, __LINE__);
        }
    }

    ~CublasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    // Delete copy
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    // Allow move
    CublasHandle(CublasHandle&& other) noexcept
        : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CublasHandle& operator=(CublasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) cublasDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    // Implicit conversion to handle
    operator cublasHandle_t() const { return handle_; }

    cublasHandle_t get() const { return handle_; }

    void setStream(cudaStream_t stream) {
        cublasSetStream(handle_, stream);
    }
};

// RAII wrapper for CUDA events
class CudaEvent {
private:
    cudaEvent_t event_;

public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    operator cudaEvent_t() const { return event_; }
    cudaEvent_t get() const { return event_; }

    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    float elapsedTime(const CudaEvent& start) const {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }
};

// RAII wrapper for CUDA streams
class CudaStream {
private:
    cudaStream_t stream_;

public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    operator cudaStream_t() const { return stream_; }
    cudaStream_t get() const { return stream_; }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
};

} // namespace cuda
} // namespace onnx_runner
```

2. **Update GPUTimer** (`src/gpu/gpu_executor.hpp`):
```cpp
class GPUTimer {
private:
    cuda::CudaEvent start_;
    cuda::CudaEvent stop_;

public:
    void start() {
        start_.record();
    }

    void stop() {
        stop_.record();
        stop_.synchronize();
    }

    float elapsedMilliseconds() const {
        return stop_.elapsedTime(start_);
    }
};
```

3. **Update matmul kernel** (`src/gpu/kernels/matmul.cu`):
```cpp
namespace {
    // Thread-local handle with RAII
    thread_local std::unique_ptr<cuda::CublasHandle> g_cublas_handle;

    cublasHandle_t getCublasHandle() {
        if (!g_cublas_handle) {
            g_cublas_handle = std::make_unique<cuda::CublasHandle>();
        }
        return g_cublas_handle->get();
    }
}
```

**Benefits**:
- Automatic cleanup, no leaks
- Exception-safe
- Clear ownership semantics
- Follows C++ best practices

**Estimated Effort**: 1 day

---

## Testing & Validation

### 7. Unit Test Infrastructure

**Current State**: Only end-to-end Python validation scripts, no unit tests.

**Problem**:
- Can't test individual components in isolation
- Hard to debug failures
- No regression detection for small changes
- Difficult to verify edge cases

**Recommended Solution**: Add Google Test framework

**Implementation**:

1. **Add GoogleTest to CMakeLists.txt**:
```cmake
# CMakeLists.txt
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Create test executable
add_executable(onnx_tests
    tests/tensor_test.cpp
    tests/tensor_shape_test.cpp
    tests/operations/matmul_test.cpp
    tests/operations/relu_test.cpp
    tests/graph_test.cpp
    tests/model_parser_test.cpp
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(onnx_tests PRIVATE
    gtest_main
    onnx_proto
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
)

include(GoogleTest)
gtest_discover_tests(onnx_tests)
```

2. **Create test directory structure**:
```bash
mkdir -p tests/operations
mkdir -p tests/kernels
```

3. **Write tensor tests** (`tests/tensor_test.cpp`):
```cpp
#include <gtest/gtest.h>
#include "utils/tensor.hpp"

using namespace onnx_runner;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup
    }
};

TEST_F(TensorTest, Construction) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.dim(0), 2);
    EXPECT_EQ(t.dim(1), 3);
    EXPECT_EQ(t.dim(2), 4);
    EXPECT_EQ(t.size(), 24);
}

TEST_F(TensorTest, ShapeCalculation) {
    Tensor t({5, 10, 20});
    EXPECT_EQ(t.size(), 1000);
}

TEST_F(TensorTest, DataAccess) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({2, 2}, data);

    const float* ptr = t.data<float>();
    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(ptr[2], 3.0f);
    EXPECT_FLOAT_EQ(ptr[3], 4.0f);
}

TEST_F(TensorTest, GpuTransfer) {
    std::vector<float> data(100, 1.5f);
    Tensor t({10, 10}, data);

    EXPECT_TRUE(t.isOnCPU());
    EXPECT_FALSE(t.isOnGPU());

    t.toGPU();
    EXPECT_TRUE(t.isOnGPU());
    EXPECT_FALSE(t.isOnCPU());

    t.toCPU();
    EXPECT_TRUE(t.isOnCPU());

    // Verify data unchanged
    const float* ptr = t.data<float>();
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], 1.5f);
    }
}

TEST_F(TensorTest, Reshape) {
    Tensor t({2, 3, 4});
    t.reshape({6, 4});

    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.dim(0), 6);
    EXPECT_EQ(t.dim(1), 4);
    EXPECT_EQ(t.size(), 24);
}

TEST_F(TensorTest, ReshapeInvalidSize) {
    Tensor t({2, 3, 4});
    EXPECT_THROW(t.reshape({5, 5}), std::runtime_error);
}

TEST_F(TensorTest, TypeSafety) {
    Tensor t({10}, DataType::INT32);

    // Should work
    EXPECT_NO_THROW(t.data<int32_t>());

    // Should throw (if type checking implemented)
    // EXPECT_THROW(t.data<float>(), TypeError);
}
```

4. **Write operation tests** (`tests/operations/matmul_test.cpp`):
```cpp
#include <gtest/gtest.h>
#include "gpu/operations/matmul_op.hpp"
#include "utils/tensor.hpp"

using namespace onnx_runner;
using namespace onnx_runner::operations;

class MatMulTest : public ::testing::Test {
protected:
    std::map<std::string, std::shared_ptr<Tensor>> tensors_;
    ExecutionContext ctx_;
    MatMulOperation op_;

    void SetUp() override {
        ctx_.use_cpu = true;  // Use CPU for deterministic testing
        ctx_.num_cpu_threads = 1;
    }
};

TEST_F(MatMulTest, Simple2x2) {
    // A = [[1, 2],
    //      [3, 4]]
    auto A = std::make_shared<Tensor>(
        TensorShape({2, 2}),
        std::vector<float>{1, 2, 3, 4}
    );

    // B = [[5, 6],
    //      [7, 8]]
    auto B = std::make_shared<Tensor>(
        TensorShape({2, 2}),
        std::vector<float>{5, 6, 7, 8}
    );

    tensors_["A"] = A;
    tensors_["B"] = B;

    // Create node
    Node node("matmul_test", OpType::MATMUL);
    node.addInput("A");
    node.addInput("B");
    node.addOutput("C");

    // Execute
    op_.execute(node, tensors_, ctx_);

    // Expected: C = [[19, 22],
    //                [43, 50]]
    auto C = tensors_["C"];
    ASSERT_NE(C, nullptr);
    EXPECT_EQ(C->shape().dims(), std::vector<int64_t>({2, 2}));

    const float* data = C->data<float>();
    EXPECT_FLOAT_EQ(data[0], 19.0f);
    EXPECT_FLOAT_EQ(data[1], 22.0f);
    EXPECT_FLOAT_EQ(data[2], 43.0f);
    EXPECT_FLOAT_EQ(data[3], 50.0f);
}

TEST_F(MatMulTest, DimensionMismatch) {
    auto A = std::make_shared<Tensor>(TensorShape({2, 3}));
    auto B = std::make_shared<Tensor>(TensorShape({4, 5}));

    tensors_["A"] = A;
    tensors_["B"] = B;

    Node node("matmul_fail", OpType::MATMUL);
    node.addInput("A");
    node.addInput("B");
    node.addOutput("C");

    EXPECT_THROW(op_.execute(node, tensors_, ctx_), DimensionMismatchError);
}

TEST_F(MatMulTest, BatchedMatMul) {
    // Test batched matrix multiplication
    auto A = std::make_shared<Tensor>(TensorShape({2, 3, 4}));
    auto B = std::make_shared<Tensor>(TensorShape({2, 4, 5}));

    // Fill with test data
    // ... initialize A and B ...

    tensors_["A"] = A;
    tensors_["B"] = B;

    Node node("batched_matmul", OpType::MATMUL);
    node.addInput("A");
    node.addInput("B");
    node.addOutput("C");

    op_.execute(node, tensors_, ctx_);

    auto C = tensors_["C"];
    EXPECT_EQ(C->shape().dims(), std::vector<int64_t>({2, 3, 5}));
}
```

5. **Write kernel tests** (`tests/kernels/matmul_kernel_test.cpp`):
```cpp
#include <gtest/gtest.h>
#include "gpu/kernels/kernels.cuh"
#include "utils/tensor.hpp"

using namespace onnx_runner;

class MatMulKernelTest : public ::testing::Test {
protected:
    void testMatMul(int M, int K, int N) {
        // Create input tensors on CPU
        std::vector<float> A_data(M * K);
        std::vector<float> B_data(K * N);
        std::vector<float> C_expected(M * N, 0);

        // Fill with test data
        for (int i = 0; i < M * K; ++i) A_data[i] = i % 10;
        for (int i = 0; i < K * N; ++i) B_data[i] = i % 10;

        // Compute expected result on CPU
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += A_data[i * K + k] * B_data[k * N + j];
                }
                C_expected[i * N + j] = sum;
            }
        }

        // Allocate GPU memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B_data.data(), K * N * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Launch kernel
        kernels::launchMatMul(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        std::vector<float> C_actual(M * N);
        CUDA_CHECK(cudaMemcpy(C_actual.data(), d_C, M * N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Compare
        for (int i = 0; i < M * N; ++i) {
            EXPECT_NEAR(C_actual[i], C_expected[i], 1e-3)
                << "Mismatch at index " << i;
        }

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

TEST_F(MatMulKernelTest, Small2x2) {
    testMatMul(2, 2, 2);
}

TEST_F(MatMulKernelTest, Medium16x16) {
    testMatMul(16, 16, 16);
}

TEST_F(MatMulKernelTest, Large256x256) {
    testMatMul(256, 256, 256);
}

TEST_F(MatMulKernelTest, NonSquare) {
    testMatMul(10, 20, 15);
}
```

6. **Run tests**:
```bash
cd build
cmake ..
make onnx_tests
ctest --output-on-failure
```

**Benefits**:
- Automated regression testing
- Fast feedback on changes
- Isolated component testing
- Documentation through examples
- CI/CD ready

**Estimated Effort**: 1-2 days initial setup + ongoing test writing

---

## Configuration & Flexibility

### 8. Configuration System

**Current State**: Magic numbers scattered throughout (TILE_SIZE=16, USE_CUBLAS_THRESHOLD=128, etc.)

**Problem**:
- Hard to tune performance
- No way to configure without recompiling
- Unclear what values are tunable
- Different optimal values for different GPUs

**Recommended Solution**: Centralized configuration system

**Implementation**:

1. **Create config header** (`src/core/config.hpp`):
```cpp
namespace onnx_runner {

struct KernelConfig {
    // MatMul tuning
    size_t matmul_tile_size = 16;
    size_t cublas_threshold = 128;

    // Memory management
    size_t max_batch_size = 32;
    size_t allocation_alignment = 256;
    bool use_memory_pool = false;

    // Execution
    int default_num_threads = 1;
    bool verbose_timing = false;

    // Logging
    LogLevel log_level = LogLevel::INFO;
};

struct ExecutorConfig {
    KernelConfig kernel_config;

    bool use_cpu_fallback = false;
    int num_cpu_threads = 1;
    ExecutionMode execution_mode = ExecutionMode::GPU_COPY;

    // GPU selection
    int gpu_device_id = 0;

    // Factory methods
    static ExecutorConfig defaults();
    static ExecutorConfig fromFile(const std::string& path);
    static ExecutorConfig fromJSON(const std::string& json);
    static ExecutorConfig fromEnv();

    // Serialization
    std::string toJSON() const;
    void saveToFile(const std::string& path) const;
};

} // namespace onnx_runner
```

2. **Implement config loading** (`src/core/config.cpp`):
```cpp
#include "config.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>

ExecutorConfig ExecutorConfig::defaults() {
    return ExecutorConfig{};
}

ExecutorConfig ExecutorConfig::fromEnv() {
    ExecutorConfig config;

    // Check environment variables
    if (const char* val = std::getenv("ONNX_RUNNER_CPU_THREADS")) {
        config.num_cpu_threads = std::atoi(val);
    }

    if (const char* val = std::getenv("ONNX_RUNNER_GPU_DEVICE")) {
        config.gpu_device_id = std::atoi(val);
    }

    if (const char* val = std::getenv("ONNX_RUNNER_VERBOSE")) {
        config.kernel_config.verbose_timing = (std::atoi(val) != 0);
    }

    if (const char* val = std::getenv("ONNX_RUNNER_LOG_LEVEL")) {
        std::string level(val);
        if (level == "DEBUG") config.kernel_config.log_level = LogLevel::DEBUG;
        else if (level == "INFO") config.kernel_config.log_level = LogLevel::INFO;
        else if (level == "WARN") config.kernel_config.log_level = LogLevel::WARNING;
        else if (level == "ERROR") config.kernel_config.log_level = LogLevel::ERROR;
    }

    return config;
}

ExecutorConfig ExecutorConfig::fromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw ConfigurationError("Cannot open config file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return fromJSON(buffer.str());
}

ExecutorConfig ExecutorConfig::fromJSON(const std::string& json) {
    // Simple JSON parsing (or use a library like nlohmann/json)
    ExecutorConfig config;

    // Parse JSON and populate config
    // For now, just return defaults
    // TODO: Implement proper JSON parsing

    return config;
}

std::string ExecutorConfig::toJSON() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"use_cpu_fallback\": " << (use_cpu_fallback ? "true" : "false") << ",\n";
    oss << "  \"num_cpu_threads\": " << num_cpu_threads << ",\n";
    oss << "  \"gpu_device_id\": " << gpu_device_id << ",\n";
    oss << "  \"kernel_config\": {\n";
    oss << "    \"matmul_tile_size\": " << kernel_config.matmul_tile_size << ",\n";
    oss << "    \"cublas_threshold\": " << kernel_config.cublas_threshold << ",\n";
    oss << "    \"verbose_timing\": " << (kernel_config.verbose_timing ? "true" : "false") << "\n";
    oss << "  }\n";
    oss << "}\n";
    return oss.str();
}
```

3. **Update GpuExecutor to use config**:
```cpp
class GpuExecutor {
private:
    ExecutorConfig config_;
    // ... other members ...

public:
    GpuExecutor(const ExecutorConfig& config = ExecutorConfig::defaults())
        : config_(config),
          use_cpu_fallback_(config.use_cpu_fallback),
          num_cpu_threads_(config.num_cpu_threads),
          exec_mode_(config.execution_mode) {

        // Set GPU device
        if (!config.use_cpu_fallback) {
            CUDA_CHECK(cudaSetDevice(config.gpu_device_id));
        }

        // Configure logging
        Logger::instance().setLevel(config.kernel_config.log_level);
    }

    const ExecutorConfig& config() const { return config_; }
};
```

4. **Update kernels to use config**:
```cpp
// Pass config to kernels
void launchMatMul(const float* A, const float* B, float* C,
                  int M, int K, int N,
                  const KernelConfig& config = KernelConfig{},
                  cudaStream_t stream = 0) {

    const int TILE_SIZE = config.matmul_tile_size;
    const int THRESHOLD = config.cublas_threshold;

    if (M >= THRESHOLD || N >= THRESHOLD || K >= THRESHOLD) {
        // Use cuBLAS
    } else {
        // Use custom kernel with configured tile size
    }
}
```

5. **Add CLI option for config file** (`src/main.cpp`):
```cpp
// Add command line argument
if (arg == "--config") {
    std::string config_path = argv[++i];
    executor_config = ExecutorConfig::fromFile(config_path);
}
```

6. **Example config file** (`config.json`):
```json
{
  "use_cpu_fallback": false,
  "num_cpu_threads": 4,
  "gpu_device_id": 0,
  "execution_mode": "GPU_PERSISTENT",
  "kernel_config": {
    "matmul_tile_size": 32,
    "cublas_threshold": 256,
    "max_batch_size": 64,
    "use_memory_pool": true,
    "verbose_timing": false
  },
  "log_level": "INFO"
}
```

**Benefits**:
- Tunable without recompilation
- Easy experimentation
- GPU-specific optimization
- Clear documentation of tunable parameters
- Can A/B test configurations

**Estimated Effort**: 3-4 hours

---

## Documentation

### 9. Inline Code Documentation (Doxygen)

**Current State**: Minimal comments, no structured documentation.

**Problem**: Hard for new contributors, unclear API contracts, no generated docs.

**Recommended Solution**: Add Doxygen-style comments

**Implementation**:

1. **Add Doxyfile** (root directory):
```bash
doxygen -g Doxyfile

# Edit Doxyfile:
# PROJECT_NAME = "OnnxRunner"
# PROJECT_BRIEF = "Custom ONNX GPU Execution Engine"
# INPUT = src
# RECURSIVE = YES
# EXTRACT_ALL = YES
# GENERATE_HTML = YES
# GENERATE_LATEX = NO
```

2. **Document classes** (example for Tensor):
```cpp
/**
 * @file tensor.hpp
 * @brief Tensor class for managing multi-dimensional arrays on CPU/GPU
 */

/**
 * @class Tensor
 * @brief Multi-dimensional array with automatic GPU memory management
 *
 * Tensor provides a high-level interface for creating and manipulating
 * multi-dimensional arrays that can reside on either CPU or GPU memory.
 * Memory is automatically managed through RAII principles.
 *
 * Example usage:
 * @code
 * // Create a 2x3 tensor on CPU
 * Tensor t({2, 3}, DataType::FLOAT32);
 *
 * // Fill with data
 * float* data = t.data<float>();
 * for (int i = 0; i < 6; ++i) data[i] = i;
 *
 * // Transfer to GPU
 * t.toGPU();
 *
 * // Transfer back
 * t.toCPU();
 * @endcode
 *
 * @note Currently only FLOAT32 data type is fully supported
 */
class Tensor {
public:
    /**
     * @brief Construct a tensor with given shape and data type
     * @param shape Dimensions of the tensor (e.g., {2, 3, 4} for 2x3x4)
     * @param dtype Data type (default: FLOAT32)
     * @throws std::runtime_error if shape is invalid
     */
    Tensor(const std::vector<int64_t>& shape,
           DataType dtype = DataType::FLOAT32);

    /**
     * @brief Construct a tensor with initial data
     * @param shape Dimensions of the tensor
     * @param data Initial data (size must match shape)
     * @param dtype Data type (default: FLOAT32)
     * @throws std::runtime_error if data size doesn't match shape
     */
    Tensor(const std::vector<int64_t>& shape,
           const std::vector<float>& data,
           DataType dtype = DataType::FLOAT32);

    /**
     * @brief Get tensor shape
     * @return Vector of dimensions
     */
    const std::vector<int64_t>& shape() const { return shape_; }

    /**
     * @brief Get dimension at specific index
     * @param idx Dimension index (0-based)
     * @return Size of dimension at index
     * @throws std::out_of_range if idx >= ndim()
     */
    int64_t dim(size_t idx) const { return shape_[idx]; }

    /**
     * @brief Get number of dimensions
     * @return Number of dimensions (rank)
     */
    size_t ndim() const { return shape_.size(); }

    /**
     * @brief Get total number of elements
     * @return Product of all dimensions
     */
    size_t size() const { return computeSize(); }

    /**
     * @brief Transfer tensor data to GPU
     *
     * If tensor is already on GPU, this is a no-op.
     * Allocates GPU memory and copies data from CPU.
     *
     * @throws GpuError if CUDA allocation or copy fails
     */
    void toGPU();

    /**
     * @brief Transfer tensor data to CPU
     *
     * If tensor is already on CPU, this is a no-op.
     * Copies data from GPU and frees GPU memory.
     *
     * @throws GpuError if CUDA copy fails
     */
    void toCPU();

    /**
     * @brief Get typed pointer to data
     * @tparam T Data type (must match tensor dtype)
     * @return Pointer to data (CPU or GPU depending on current device)
     * @warning Pointer is invalidated if device is changed
     */
    template<typename T>
    T* data() {
        return static_cast<T*>(data());
    }

private:
    std::vector<int64_t> shape_;  ///< Tensor dimensions
    DataType dtype_;               ///< Data type
    DeviceType device_;            ///< Current device (CPU/GPU)
    std::vector<uint8_t> cpu_data_; ///< CPU storage
    std::shared_ptr<void> gpu_data_; ///< GPU storage (RAII managed)
};
```

3. **Document operations**:
```cpp
/**
 * @class MatMulOperation
 * @brief Matrix multiplication operation (Y = A @ B)
 *
 * Supports both 2D and batched matrix multiplication with broadcasting.
 * For large matrices (>128x128 by default), delegates to cuBLAS.
 * For smaller matrices, uses custom tiled kernel.
 *
 * Input requirements:
 * - A: shape [..., M, K]
 * - B: shape [..., K, N]
 *
 * Output:
 * - Y: shape [..., M, N]
 *
 * Batch dimensions are broadcast following numpy rules.
 *
 * Performance characteristics:
 * - Small matrices (<128): ~50 GFLOPS (custom kernel)
 * - Large matrices: ~2 TFLOPS (cuBLAS on RTX 3090)
 *
 * @see https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul
 */
class MatMulOperation : public Operation {
    // ...
};
```

4. **Generate documentation**:
```bash
doxygen Doxyfile
# Open html/index.html in browser
```

**Benefits**:
- Searchable API documentation
- Clear contracts for functions
- Examples embedded in code
- Easier onboarding for contributors

**Estimated Effort**: 2-3 hours + ongoing maintenance

---

## Performance & Observability

### 10. Profiling and Instrumentation

**Current State**: Basic timing with GPUTimer, no aggregated statistics.

**Problem**: Can't identify bottlenecks across multiple runs, no performance regression detection.

**Recommended Solution**: Performance monitoring system

**Implementation**:

1. **Create performance monitor** (`src/utils/performance_monitor.hpp`):
```cpp
namespace onnx_runner {

struct OperationStats {
    std::string name;
    size_t call_count = 0;
    double total_time_ms = 0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0;
    double mean_time_ms = 0;

    void recordTime(double time_ms) {
        call_count++;
        total_time_ms += time_ms;
        min_time_ms = std::min(min_time_ms, time_ms);
        max_time_ms = std::max(max_time_ms, time_ms);
        mean_time_ms = total_time_ms / call_count;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << name << ": " << call_count << " calls, "
            << "total=" << total_time_ms << "ms, "
            << "mean=" << mean_time_ms << "ms, "
            << "min=" << min_time_ms << "ms, "
            << "max=" << max_time_ms << "ms";
        return oss.str();
    }
};

class PerformanceMonitor {
private:
    std::map<std::string, OperationStats> stats_;
    bool enabled_ = false;

public:
    static PerformanceMonitor& instance() {
        static PerformanceMonitor monitor;
        return monitor;
    }

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }

    void recordOperation(const std::string& op_name, double duration_ms) {
        if (!enabled_) return;
        stats_[op_name].recordTime(duration_ms);
    }

    const std::map<std::string, OperationStats>& getStats() const {
        return stats_;
    }

    void reset() {
        stats_.clear();
    }

    void printReport(std::ostream& os = std::cout) const {
        os << "\n========== Performance Report ==========\n";
        for (const auto& [name, stat] : stats_) {
            os << stat.toString() << "\n";
        }
        os << "========================================\n";
    }

    std::string toJSON() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"operations\": [\n";
        bool first = true;
        for (const auto& [name, stat] : stats_) {
            if (!first) oss << ",\n";
            oss << "    {\n";
            oss << "      \"name\": \"" << name << "\",\n";
            oss << "      \"call_count\": " << stat.call_count << ",\n";
            oss << "      \"total_time_ms\": " << stat.total_time_ms << ",\n";
            oss << "      \"mean_time_ms\": " << stat.mean_time_ms << ",\n";
            oss << "      \"min_time_ms\": " << stat.min_time_ms << ",\n";
            oss << "      \"max_time_ms\": " << stat.max_time_ms << "\n";
            oss << "    }";
            first = false;
        }
        oss << "\n  ]\n";
        oss << "}\n";
        return oss.str();
    }

    void exportJSON(const std::string& path) const {
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }
        file << toJSON();
    }
};

// RAII helper for timing operations
class ScopedTimer {
private:
    std::string operation_name_;
    std::chrono::high_resolution_clock::time_point start_;

public:
    explicit ScopedTimer(const std::string& op_name)
        : operation_name_(op_name),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start_).count() / 1000.0;
        PerformanceMonitor::instance().recordOperation(operation_name_, duration);
    }
};

// Macro for easy scoped timing
#define PROFILE_OPERATION(name) \
    ScopedTimer _scoped_timer_##__LINE__(name)

} // namespace onnx_runner
```

2. **Integrate into executor**:
```cpp
void GpuExecutor::executeNode(const Node& node) {
    PROFILE_OPERATION(opTypeToString(node.opType()));

    // ... execute operation ...
}
```

3. **Add CLI option**:
```cpp
// main.cpp
bool enable_profiling = false;

// Parse args
if (arg == "--profile") {
    enable_profiling = true;
}

// Enable profiling
if (enable_profiling) {
    PerformanceMonitor::instance().enable();
}

// After execution
if (enable_profiling) {
    PerformanceMonitor::instance().printReport();
    PerformanceMonitor::instance().exportJSON("profile.json");
}
```

**Benefits**:
- Identify bottlenecks
- Track performance regressions
- Compare CPU vs GPU performance
- Export for analysis

**Estimated Effort**: 3-4 hours

---

## Specific Code Quality Issues

### 11. Resource Location (Tokenizer Script)

**Current Issue**: `findTokenizerScript()` (main.cpp:70-96) uses trial-and-error file searching.

**Recommendation**: Use CMake to install resources properly

**Implementation**:

```cmake
# CMakeLists.txt

# Install Python scripts
install(FILES scripts/hf_tokenizer.py
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/onnx_runner/scripts
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Define macro with install path
add_definitions(-DONNX_RUNNER_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

# Or configure header
configure_file(
    "${CMAKE_SOURCE_DIR}/src/config.hpp.in"
    "${CMAKE_BINARY_DIR}/src/config.hpp"
)
```

```cpp
// src/config.hpp.in
#pragma once

#define ONNX_RUNNER_TOKENIZER_SCRIPT "@CMAKE_INSTALL_LIBDIR@/onnx_runner/scripts/hf_tokenizer.py"
```

```cpp
// main.cpp
#include "config.hpp"

std::string getTokenizerScript() {
    // First try installed location
    std::ifstream file(ONNX_RUNNER_TOKENIZER_SCRIPT);
    if (file.good()) {
        return ONNX_RUNNER_TOKENIZER_SCRIPT;
    }

    // Fall back to development locations
    std::vector<std::string> dev_paths = {
        "scripts/hf_tokenizer.py",
        "../scripts/hf_tokenizer.py"
    };

    for (const auto& path : dev_paths) {
        std::ifstream f(path);
        if (f.good()) return path;
    }

    throw ConfigurationError("Cannot find tokenizer script");
}
```

---

### 12. Const Correctness

**Current Issue**: Many methods that don't modify state aren't marked const.

**Recommendation**: Add const where appropriate

**Examples**:

```cpp
// Tensor class
const void* data() const;  // Already const
void* mutableData();       // Rename from data()

std::string shapeStr() const;  // Already const
size_t size() const;           // Already const

// Graph class
bool hasInitializer(const std::string& name) const;  // Already const
std::shared_ptr<Tensor> getInitializer(const std::string& name) const;  // Good

// Node class
const std::string& name() const;  // Already const
OpType opType() const;            // Already const
```

**Estimated Effort**: 2-3 hours

---

## Implementation Priorities

### Phase 1: Quick Wins (1 week)
**High impact, low effort improvements**

1. **Add custom exception types** (1-2 hours)
   - Immediate benefit for debugging
   - Low risk, easy to implement

2. **Add const correctness** (2-3 hours)
   - Better compiler checks
   - Documents intent

3. **Create configuration system** (3-4 hours)
   - Enables tuning without recompilation
   - Improves flexibility

4. **Add runtime type checking** (2-3 hours)
   - Catches bugs early
   - Minimal performance impact

5. **Wrap CUDA resources in RAII** (1 day)
   - Prevents leaks
   - Exception-safe

### Phase 2: Code Organization (1-2 weeks)
**Structural improvements**

6. **Consolidate .inl files** (3-4 hours)
   - Standard structure
   - Better IDE support

7. **Add operation registry pattern** (1-2 days)
   - Cleaner architecture
   - Easier to extend

8. **Add Doxygen documentation** (2-3 hours + ongoing)
   - Better onboarding
   - Clear API contracts

### Phase 3: Testing Infrastructure (1-2 weeks)
**Long-term quality**

9. **Add unit test framework** (1-2 days)
   - Regression detection
   - Component isolation

10. **Write core tests** (1 week)
    - Tensor tests
    - Operation tests
    - Kernel tests

### Phase 4: Advanced Improvements (2-3 weeks)
**Major refactoring**

11. **Split Tensor class** (2-3 days)
    - Better SRP compliance
    - More testable

12. **Add performance monitoring** (3-4 hours)
    - Identify bottlenecks
    - Track regressions

13. **Implement memory pooling** (3-5 days)
    - Reduced allocation overhead
    - Better performance

---

## Migration Strategy

When implementing these changes:

1. **Create feature branch** for each improvement
2. **Maintain backward compatibility** where possible
3. **Add tests** before refactoring
4. **Migrate incrementally** (e.g., one operation at a time)
5. **Document changes** in commit messages and CHANGELOG

Example workflow:
```bash
# Create branch
git checkout -b feature/add-exception-hierarchy

# Implement changes
# Write tests
# Update documentation

# Verify tests pass
make onnx_tests && ctest

# Verify end-to-end still works
./scripts/validate_onnx.py

# Commit and push
git commit -m "Add typed exception hierarchy

- Added OnnxRunnerException base class
- Created specific exception types for common errors
- Updated CUDA_CHECK macro to throw GpuError
- Added structured error information

Refs #123"

git push origin feature/add-exception-hierarchy
```

---

## Conclusion

These improvements will make OnnxRunner more:
- **Maintainable**: Clear structure, documented code, testable components
- **Extensible**: Easy to add new operations, flexible configuration
- **Robust**: Better error handling, automated testing, type safety
- **Performant**: Profiling tools, optimization opportunities

Start with Phase 1 quick wins for immediate benefit, then proceed through phases based on priority and available time.

For questions or clarifications on any recommendation, refer to the specific section or create an issue in the project tracker.
