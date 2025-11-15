#pragma once

// Main tensor header - provides unified interface
#include "tensor/tensor_base.hpp"
#include "tensor/cpu_tensor.hpp"
#ifndef CPU_ONLY
#include "tensor/gpu_tensor.hpp"
#endif
#include <memory>

namespace onnx_runner {

// Type alias for backward compatibility
// During migration, code can use TensorBase* or std::shared_ptr<TensorBase>

} // namespace onnx_runner
