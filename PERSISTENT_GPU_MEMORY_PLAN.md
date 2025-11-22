# Persistent GPU Memory Implementation Plan

## Goal
Eliminate all intermediate CPU↔GPU memory transfers. Data should be copied to GPU once at the start of generation, stay on GPU for all computations, and be copied back to CPU only at the very end.

**Target**: Reduce memory transfers from 100+ copies per token to just 2 total copies.

---

## Current Architecture Problems

### Problem 1: Tensor Device Ambiguity
```cpp
// Current Tensor class
class Tensor {
    std::shared_ptr<void> data_;  // Could be CPU or GPU!
    DeviceType device_ = DeviceType::CPU;  // Track location

    // But operations do this:
    const float* getHostData() {  // Always assumes CPU!
        if (device_ == GPU) toCPU();  // Hidden copy!
        return data<float>();
    }
};
```

### Problem 2: Every Operation Copies
```cpp
// executeGroupQueryAttention.inl - CURRENT (BAD)
const float* q_data = getHostData<float>(Q, q_cache);  // Copy GPU→CPU
// ... then later ...
cudaMemcpy(d_Q, q_data, bytes, cudaMemcpyHostToDevice);  // Copy CPU→GPU
```

This pattern is **everywhere**: Gemm, MatMul, Attention, RoPE, LayerNorm, etc.

### Problem 3: No GPU Memory Reuse
- Every operation allocates new GPU memory
- Immediately freed after use
- No memory pooling
- No persistent KV cache

---

## Solution Architecture

### Design Principle: "GPU-First Execution"

1. **Input tensors** → Copied to GPU once
2. **All operations** → Work directly with GPU pointers
3. **Intermediate tensors** → Created on GPU, never touch CPU
4. **KV cache** → Persistent GPU allocation, reused across tokens
5. **Output tensors** → Copied from GPU once at end

---

## Implementation Phases

## Phase 1: Enhance Tensor Class (Foundation)

### 1.1: Add GPU-Aware Data Access Methods

**File**: `src/utils/tensor.hpp`, `src/utils/tensor.cpp`

**Current**:
```cpp
class Tensor {
public:
    template<typename T>
    const T* data() const { return static_cast<const T*>(data_.get()); }

    void toGPU();
    void toCPU();
};
```

**Enhanced**:
```cpp
class Tensor {
public:
    // Existing methods
    template<typename T>
    const T* data() const { return static_cast<const T*>(data_.get()); }

    // NEW: Get device pointer without forcing transfer
    template<typename T>
    const T* deviceData() const {
        if (device_ != DeviceType::CUDA) {
            throw std::runtime_error("Tensor is not on GPU");
        }
        return static_cast<const T*>(data_.get());
    }

    template<typename T>
    T* mutableDeviceData() {
        if (device_ != DeviceType::CUDA) {
            throw std::runtime_error("Tensor is not on GPU");
        }
        return static_cast<T*>(data_.get());
    }

    // NEW: Check device location
    bool isOnGPU() const { return device_ == DeviceType::CUDA; }
    bool isOnCPU() const { return device_ == DeviceType::CPU; }

    // NEW: Ensure tensor is on GPU (copy only if needed)
    void ensureOnGPU() {
        if (device_ != DeviceType::CUDA) {
            toGPU();
        }
    }

    // NEW: Create tensor directly on GPU
    static std::shared_ptr<Tensor> createOnGPU(
        const std::vector<int64_t>& shape,
        DataType dtype
    );

    // Existing
    void toGPU();
    void toCPU();
    DeviceType device() const { return device_; }

private:
    std::shared_ptr<void> data_;
    DeviceType device_ = DeviceType::CPU;
    // ... rest of members ...
};
```

**Implementation**:
```cpp
// src/utils/tensor.cpp

std::shared_ptr<Tensor> Tensor::createOnGPU(
    const std::vector<int64_t>& shape,
    DataType dtype
) {
    auto tensor = std::make_shared<Tensor>(shape, dtype);

    // Allocate directly on GPU
    size_t bytes = tensor->size() * sizeof(float);  // Assume FLOAT32
    void* d_ptr;
    cudaMalloc(&d_ptr, bytes);

    tensor->data_ = std::shared_ptr<void>(d_ptr, CudaDeleter());
    tensor->device_ = DeviceType::CUDA;

    return tensor;
}
```

**Testing**:
```bash
# Add unit test
python3 scripts/test_tensor_gpu.py
```

**Estimated Time**: 2-3 hours
**Risk**: Low (additive, doesn't break existing code)

---

### 1.2: Add GPU Memory Pool (Optional but Recommended)

**File**: `src/utils/gpu_memory_pool.hpp` (NEW)

**Purpose**: Reuse GPU allocations to avoid repeated cudaMalloc/cudaFree

```cpp
class GPUMemoryPool {
public:
    struct Allocation {
        void* ptr;
        size_t size;
        bool in_use;
    };

    void* allocate(size_t bytes);
    void free(void* ptr);
    void clear();  // Release all memory

private:
    std::vector<Allocation> allocations_;
    std::mutex mutex_;
};
```

**Testing**:
```cpp
// Test reuse
auto pool = GPUMemoryPool();
void* ptr1 = pool.allocate(1024);
pool.free(ptr1);
void* ptr2 = pool.allocate(1024);
// ptr1 == ptr2 (reused!)
```

**Estimated Time**: 3-4 hours
**Risk**: Low (self-contained utility)

---

## Phase 2: Refactor GpuExecutor for GPU-First Execution

### 2.1: Add GPU Execution Mode Flag

**File**: `src/gpu/gpu_executor.hpp`

**Current**:
```cpp
class GpuExecutor {
    bool use_cpu_fallback_;
    int num_cpu_threads_;
};
```

**Enhanced**:
```cpp
class GpuExecutor {
    bool use_cpu_fallback_;
    int num_cpu_threads_;

    // NEW: Execution mode
    enum class ExecutionMode {
        CPU_ONLY,      // All tensors on CPU, use CPU kernels
        GPU_COPY,      // Current mode: copy per operation
        GPU_PERSISTENT // NEW: Keep tensors on GPU
    };

    ExecutionMode exec_mode_ = ExecutionMode::GPU_COPY;

    // NEW: Helper to get GPU pointer from tensor
    template<typename T>
    const T* getGPUData(const std::shared_ptr<Tensor>& tensor) {
        if (!tensor->isOnGPU()) {
            throw std::runtime_error("Expected GPU tensor in GPU_PERSISTENT mode");
        }
        return tensor->deviceData<T>();
    }
};
```

**Estimated Time**: 1 hour
**Risk**: Low

---

### 2.2: Update allocateOutput() to Create GPU Tensors

**File**: `src/gpu/gpu_executor.cpp`

**Current**:
```cpp
std::shared_ptr<Tensor> GpuExecutor::allocateOutput(
    const std::vector<int64_t>& shape,
    DataType dtype
) {
    auto output = std::make_shared<Tensor>(shape, dtype);
    if (!use_cpu_fallback_) {
        output->toGPU();  // Allocates on GPU, but data is on CPU first!
    }
    return output;
}
```

**Updated**:
```cpp
std::shared_ptr<Tensor> GpuExecutor::allocateOutput(
    const std::vector<int64_t>& shape,
    DataType dtype
) {
    if (use_cpu_fallback_ || exec_mode_ == ExecutionMode::CPU_ONLY) {
        // CPU tensor
        return std::make_shared<Tensor>(shape, dtype);
    } else if (exec_mode_ == ExecutionMode::GPU_PERSISTENT) {
        // NEW: Create directly on GPU
        return Tensor::createOnGPU(shape, dtype);
    } else {
        // GPU_COPY mode (current behavior)
        auto output = std::make_shared<Tensor>(shape, dtype);
        output->toGPU();
        return output;
    }
}
```

**Estimated Time**: 30 minutes
**Risk**: Low

---

### 2.3: Refactor Operation Execution Pattern

**File**: All `src/gpu/ops/execute*.inl` files

**Current Pattern** (executeGroupQueryAttention.inl):
```cpp
void GpuExecutor::executeGroupQueryAttention(const Node& node) {
    // Get CPU data
    std::vector<uint8_t> q_cache;
    const float* q_data = getHostData<float>(Q, q_cache);  // GPU→CPU copy

    if (use_cpu_fallback_) {
        // CPU path
        launchGroupQueryAttention(..., q_data, ..., true);
    } else {
        // GPU path: allocate and copy
        float* d_Q;
        cudaMalloc(&d_Q, bytes);
        cudaMemcpy(d_Q, q_data, bytes, cudaMemcpyHostToDevice);  // CPU→GPU copy

        launchGroupQueryAttention(..., d_Q, ..., false);

        cudaFree(d_Q);
    }
}
```

**New Pattern**:
```cpp
void GpuExecutor::executeGroupQueryAttention(const Node& node) {
    auto Q = getTensor(inputs[0]);
    auto K = getTensor(inputs[1]);
    auto V = getTensor(inputs[2]);

    if (use_cpu_fallback_) {
        // CPU path: get CPU data
        std::vector<uint8_t> q_cache, k_cache, v_cache;
        const float* q_data = getHostData<float>(Q, q_cache);
        const float* k_data = getHostData<float>(K, k_cache);
        const float* v_data = getHostData<float>(V, v_cache);

        // ... build key_storage, value_storage on CPU ...

        auto output = allocateOutput(Q->shape(), DataType::FLOAT32);
        launchGroupQueryAttention(..., q_data, ..., true, num_cpu_threads_);

        tensors_[outputs[0]] = output;

    } else if (exec_mode_ == ExecutionMode::GPU_PERSISTENT) {
        // NEW: GPU-persistent path

        // Ensure inputs are on GPU (should already be!)
        Q->ensureOnGPU();
        K->ensureOnGPU();
        V->ensureOnGPU();

        // Get GPU pointers directly
        const float* d_Q = Q->deviceData<float>();
        const float* d_K = K->deviceData<float>();
        const float* d_V = V->deviceData<float>();

        // Build KV cache ON GPU
        auto d_K_storage = buildKVCacheGPU(K, past_key, ...);
        auto d_V_storage = buildKVCacheGPU(V, past_value, ...);

        // Allocate output on GPU
        auto output = Tensor::createOnGPU(Q->shape(), DataType::FLOAT32);

        // Launch kernel with GPU pointers
        launchGroupQueryAttention(
            d_Q,
            d_K_storage->deviceData<float>(),
            d_V_storage->deviceData<float>(),
            output->mutableDeviceData<float>(),
            ...,
            false,  // use_cpu = false
            num_cpu_threads_
        );

        // Store outputs (all on GPU)
        tensors_[outputs[0]] = output;
        tensors_[outputs[1]] = d_K_storage;  // present_key
        tensors_[outputs[2]] = d_V_storage;  // present_value

    } else {
        // GPU_COPY mode (current behavior - keep for compatibility)
        // ... existing code ...
    }
}
```

**Key Changes**:
- No `getHostData()` calls in GPU path
- Use `ensureOnGPU()` to verify tensor location
- Use `deviceData<T>()` to get GPU pointers
- Build intermediate data structures on GPU
- Create outputs directly on GPU

**Estimated Time**: 2-3 hours per operation
**Risk**: Medium (requires careful refactoring)

---

## Phase 3: Implement GPU-Side KV Cache Management

### 3.1: Persistent KV Cache

**File**: `src/gpu/gpu_executor.hpp`, `src/gpu/ops/executeGroupQueryAttention.inl`

**Current Problem**:
```cpp
// Allocates new cache every token!
std::vector<float> key_storage(key_storage_elems, 0.f);
std::vector<float> value_storage(value_storage_elems, 0.f);
```

**Solution: Persistent Cache**:

```cpp
// In GpuExecutor class
class GpuExecutor {
    // NEW: Persistent KV cache for autoregressive generation
    struct KVCacheEntry {
        std::shared_ptr<Tensor> key_cache;    // [batch, kv_heads, max_seq, head_dim]
        std::shared_ptr<Tensor> value_cache;
        int current_length = 0;  // How many tokens cached
    };

    std::unordered_map<std::string, KVCacheEntry> kv_cache_;

    // NEW: Initialize cache for a layer
    void initializeKVCache(
        const std::string& cache_key,
        int batch,
        int kv_heads,
        int max_seq_length,
        int head_dim
    );

    // NEW: Append to cache (on GPU)
    void appendKVCache(
        const std::string& cache_key,
        const std::shared_ptr<Tensor>& new_keys,
        const std::shared_ptr<Tensor>& new_values
    );

    // NEW: Get cache slice for attention
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
    getKVCache(const std::string& cache_key, int length);
};
```

**Implementation**:
```cpp
void GpuExecutor::initializeKVCache(
    const std::string& cache_key,
    int batch,
    int kv_heads,
    int max_seq_length,
    int head_dim
) {
    // Allocate large cache on GPU
    std::vector<int64_t> shape = {batch, kv_heads, max_seq_length, head_dim};

    auto key_cache = Tensor::createOnGPU(shape, DataType::FLOAT32);
    auto value_cache = Tensor::createOnGPU(shape, DataType::FLOAT32);

    // Zero initialize
    cudaMemset(key_cache->mutableDeviceData<void>(), 0,
               key_cache->size() * sizeof(float));
    cudaMemset(value_cache->mutableDeviceData<void>(), 0,
               value_cache->size() * sizeof(float));

    kv_cache_[cache_key] = {key_cache, value_cache, 0};
}

void GpuExecutor::appendKVCache(
    const std::string& cache_key,
    const std::shared_ptr<Tensor>& new_keys,
    const std::shared_ptr<Tensor>& new_values
) {
    auto& entry = kv_cache_[cache_key];

    // new_keys: [batch, kv_heads, new_seq, head_dim]
    // Copy to cache at position current_length

    int batch = entry.key_cache->dim(0);
    int kv_heads = entry.key_cache->dim(1);
    int head_dim = entry.key_cache->dim(3);
    int new_seq = new_keys->dim(2);

    // Use cudaMemcpy2D or custom kernel to copy into position
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kv_heads; ++h) {
            float* dst_key = entry.key_cache->mutableDeviceData<float>() +
                ((b * kv_heads + h) * entry.key_cache->dim(2) + entry.current_length) * head_dim;
            const float* src_key = new_keys->deviceData<float>() +
                ((b * kv_heads + h) * new_seq) * head_dim;

            cudaMemcpy(dst_key, src_key, new_seq * head_dim * sizeof(float),
                      cudaMemcpyDeviceToDevice);  // GPU→GPU (fast!)

            // Same for values
            float* dst_val = entry.value_cache->mutableDeviceData<float>() +
                ((b * kv_heads + h) * entry.value_cache->dim(2) + entry.current_length) * head_dim;
            const float* src_val = new_values->deviceData<float>() +
                ((b * kv_heads + h) * new_seq) * head_dim;

            cudaMemcpy(dst_val, src_val, new_seq * head_dim * sizeof(float),
                      cudaMemcpyDeviceToDevice);
        }
    }

    entry.current_length += new_seq;
}
```

**Usage in executeGroupQueryAttention**:
```cpp
// First token: initialize cache
if (!kv_cache_.count("layer_0_kv")) {
    initializeKVCache("layer_0_kv", batch, kv_heads, max_seq_length, head_dim);
}

// Append new K/V to cache
appendKVCache("layer_0_kv", new_keys, new_values);

// Get full cache for attention
auto [k_cache, v_cache] = getKVCache("layer_0_kv", current_length);
```

**Estimated Time**: 4-6 hours
**Risk**: Medium-High (complex state management)

---

### 3.2: GPU-Side KV Cache Building Kernel

**File**: `src/gpu/kernels/kv_cache.cu` (NEW)

Instead of building KV cache on CPU then copying to GPU, build it directly on GPU:

```cpp
// Build cache from scattered K/V projections directly on GPU
__global__ void buildKVCacheKernel(
    const float* K_proj,          // [batch, kv_seq, kv_hidden]
    const float* V_proj,
    const float* past_K,          // [batch, kv_heads, past_len, head_dim] or nullptr
    const float* past_V,
    float* K_cache,               // [batch, kv_heads, total_seq, head_dim] (output)
    float* V_cache,
    int batch,
    int kv_heads,
    int head_dim,
    int past_len,
    int new_len,
    int total_len
) {
    // Each thread handles one element of the cache
    // Rearrange from [batch, seq, kv_hidden] to [batch, kv_heads, seq, head_dim]
    // And concatenate with past cache
}
```

**Estimated Time**: 3-4 hours
**Risk**: Medium

---

## Phase 4: Refactor All Operations for GPU-Persistent Mode

Apply the pattern from Phase 2.3 to all operations:

### Priority Order:

1. **executeGroupQueryAttention** (CRITICAL - most impactful)
2. **executeRotaryEmbedding** (CRITICAL - runs every layer)
3. **executeSimplifiedLayerNormalization** (runs every layer)
4. **executeMatMul / executeGemm** (runs constantly)
5. **executeAdd** (residual connections)
6. **executeReduceSum** (used in normalization)
7. **executeSigmoid** (used in activations)
8. **executeGather** (embedding lookups)
9. All remaining operations

**Per Operation Checklist**:
- [ ] Remove `getHostData()` calls in GPU path
- [ ] Add `exec_mode_ == GPU_PERSISTENT` branch
- [ ] Use `ensureOnGPU()` for inputs
- [ ] Use `deviceData<T>()` for GPU pointers
- [ ] Create outputs with `Tensor::createOnGPU()`
- [ ] Eliminate all temporary CPU buffers
- [ ] Eliminate all `cudaMemcpy` H2D and D2H calls
- [ ] Keep any D2D (device-to-device) copies if needed

**Estimated Time**: 2-3 hours per operation × ~15 operations = **30-45 hours**
**Risk**: Medium (systematic refactoring)

---

## Phase 5: Update Autoregressive Generator

### 5.1: Enable GPU-Persistent Mode

**File**: `src/gpu/autoregressive_generator.cpp`

**Current**:
```cpp
void AutoregressiveGenerator::generate(...) {
    // Sets up executor but doesn't configure persistent mode
    GpuExecutor executor;

    for (int i = 0; i < max_tokens; ++i) {
        executor.execute(graph);  // Copies everywhere!
    }
}
```

**Updated**:
```cpp
void AutoregressiveGenerator::generate(...) {
    GpuExecutor executor;

    if (!use_cpu) {
        // Enable GPU-persistent mode
        executor.setExecutionMode(GpuExecutor::ExecutionMode::GPU_PERSISTENT);
    }

    // Move input embeddings to GPU ONCE
    auto input_tensor = getTensor("input_ids");
    input_tensor->ensureOnGPU();

    // Execute generation loop
    for (int i = 0; i < max_tokens; ++i) {
        executor.execute(graph);  // Everything stays on GPU!
    }

    // Copy final output from GPU ONCE
    auto logits = getTensor("logits");
    logits->toCPU();
}
```

**Estimated Time**: 2-3 hours
**Risk**: Low

---

## Phase 6: Testing & Validation

### 6.1: Correctness Tests

```python
# scripts/test_persistent_gpu.py

import onnxruntime as ort
import numpy as np

def test_persistent_mode_correctness():
    """Compare GPU_PERSISTENT vs CPU_ONLY output"""

    # Run with CPU
    result_cpu = run_model(mode="cpu")

    # Run with GPU_PERSISTENT
    result_gpu = run_model(mode="gpu_persistent")

    # Should be identical (within FP32 tolerance)
    assert np.allclose(result_cpu, result_gpu, rtol=1e-5, atol=1e-6)

def test_memory_transfer_count():
    """Verify we're only copying twice"""

    # Instrument CUDA API with profiler
    # Count cudaMemcpy H2D and D2H calls

    with CUDAProfiler() as prof:
        generate_tokens(num_tokens=10)

    h2d_count = prof.count_memcpy_h2d()
    d2h_count = prof.count_memcpy_d2h()

    # Should be minimal: input embedding + final logits per token
    assert h2d_count <= 2, f"Too many H2D copies: {h2d_count}"
    assert d2h_count <= 2, f"Too many D2H copies: {d2h_count}"
```

**Testing Commands**:
```bash
# Correctness
python3 scripts/test_persistent_gpu.py

# Memory profiling
nvprof --print-gpu-trace ./build/onnx_gpu_engine model.onnx --generate

# Should see minimal cudaMemcpy calls
```

**Estimated Time**: 4-5 hours
**Risk**: Low

---

### 6.2: Performance Benchmarking

**Test**: SmolLM2-135M token generation

**Metrics to Track**:
- Tokens per second (target: 50-100 tps, 5-10x improvement)
- Total memory transfers (target: <10 for entire generation)
- Peak GPU memory usage
- Time breakdown per operation

```bash
# Before (GPU_COPY mode)
./build/onnx_gpu_engine model.onnx --generate --max-tokens 50

# After (GPU_PERSISTENT mode)
./build/onnx_gpu_engine model.onnx --generate --max-tokens 50 --gpu-persistent

# Expected: 5-15x faster token generation
```

**Estimated Time**: 2-3 hours
**Risk**: Low

---

## Implementation Timeline

| Phase | Description | Time Estimate | Dependencies |
|-------|-------------|---------------|--------------|
| **1.1** | Enhance Tensor class | 2-3 hours | None |
| **1.2** | GPU memory pool (optional) | 3-4 hours | Phase 1.1 |
| **2.1** | Add execution mode flag | 1 hour | Phase 1.1 |
| **2.2** | Update allocateOutput | 30 min | Phase 2.1 |
| **2.3** | Refactor operation pattern | 2-3 hours | Phase 2.2 |
| **3.1** | Persistent KV cache | 4-6 hours | Phase 1.1 |
| **3.2** | GPU-side KV cache kernel | 3-4 hours | Phase 3.1 |
| **4** | Refactor all operations | 30-45 hours | Phases 2-3 |
| **5** | Update generator | 2-3 hours | Phase 4 |
| **6.1** | Testing | 4-5 hours | Phase 5 |
| **6.2** | Benchmarking | 2-3 hours | Phase 6.1 |

**Total Estimated Time**: **54-78 hours** (7-10 working days)

---

## Incremental Rollout Strategy

Don't refactor everything at once! Use this approach:

### Week 1: Foundation (Phases 1-2)
- ✅ Enhance Tensor class
- ✅ Add execution mode
- ✅ Test with one simple operation (Add)

### Week 2: Critical Path (Phase 3-4 partial)
- ✅ Persistent KV cache
- ✅ Refactor: GroupQueryAttention
- ✅ Refactor: RotaryEmbedding
- ✅ Refactor: MatMul/Gemm
- ✅ Test end-to-end generation

**At this point, measure speedup! Should see 3-7x improvement.**

### Week 3: Complete (Phase 4-6)
- ✅ Refactor remaining operations
- ✅ Full validation
- ✅ Performance tuning

---

## Expected Results

### Memory Transfer Reduction

**Before (Current GPU_COPY mode)**:
- Per-token transfers for 12-layer model:
  - 12 layers × (Q/K/V proj + RoPE + Attention + Output proj + LayerNorm + Add)
  - ≈ 100-200 cudaMemcpy calls per token
  - Each H2D + D2H ≈ 2× the data size
  - Total: **Gigabytes of unnecessary transfers**

**After (GPU_PERSISTENT mode)**:
- Initial: Input embeddings to GPU (once per prompt)
- Per-token: Only position IDs to GPU (small)
- Final: Logits from GPU (per token)
- Total: **<10 transfers for entire generation**

**Reduction**: **20-40x fewer memory transfers**

---

### Performance Improvement

Conservative estimates:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory transfers/token | 100-200 | 2-3 | **50-100x** |
| Token generation time | 100-200ms | 10-30ms | **5-10x** |
| Tokens per second | 5-10 | 30-100 | **5-10x** |

**Still slower than Ollama** (377-454 tps) due to:
- Ollama uses quantization (4x advantage)
- Ollama has better kernel fusion
- Ollama uses FlashAttention

**But competitive with CPU-based inference** and good for a pure PyTorch-style ONNX engine!

---

## Risk Mitigation

### Backwards Compatibility
- Keep `GPU_COPY` mode as fallback
- Add `--gpu-persistent` flag (opt-in initially)
- Existing `--cpu` flag still works

### Debugging
- Add `--verbose-memory` flag to log all GPU allocations
- Add CUDA error checking after every operation
- Validate tensor device location at each step

### Incremental Testing
- Test each refactored operation individually
- Compare outputs against CPU reference
- Don't merge until full validation passes

---

## Success Criteria

✅ **Correctness**: Output matches CPU reference (within 1e-5 tolerance)
✅ **Performance**: 5-10x faster token generation
✅ **Memory**: <10 H2D/D2H transfers for full generation
✅ **Stability**: No memory leaks, no CUDA errors
✅ **Compatibility**: Old code paths still work

---

## Next Steps After Completion

Once persistent GPU memory is working, the next optimizations would be:

1. **Kernel Fusion** (2-3x gain)
   - Fuse Q/K/V projections into single matmul
   - Fuse residual + layernorm

2. **FP16 Mode** (2-3x gain)
   - Half precision inference
   - Tensor core utilization

3. **FlashAttention** (2-3x gain)
   - Memory-efficient attention
   - Especially beneficial for long sequences

4. **INT8 Quantization** (3-4x gain)
   - Dynamic quantization
   - Static quantization with calibration

Combined with persistent GPU memory, these could bring us to **50-200+ tokens/s**, competitive with most CPU-based inference engines.

---

## Conclusion

This plan focuses on the **single highest-impact optimization**: eliminating redundant memory transfers. By keeping tensors on GPU throughout execution, we can achieve **5-10x speedup** with relatively straightforward refactoring.

The architecture is designed to be:
- **Incremental**: Can refactor operations one at a time
- **Testable**: Each change is independently verifiable
- **Safe**: Backwards compatible with existing modes
- **Practical**: 7-10 days of focused work

Let's build a GPU engine that actually stays on the GPU! 🚀
