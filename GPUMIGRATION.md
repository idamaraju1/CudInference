# GPU Migration Plan

## Overview

This document outlines an incremental plan to migrate CPU-only operations to GPU implementations to improve performance and speed up token generation for LLM inference. Each step includes implementation details and incremental testing to ensure code correctness.

---

## Current Status Analysis

### ✅ Operations with GPU Kernels (Already Optimized)
1. **MatMul** - GPU kernel with cuBLAS fallback for large matrices
2. **ReLU** - GPU kernel with float4 vectorization
3. **Add** - GPU kernel with scalar broadcasting
4. **Sub** - GPU kernel with scalar broadcasting
5. **Mul** - GPU kernel in elementwise.cu
6. **SimplifiedLayerNormalization** - GPU kernel with proper variance calculation

### ⚠️ Operations with Partial GPU Support
7. **Gather** - Has GPU kernel (`launchGatherKernel`) but **always runs on CPU** (see `src/gpu/ops/executeGather.inl:120-134`)
8. **Transpose** - Has GPU kernel but may not be optimized

### ❌ Operations Running on CPU Only (Migration Targets)
9. **Sigmoid** - Pure CPU, simple activation function
10. **ReduceSum** - Pure CPU, used in normalization operations
11. **Cast** - Pure CPU, type conversion operation
12. **RotaryEmbedding** - Pure CPU, **CRITICAL FOR LLM PERFORMANCE** - used in every transformer layer
13. **GroupQueryAttention** - Pure CPU, **CRITICAL FOR LLM PERFORMANCE** - the main attention bottleneck
14. **SkipSimplifiedLayerNormalization** - CPU-only residual + normalization fusion

---

## Migration Priority (Ordered by Impact on Token Generation)

### Phase 1: Quick Wins (Low Complexity, Immediate Impact)
These operations are simple element-wise kernels that can be implemented quickly.

### Phase 2: Medium Complexity (Moderate Impact)
Operations requiring data movement or reduction patterns.

### Phase 3: Critical LLM Operations (High Complexity, Highest Impact)
Complex operations that are bottlenecks for autoregressive generation.

---

## Phase 1: Quick Wins

### Step 1.1: Fix Gather to Actually Use GPU Kernel ✨
**Priority: HIGH** | **Complexity: LOW** | **Impact: MEDIUM**

**Current Issue:**
- `executeGather.inl:120-134` always calls `launchGatherKernel` with `use_cpu=true`
- The GPU kernel exists but is never used in GPU mode

**Implementation:**
1. Modify `src/gpu/ops/executeGather.inl` to conditionally use GPU
2. Add GPU memory path for indices tensor (copy int64_t indices to device)
3. Update `launchGatherKernel` to support device pointers when `use_cpu=false`

**Testing:**
```bash
# Test with existing models
./build/onnx_gpu_engine model.onnx --verbose
./build/onnx_gpu_engine model.onnx --cpu --verbose

# Compare outputs - should be identical
python3 scripts/validate_onnx.py

# Benchmark CPU vs GPU
./build/onnx_gpu_engine model.onnx --benchmark
```

**Files to Modify:**
- `src/gpu/ops/executeGather.inl`
- `src/gpu/kernels/gather.cu` (if GPU kernel doesn't exist yet)

---

### Step 1.2: Implement GPU Sigmoid Kernel
**Priority: MEDIUM** | **Complexity: LOW** | **Impact: LOW-MEDIUM**

**Current Implementation:**
- `executeSigmoid.inl` uses CPU loop: `dst[i] = 1.0f / (1.0f + std::exp(-src[i]))`

**Implementation:**
1. Create `src/gpu/kernels/sigmoid.cu`
2. Implement `__global__ void sigmoidKernel(const float* input, float* output, int size)`
3. Use vectorized loads (float4) for better memory throughput
4. Add launcher function `void launchSigmoid(const float* input, float* output, int size, cudaStream_t stream)`
5. Add CPU fallback `void sigmoidCPU(const float* input, float* output, int size)`
6. Add multi-threaded CPU fallback `void sigmoidCPUMultiThreaded(...)`
7. Update `executeSigmoid.inl` to use GPU kernel when not in fallback mode
8. Declare in `kernels.cuh`

**Kernel Pseudo-code:**
```cuda
__global__ void sigmoidKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}
```

**Testing:**
```bash
# Create test model with Sigmoid operation
python3 scripts/create_sigmoid_test.py

# Validate correctness
python3 scripts/validate_onnx.py

# Benchmark
./build/onnx_gpu_engine sigmoid_test.onnx --benchmark
```

**Files to Create/Modify:**
- `src/gpu/kernels/sigmoid.cu` (new)
- `src/gpu/kernels/kernels.cuh` (add declarations)
- `src/gpu/ops/executeSigmoid.inl` (modify to use GPU kernel)
- `CMakeLists.txt` (add sigmoid.cu to CUDA_SOURCES)

---

### Step 1.3: Implement GPU Cast Kernel
**Priority: LOW** | **Complexity: LOW** | **Impact: LOW**

**Current Implementation:**
- `executeCast.inl` uses CPU template functions for type conversion

**Implementation:**
1. Create `src/gpu/kernels/cast.cu`
2. Implement templated CUDA kernels for common conversions:
   - FLOAT32 ↔ INT32
   - FLOAT32 ↔ INT64
   - INT32 ↔ INT64
3. Use template specialization for each source/target type pair
4. Add launcher function and CPU fallback
5. Update `executeCast.inl` to use GPU kernel

**Testing:**
```bash
python3 scripts/create_cast_test.py
python3 scripts/validate_onnx.py
./build/onnx_gpu_engine cast_test.onnx --benchmark
```

**Files to Create/Modify:**
- `src/gpu/kernels/cast.cu` (new)
- `src/gpu/kernels/kernels.cuh` (add declarations)
- `src/gpu/ops/executeCast.inl` (modify)
- `CMakeLists.txt` (add cast.cu)

---

## Phase 2: Medium Complexity Operations

### Step 2.1: Implement GPU ReduceSum Kernel
**Priority: MEDIUM** | **Complexity: MEDIUM** | **Impact: MEDIUM**

**Current Implementation:**
- `executeReduceSum.inl` uses nested CPU loops with coordinate mapping

**Implementation:**
1. Create `src/gpu/kernels/reducesum.cu`
2. Implement reduction using:
   - **Approach A (Simple):** Atomic adds on output tensor
   - **Approach B (Optimized):** Shared memory reduction with warp shuffle
3. Handle multi-axis reduction
4. Support keepdims parameter
5. Handle different data types (FLOAT32, INT64)

**Kernel Strategy:**
- For single-axis reduction: Use optimized warp-level primitives
- For multi-axis: Chain single-axis reductions or use atomic adds
- Use `__shfl_down_sync` for warp-level reduction

**Testing:**
```bash
python3 scripts/create_reducesum_test.py  # Various reduction patterns
python3 scripts/validate_onnx.py
./build/onnx_gpu_engine reducesum_test.onnx --benchmark

# Test edge cases
# - Single element reduction
# - Full tensor reduction (all axes)
# - Multi-axis reduction with keepdims
```

**Files to Create/Modify:**
- `src/gpu/kernels/reducesum.cu` (new)
- `src/gpu/kernels/kernels.cuh` (add declarations)
- `src/gpu/ops/executeReduceSum.inl` (modify)
- `CMakeLists.txt` (add reducesum.cu)

---

### Step 2.2: Optimize Transpose Kernel
**Priority: MEDIUM** | **Complexity: MEDIUM** | **Impact: MEDIUM**

**Current Status:**
- `launchTransposeKernel` exists but may not be optimized for all transpose patterns

**Implementation:**
1. Audit current `tensor_ops.cu` transpose implementation
2. Add shared memory tiling for 2D transposes to avoid bank conflicts
3. Optimize for common patterns (matrix transpose, dimension swaps)
4. Add specialized kernels for:
   - 2D transpose (most common)
   - 3D/4D permutations (common in attention)

**Testing:**
```bash
python3 scripts/create_transpose_test.py
python3 scripts/validate_onnx.py
./build/onnx_gpu_engine transpose_test.onnx --benchmark
```

**Files to Modify:**
- `src/gpu/kernels/tensor_ops.cu` (optimize existing kernel)

---

## Phase 3: Critical LLM Operations (Highest Impact)

### Step 3.1: Implement GPU RotaryEmbedding Kernel ⚡
**Priority: CRITICAL** | **Complexity: MEDIUM-HIGH** | **Impact: VERY HIGH**

**Why Critical:**
- Used in **every transformer layer** during token generation
- Called on every forward pass for Q/K projections
- Current CPU implementation processes cos/sin cache on host

**Current Implementation Analysis:**
- `executeRotaryEmbedding.inl` performs:
  1. Position lookup from cos/sin caches (lines 174-277)
  2. RoPE rotation computation (lines 305-336)
  3. Dimension reordering for BNSH ↔ BSNH layouts (lines 289-303, 349-363)
- All operations on CPU with manual loops

**Implementation Strategy:**

**Stage 3.1a: Basic GPU RoPE Kernel**
1. Create `src/gpu/kernels/rotary_embedding.cu`
2. Implement `ropeKernel` that:
   - Takes input tensor [batch, seq, num_heads, head_size]
   - Takes cos/sin values [batch, seq, rotary_half]
   - Performs rotation: `x_rot[i] = x[i] * cos[i] - x[i+half] * sin[i]`
   - Supports both interleaved and non-interleaved modes
3. Move cos/sin cache selection to GPU

**Stage 3.1b: Optimize for Autoregressive Generation**
1. Add KV-cache optimized path for single-token inference
2. Pre-upload cos/sin caches to GPU device memory (avoid repeated transfers)
3. Add position lookup kernel for dynamic position_ids

**Kernel Pseudo-code:**
```cuda
__global__ void ropeKernel(
    const float* input,        // [B, S, H, D]
    const float* cos_cache,    // [B, S, rotary_half]
    const float* sin_cache,    // [B, S, rotary_half]
    float* output,             // [B, S, H, D]
    int batch, int seq, int num_heads, int head_size,
    int rotary_dim, bool interleaved
) {
    int token_idx = blockIdx.x;  // batch * seq
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= batch * seq || head_idx >= num_heads) return;

    const float* input_head = input + (token_idx * num_heads + head_idx) * head_size;
    float* output_head = output + (token_idx * num_heads + head_idx) * head_size;
    const float* cos_vals = cos_cache + token_idx * (rotary_dim / 2);
    const float* sin_vals = sin_cache + token_idx * (rotary_dim / 2);

    int rotary_half = rotary_dim / 2;

    if (!interleaved) {
        // Non-interleaved: first half and second half
        if (tid < rotary_half) {
            float x1 = input_head[tid];
            float x2 = input_head[tid + rotary_half];
            float c = cos_vals[tid];
            float s = sin_vals[tid];
            output_head[tid] = x1 * c - x2 * s;
            output_head[tid + rotary_half] = x2 * c + x1 * s;
        }
    } else {
        // Interleaved: pairs at [2i, 2i+1]
        if (tid < rotary_half) {
            int even_idx = tid * 2;
            int odd_idx = even_idx + 1;
            float x_even = input_head[even_idx];
            float x_odd = input_head[odd_idx];
            float c = cos_vals[tid];
            float s = sin_vals[tid];
            output_head[even_idx] = x_even * c - x_odd * s;
            output_head[odd_idx] = x_odd * c + x_even * s;
        }
    }

    // Copy non-rotated dimensions
    for (int i = tid + rotary_dim; i < head_size; i += blockDim.x) {
        output_head[i] = input_head[i];
    }
}
```

**Testing Strategy:**
```bash
# Unit test with SmolLM2 or similar LLM
python3 scripts/export_smollm_onnx.py

# Test prefill phase (long sequence)
./build/onnx_gpu_engine smollm.onnx --input "The sky is blue" --tokenizer tokenizer.json --generate --max-tokens 1 --verbose

# Test decode phase (single token, critical for speed)
./build/onnx_gpu_engine smollm.onnx --input "Hello" --tokenizer tokenizer.json --generate --max-tokens 10 --verbose

# Validate against ONNX Runtime
python3 scripts/validate_rope.py

# Benchmark impact on token generation
./build/onnx_gpu_engine smollm.onnx --input "Test" --tokenizer tokenizer.json --generate --max-tokens 50 --benchmark
```

**Success Metrics:**
- ✅ Numerical accuracy: max error < 1e-5 vs ONNX Runtime
- ✅ Performance: 5-10x speedup vs CPU for single token decode
- ✅ Token generation: measurable improvement in tokens/sec

**Files to Create/Modify:**
- `src/gpu/kernels/rotary_embedding.cu` (new)
- `src/gpu/kernels/kernels.cuh` (add declarations)
- `src/gpu/ops/executeRotaryEmbedding.inl` (major refactor)
- `CMakeLists.txt` (add rotary_embedding.cu)

---

### Step 3.2: Implement GPU GroupQueryAttention Kernel ⚡⚡⚡
**Priority: CRITICAL** | **Complexity: VERY HIGH** | **Impact: MAXIMUM**

**Why Critical:**
- **THE MAIN BOTTLENECK** for LLM token generation
- Performs the expensive attention computation
- Involves multiple matrix multiplications and softmax
- Benefits greatly from GPU parallelism

**Current Implementation Analysis:**
- `executeGroupQueryAttention.inl` performs:
  1. KV-cache concatenation (past + current)
  2. Q·K^T matrix multiplication
  3. Scaling + optional masking
  4. Softmax normalization
  5. Attention·V multiplication
  6. Output projection

**Implementation Strategy:**

This is the most complex operation. Break into sub-stages:

**Stage 3.2a: Basic Flash Attention (Simplified)**
1. Create `src/gpu/kernels/attention.cu`
2. Implement basic attention kernel:
   ```
   scores = Q @ K^T / sqrt(head_dim)
   attn = softmax(scores)
   output = attn @ V
   ```
3. Use cuBLAS for matrix multiplications (Q@K^T and attn@V)
4. Implement GPU softmax kernel

**Stage 3.2b: Add Group Query Attention Support**
1. Handle KV head broadcasting (num_q_heads > num_kv_heads)
2. Each KV head serves multiple Q heads (grouped query)

**Stage 3.2c: KV-Cache Optimization**
1. Implement efficient KV-cache concatenation on GPU
2. Pre-allocate KV cache buffers to avoid repeated allocations
3. Update cache in-place during autoregressive generation

**Stage 3.2d: Fused Kernel Optimization**
1. Fuse scaling + masking + softmax into single kernel
2. Consider Flash Attention-style tiling for large sequences
3. Use shared memory for intermediate results

**Kernel Pseudo-code (Simplified):**
```cuda
// This is a simplified version - real implementation needs tiling
__global__ void groupQueryAttentionKernel(
    const float* Q,           // [batch, q_heads, seq_q, head_dim]
    const float* K,           // [batch, kv_heads, seq_kv, head_dim]
    const float* V,           // [batch, kv_heads, seq_kv, head_dim]
    float* output,            // [batch, q_heads, seq_q, head_dim]
    int batch, int q_heads, int kv_heads,
    int seq_q, int seq_kv, int head_dim,
    float scale
) {
    // This would need significant optimization with shared memory,
    // tiling, and possibly flash attention techniques
    // Consider using cuBLAS for matmuls and custom kernels for softmax
}
```

**Recommended Approach:**
Given complexity, consider:
1. **Option A:** Use cuBLAS for matmuls + custom softmax kernel (easier)
2. **Option B:** Implement Flash Attention (optimal but very complex)
3. **Option C:** Use existing libraries (cuDNN attention or vendor SDKs)

**Testing Strategy:**
```bash
# Create attention-specific test
python3 scripts/create_attention_test.py

# Test with actual LLM model
./build/onnx_gpu_engine smollm.onnx --input "Test prompt" --tokenizer tokenizer.json --generate --max-tokens 20 --verbose

# Incremental validation at each stage
python3 scripts/validate_attention.py  # Compare Q@K^T, softmax, output separately

# Profile with nsys
nsys profile ./build/onnx_gpu_engine smollm.onnx --input "Test" --tokenizer tokenizer.json --generate --max-tokens 10

# Benchmark tokens/second improvement
./build/onnx_gpu_engine smollm.onnx --input "The" --tokenizer tokenizer.json --generate --max-tokens 100 --benchmark
```

**Success Metrics:**
- ✅ Correctness: matches ONNX Runtime output within 1e-4
- ✅ Performance: 10-20x speedup vs CPU for decode phase
- ✅ Token generation: 2-5x improvement in tokens/sec overall
- ✅ Memory: efficient KV cache management

**Files to Create/Modify:**
- `src/gpu/kernels/attention.cu` (new)
- `src/gpu/kernels/softmax.cu` (new, if not exists)
- `src/gpu/kernels/kernels.cuh` (add declarations)
- `src/gpu/ops/executeGroupQueryAttention.inl` (major refactor)
- `CMakeLists.txt` (add attention.cu, softmax.cu)

**References:**
- Flash Attention paper: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691
- cuDNN attention documentation

---

### Step 3.3: Implement GPU SkipSimplifiedLayerNormalization
**Priority: MEDIUM-HIGH** | **Complexity: MEDIUM** | **Impact: MEDIUM-HIGH**

**Current Implementation:**
- Likely CPU-only residual connection + layer norm fusion

**Implementation:**
1. Create `src/gpu/kernels/skip_layernorm.cu`
2. Fuse residual addition + normalization in single kernel
3. Reuse existing layer norm patterns from `layernorm.cu`
4. Formula: `output = LayerNorm(input + skip)`

**Testing:**
```bash
python3 scripts/create_skip_layernorm_test.py
python3 scripts/validate_onnx.py
./build/onnx_gpu_engine skip_test.onnx --benchmark
```

**Files to Create/Modify:**
- `src/gpu/kernels/skip_layernorm.cu` (new)
- `src/gpu/kernels/kernels.cuh`
- `src/gpu/ops/executeSkipSimplifiedLayerNormalization.inl`
- `CMakeLists.txt`

---

## Phase 4: Advanced Optimizations (Post-Migration)

### Step 4.1: Kernel Fusion Opportunities
After all operations run on GPU, identify fusion opportunities:
- Fuse MatMul + ReLU (common in FFN layers)
- Fuse LayerNorm + MatMul (common pattern)
- Fuse multiple element-wise ops

### Step 4.2: Memory Layout Optimization
- Add support for NCHW/NHWC layout conversions
- Optimize for tensor core usage (require specific alignments)

### Step 4.3: Multi-Stream Execution
- Pipeline independent operations across CUDA streams
- Overlap compute and memory transfers

---

## Testing Guidelines

### For Each Migration Step:

1. **Unit Test Creation**
   ```bash
   python3 scripts/create_<op>_test.py
   # Creates minimal ONNX model testing the specific operation
   ```

2. **Numerical Validation**
   ```bash
   python3 scripts/validate_onnx.py
   # Compares C++ output vs ONNX Runtime reference
   # Should show max error < 1e-5 for FLOAT32
   ```

3. **CPU Fallback Test**
   ```bash
   ./build/onnx_gpu_engine model.onnx --cpu --verbose
   # Ensure CPU fallback still works
   ```

4. **GPU Execution Test**
   ```bash
   ./build/onnx_gpu_engine model.onnx --verbose
   # Test GPU path
   ```

5. **Benchmark Comparison**
   ```bash
   ./build/onnx_gpu_engine model.onnx --benchmark
   # Compare CPU (1-N threads) vs GPU performance
   ```

6. **Integration Test with LLM**
   ```bash
   ./build/onnx_gpu_engine smollm.onnx --input "Hello world" --tokenizer tokenizer.json --generate --max-tokens 50
   # Ensure token generation still works correctly
   ```

### Regression Testing After Each Step:
- Run all existing test models
- Verify no performance degradation in already-optimized ops
- Check memory usage hasn't increased significantly

---

## Expected Performance Gains

### After Phase 1 (Quick Wins):
- **Gather fix:** 2-5x speedup for embedding lookups
- **Sigmoid GPU:** 3-10x speedup for activation
- Overall token generation: +5-10% faster

### After Phase 2 (Medium Complexity):
- **ReduceSum GPU:** 3-8x speedup for reduction operations
- Overall token generation: +10-15% faster (cumulative)

### After Phase 3 (Critical Ops):
- **RoPE GPU:** 5-10x speedup for rotary embedding
- **Attention GPU:** 10-20x speedup for attention computation
- **Overall token generation:** **2-5x faster end-to-end** 🚀

### After Phase 4 (Optimizations):
- Kernel fusion: Additional 10-20% improvement
- Multi-stream: Additional 5-15% improvement
- **Total expected improvement: 3-7x faster token generation** 🎯

---

## Risk Mitigation

### Numerical Stability Risks:
- Use `expf()`, `logf()` for float operations (not double)
- Add epsilon to denominators in normalization
- Use Kahan summation for long reductions if needed
- Validate against reference implementations at each step

### Memory Management Risks:
- Use CUDA_CHECK macro for all CUDA API calls
- Leverage RAII Tensor class for automatic cleanup
- Profile with `cuda-memcheck` to catch leaks
- Test with large models to catch OOM issues

### Performance Risks:
- Profile before and after with `nsys`/`nvprof`
- Watch for kernel launch overhead (batch small operations)
- Monitor memory bandwidth utilization
- Use cuBLAS for large matrix ops (don't reinvent)

---

## Tools and Commands Reference

### Build Commands:
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

### Profiling:
```bash
# NVIDIA Nsight Systems
nsys profile --stats=true ./build/onnx_gpu_engine model.onnx

# CUDA Memcheck
cuda-memcheck ./build/onnx_gpu_engine model.onnx

# Compute Sanitizer (newer)
compute-sanitizer ./build/onnx_gpu_engine model.onnx
```

### Debugging:
```bash
# Enable debug logging
./build/onnx_gpu_engine model.onnx --debug

# GDB with CUDA
cuda-gdb ./build/onnx_gpu_engine
```

---

## Summary Checklist

### Phase 1: ✅ Quick Wins
- [ ] Fix Gather GPU execution
- [ ] Implement Sigmoid GPU kernel
- [ ] Implement Cast GPU kernel

### Phase 2: ⏳ Medium Complexity
- [ ] Implement ReduceSum GPU kernel
- [ ] Optimize Transpose kernel

### Phase 3: 🎯 Critical LLM Operations
- [ ] Implement RoPE GPU kernel (HIGH PRIORITY)
- [ ] Implement GroupQueryAttention GPU kernel (HIGHEST PRIORITY)
- [ ] Implement SkipSimplifiedLayerNormalization GPU kernel

### Phase 4: 🚀 Advanced Optimizations
- [ ] Identify and implement kernel fusion
- [ ] Optimize memory layouts
- [ ] Implement multi-stream execution

---

## Conclusion

This migration plan prioritizes operations based on their impact on LLM token generation performance. The most critical operations are **RotaryEmbedding** and **GroupQueryAttention**, which together account for the majority of computation time during autoregressive generation.

By following this incremental approach with thorough testing at each step, we can safely migrate operations to GPU while maintaining correctness and avoiding regressions. The expected end result is **3-7x faster token generation** for LLM inference workloads.

**Next Steps:**
1. Start with Phase 1 (quick wins) to build confidence
2. Move to Phase 3.1 (RoPE) as soon as possible - high impact, medium complexity
3. Tackle Phase 3.2 (Attention) - highest impact but most complex
4. Fill in remaining operations and optimizations

Good luck! 🚀
