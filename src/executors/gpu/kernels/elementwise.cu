#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>


namespace onnx_runner {

// ============================================================================
// Mul (Element-wise Multiplication)
// ============================================================================

__global__ void mulKernel(const float* __restrict__ A, 
                          const float* __restrict__ B, 
                          float* __restrict__ C, 
                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// vectorized kernel - 4 elements per thread when aligned.
__global__ void mulVectorKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx4]);
        float4 b = *reinterpret_cast<const float4*>(&B[idx4]);
        float4 c;
        c.x = a.x * b.x;
        c.y = a.y * b.y;
        c.z = a.z * b.z;
        c.w = a.w * b.w;
        *reinterpret_cast<float4*>(&C[idx4]) = c;
    } else if (idx4 < size) {
        // tail
        for (int i = idx4; i < size; ++i) {
            C[i] = A[i] * B[i];
        }
    }
}

__global__ void mulScalarKernel(const float* __restrict__ A, float scalar, float* __restrict__ C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}

__global__ void mulScalarVectorKernel(const float* __restrict__ A,
                                      float scalar,
                                      float* __restrict__ C,
                                      int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx4]);
        a.x *= scalar;
        a.y *= scalar;
        a.z *= scalar;
        a.w *= scalar;
        *reinterpret_cast<float4*>(&C[idx4]) = a;
    } else if (idx4 < size) {
        for (int i = idx4; i < size; ++i) {
            C[i] = A[i] * scalar;
        }
    }
}

void mulCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] * B[i];
    }
}

void launchMulKernel(const float* A, const float* B, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        mulCPU(A, B, C, size, num_threads);
    } else {
        int block_size = 256;
        // vectorized path if large and 16 byet aligned.
        bool aligned =
            (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(C) % 16 == 0);
        
        if (aligned && size >= 1024) {
            int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
            mulVectorKernel<<<grid_size, block_size>>>(A, B, C, size);
        } else {
            int grid_size = (size + block_size - 1) / block_size;
            mulKernel<<<grid_size, block_size>>>(A, B, C, size);
        }

        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Mul kernel launch failed: ") +
                                    cudaGetErrorString(err));
        }
    }
}

void launchMulScalarKernel(const float* A, float scalar, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < size; ++i) {
            C[i] = A[i] * scalar;
        }
    } else {
        int block_size = 256;
        bool aligned =
            (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(C) % 16 == 0);
        
        if (aligned && size >= 1024) {
            int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
            mulScalarVectorKernel<<<grid_size, block_size>>>(A, scalar, C, size);
        } else {
            int grid_size = (size + block_size - 1) / block_size;
            mulScalarKernel<<<grid_size, block_size>>>(A, scalar, C, size);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("MulScalar kernel launch failed: ") +
                                    cudaGetErrorString(err));
        }
    }
}

// ============================================================================
// Div (Element-wise Division)
// ============================================================================

__global__ void divKernel(const float* __restrict__ A, 
                          const float* __restrict__ B, 
                          float* __restrict__ C, 
                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] / B[idx];
    }
}

__global__ void divVectorKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx4]);
        float4 b = *reinterpret_cast<const float4*>(&B[idx4]);
        float4 c;
        c.x = a.x / b.x;
        c.y = a.y / b.y;
        c.z = a.z / b.z;
        c.w = a.w / b.w;
        *reinterpret_cast<float4*>(&C[idx4]) = c;
    } else if (idx4 < size) {
        for (int i = idx4; i < size; ++i) {
            C[i] = A[i] / B[i];
        }
    }
}

__global__ void divScalarKernel(const float* __restrict__ A, float scalar, float* __restrict__ C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] / scalar;
    }
}

__global__ void divScalarVectorKernel(const float* __restrict__ A,
                                      float scalar,
                                      float* __restrict__ C,
                                      int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float inv = 1.0f / scalar;  // precompute
    if (idx4 + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx4]);
        a.x *= inv;
        a.y *= inv;
        a.z *= inv;
        a.w *= inv;
        *reinterpret_cast<float4*>(&C[idx4]) = a;
    } else if (idx4 < size) {
        for (int i = idx4; i < size; ++i) {
            C[i] = A[i] * inv;
        }
    }
}

void divCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] / B[i];
    }
}

void launchDivKernel(const float* A, const float* B, float* C, int size, bool use_cpu, int num_threads) {
    // fast exit
    if (size <= 0) return;

    if (use_cpu) {
        divCPU(A, B, C, size, num_threads);
        return;
    } 

    const int block_size = 256;
    bool aligned =
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (aligned && size >= 1024) {
        int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
        divVectorKernel<<<grid_size, block_size>>>(A, B, C, size);
    } else {
        int grid_size = (size + block_size - 1) / block_size;
        divKernel<<<grid_size, block_size>>>(A, B, C, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Div kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

void launchDivScalarKernel(const float* A, float scalar, float* C, int size, bool use_cpu, int num_threads) {
    
    if (size <= 0) return;
    if (use_cpu) {
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < size; ++i) {
            C[i] = A[i] / scalar;
        }
        return;
    }
        
    const int block_size = 256;
    bool aligned = 
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (aligned && size >= 1024) {
        int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
        divScalarVectorKernel<<<grid_size, block_size>>>(A, scalar, C, size);
    } else {
        int grid_size = (size + block_size - 1) / block_size;
        divScalarKernel<<<grid_size, block_size>>>(A, scalar, C, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("DivScalar kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

// ============================================================================
// Pow (Element-wise Power)
// ============================================================================

__global__ void powKernel(const float* __restrict__ A, 
                          const float* __restrict__ B, 
                          float* __restrict__ C, 
                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = powf(A[idx], B[idx]);
    }
}

__global__ void powVectorKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx4]);
        float4 b = *reinterpret_cast<const float4*>(&B[idx4]);
        float4 c;
        c.x = powf(a.x, b.x);
        c.y = powf(a.y, b.y);
        c.z = powf(a.z, b.z);
        c.w = powf(a.w, b.w);
        *reinterpret_cast<float4*>(&C[idx4]) = c;
    } else if (idx4 < size) {
        for (int i = idx4; i < size; ++i) {
            C[i] = powf(A[i], B[i]);
        }
    }
}

__global__ void powScalarKernel(const float* __restrict__ A, float exponent, float* __restrict__ C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = powf(A[idx], exponent);
    }
}

__global__ void powScalarVectorKernel(const float* __restrict__ A,
                                      float exponent,
                                      float* __restrict__ C,
                                      int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx4]);
        float4 c;
        c.x = powf(a.x, exponent);
        c.y = powf(a.y, exponent);
        c.z = powf(a.z, exponent);
        c.w = powf(a.w, exponent);
        *reinterpret_cast<float4*>(&C[idx4]) = c;
    } else if (idx4 < size) {
        for (int i = idx4; i < size; ++i) {
            C[i] = powf(A[i], exponent);
        }
    }
}

void powCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = std::pow(A[i], B[i]);
    }
}

void launchPowKernel(const float* A, const float* B, float* C, int size, bool use_cpu, int num_threads) {
    // fast exit.
    if (size <= 0) return;

    if (use_cpu) {
        powCPU(A, B, C, size, num_threads);
        return;
    } 
    const int block_size = 256;
    bool aligned =
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (aligned && size >= 1024) {
        int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
        powVectorKernel<<<grid_size, block_size>>>(A, B, C, size);
    } else {
        int grid_size = (size + block_size - 1) / block_size;
        powKernel<<<grid_size, block_size>>>(A, B, C, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Pow kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

void launchPowScalarKernel(const float* A, float exponent, float* C, int size, bool use_cpu, int num_threads) {
    if (size <= 0) return;

    if (use_cpu) {
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < size; ++i) {
            C[i] = std::pow(A[i], exponent);
        }
        return;
    } 
    const int block_size = 256;
    
    bool aligned =
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (aligned && size >= 1024) {
        int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
        powScalarVectorKernel<<<grid_size, block_size>>>(A, exponent, C, size);
    } else {
        int grid_size = (size + block_size - 1) / block_size;
        powScalarKernel<<<grid_size, block_size>>>(A, exponent, C, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("PowScalar kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

// ============================================================================
// Sqrt (Element-wise Square Root)
// ============================================================================

__global__ void sqrtKernel(const float* __restrict__ A, float* __restrict__ C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = sqrtf(A[idx]);
    }
}

__global__ void sqrtVectorKernel(const float* __restrict__ A,
                                 float* __restrict__ C,
                                 int size) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 + 3 < size) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx4]);
        a.x = sqrtf(a.x);
        a.y = sqrtf(a.y);
        a.z = sqrtf(a.z);
        a.w = sqrtf(a.w);
        *reinterpret_cast<float4*>(&C[idx4]) = a;
    } else if (idx4 < size) {
        for (int i = idx4; i < size; ++i) {
            C[i] = sqrtf(A[i]);
        }
    }
}

void sqrtCPU(const float* A, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = std::sqrt(A[i]);
    }
}

void launchSqrtKernel(const float* A, float* C, int size, bool use_cpu, int num_threads) {
    if (size <= 0) return;

    if (use_cpu) {
        sqrtCPU(A, C, size, num_threads);
        return;
    } 
    const int block_size = 256;

    bool aligned =
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (aligned && size >= 1024) {
        int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
        sqrtVectorKernel<<<grid_size, block_size>>>(A, C, size);
    } else {
        int grid_size = (size + block_size - 1) / block_size;
        sqrtKernel<<<grid_size, block_size>>>(A, C, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Sqrt kernel launch failed: ") +
                                 cudaGetErrorString(err));
    
    }
}

} // namespace onnx_runner
