cmake_minimum_required(VERSION 3.18)

# Option to build CPU-only version
option(CPU_ONLY "Build without GPU support" OFF)

# Project declaration with conditional CUDA
if(CPU_ONLY)
    project(OnnxRunner LANGUAGES CXX)
else()
    project(OnnxRunner LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Only set CUDA standard if GPU build
if(NOT CPU_ONLY)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")  # Adjust as needed
endif()

# Define CPU_ONLY macro in source code if enabled
if(CPU_ONLY)
    add_definitions(-DCPU_ONLY)
endif()

# Find packages
find_package(Protobuf REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT CPU_ONLY)
    find_package(CUDAToolkit REQUIRED)
endif()

# ONNX proto sources
set(ONNX_PROTO_DIR ${CMAKE_SOURCE_DIR}/third_party/onnx)
set(ONNX_PROTO_SOURCES
    ${ONNX_PROTO_DIR}/onnx.pb.cc
)
add_library(onnx_proto ${ONNX_PROTO_SOURCES})
target_include_directories(onnx_proto PUBLIC ${ONNX_PROTO_DIR})
target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)

# Core sources
set(CORE_SOURCES
    src/core/model_parser.cpp
    src/core/graph.cpp
    src/core/node.cpp
)

# Utils sources
set(UTILS_SOURCES
    src/utils/tensor.cpp
    src/utils/logger.cpp
    src/utils/tensor/cpu_tensor.cpp
)

# Executor sources
set(EXECUTOR_SOURCES
    src/executors/cpu_executor.cpp
)
if(NOT CPU_ONLY)
    list(APPEND EXECUTOR_SOURCES
        src/executors/gpu_executor.cpp
    )
endif()

# GPU sources (only if GPU build)
set(GPU_SOURCES 
    src/gpu/autoregressive_generator.cpp)
set(CUDA_SOURCES "")
if(NOT CPU_ONLY)
    set(GPU_SOURCES
        src/gpu/gpu_executor.cpp
        src/gpu/benchmark.cpp
        src/gpu/autoregressive_generator.cpp
    )

    set(CUDA_SOURCES
        src/gpu/kernels/matmul.cu
        src/gpu/kernels/relu.cu
        src/gpu/kernels/add.cu
        src/gpu/kernels/gather.cu
        src/gpu/kernels/elementwise.cu
        src/gpu/kernels/reducemean.cu
        src/gpu/kernels/tensor_ops.cu
        src/gpu/kernels/layernorm.cu
        src/gpu/kernels/sub.cu
    )
endif()

# Main executable
add_executable(onnx_gpu_engine
    src/main.cpp
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${EXECUTOR_SOURCES}
    ${GPU_SOURCES}
    ${CUDA_SOURCES}
)

# Include directories
target_include_directories(onnx_gpu_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${ONNX_PROTO_DIR}
)

# Linking
if(CPU_ONLY)
    target_link_libraries(onnx_gpu_engine PRIVATE
        onnx_proto
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(onnx_gpu_engine PRIVATE
        onnx_proto
        CUDA::cudart
        CUDA::cublas
        OpenMP::OpenMP_CXX
    )
endif()

# Compiler options
target_compile_options(onnx_gpu_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -fopenmp>
)

# CUDA target properties
if(NOT CPU_ONLY)
    set_target_properties(onnx_gpu_engine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    set_property(TARGET onnx_gpu_engine PROPERTY CUDA_LINK_FLAGS "-Xcompiler -fopenmp")
endif()
