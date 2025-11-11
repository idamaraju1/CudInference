cmake_minimum_required(VERSION 3.18)
project(OnnxRunner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - adjust based on your GPU
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

# Workaround for glibc compatibility with CUDA
# Suppress cudafe++ errors about exception specification mismatches
# NOTE: These flags cause "too many arguments" error with CUDA 12.4 + gcc-13
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20014,20011")

# Find required packages
find_package(Protobuf REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# ONNX protobuf files
# Instructions: Download onnx.proto and onnx-ml.proto from https://github.com/onnx/onnx
# Place them in third_party/onnx/ and run:
# protoc --cpp_out=. onnx.proto onnx-ml.proto
set(ONNX_PROTO_DIR ${CMAKE_SOURCE_DIR}/third_party/onnx)
set(ONNX_PROTO_SOURCES
    ${ONNX_PROTO_DIR}/onnx.pb.cc
)

# Create onnx proto library
add_library(onnx_proto ${ONNX_PROTO_SOURCES})
target_include_directories(onnx_proto PUBLIC ${ONNX_PROTO_DIR})
target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)

# Source files
set(CORE_SOURCES
    src/core/model_parser.cpp
    src/core/graph.cpp
    src/core/node.cpp
)

set(UTILS_SOURCES
    src/utils/tensor.cpp
    src/utils/logger.cpp
)

set(GPU_SOURCES
    src/gpu/gpu_executor.cpp
    src/gpu/benchmark.cpp
)

set(CUDA_SOURCES
    src/gpu/kernels/matmul.cu
    src/gpu/kernels/relu.cu
    src/gpu/kernels/add.cu
)

# Main executable
add_executable(onnx_gpu_engine
    src/main.cpp
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
    ${CUDA_SOURCES}
)

target_include_directories(onnx_gpu_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${ONNX_PROTO_DIR}
)

target_link_libraries(onnx_gpu_engine PRIVATE
    onnx_proto
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    /lib/x86_64-linux-gnu/libstdc++.so.6
)

# Compiler options
target_compile_options(onnx_gpu_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -fopenmp>
)

# Enable separate compilation for CUDA
set_target_properties(onnx_gpu_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    BUILD_RPATH "/lib/x86_64-linux-gnu"
    INSTALL_RPATH "/lib/x86_64-linux-gnu"
    LINK_FLAGS "-Wl,-rpath,/lib/x86_64-linux-gnu"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add OpenMP flags for CUDA device link step
set_property(TARGET onnx_gpu_engine PROPERTY CUDA_LINK_FLAGS "-Xcompiler -fopenmp")
