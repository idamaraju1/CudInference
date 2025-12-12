#!/usr/bin/env python3
"""
Generate a comprehensive ONNX model that exercises the core operations
implemented by the custom GPU engine (Cast, MatMul, Add/Sub/Mul, ReLU,
Sigmoid, Transpose, Gather, ReduceSum, Shape).
"""

import argparse
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def build_model(output_path: str):
    input_shape = [2, 3]
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)

    # Parameters
    W = numpy_helper.from_array(
        np.array([[0.2, -0.1, 0.5, 1.0],
                  [1.2,  0.3, 0.0, -0.7],
                  [-0.6, 0.8, 0.9, 0.4]], dtype=np.float32),
        name="W"
    )
    B = numpy_helper.from_array(
        np.array([[0.1, -0.2, 0.3, 0.0],
                  [0.05, 0.0, -0.1, 0.2]], dtype=np.float32),
        name="B"
    )
    scale = numpy_helper.from_array(
        np.array([[1.5, 0.5, -1.0, 2.0],
                  [0.8, -1.2, 0.6, 1.1]], dtype=np.float32),
        name="scale"
    )
    offset = numpy_helper.from_array(
        np.array([[0.05, -0.05, 0.1, 0.0],
                  [0.0,  0.2,  -0.1, 0.05]], dtype=np.float32),
        name="offset"
    )
    gather_indices = numpy_helper.from_array(np.array([1, 3], dtype=np.int64), name="gather_idx")
    reduce_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="reduce_axes")

    nodes = [
        helper.make_node("Cast", ["input"], ["cast_out"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["cast_out", "W"], ["matmul_out"]),
        helper.make_node("Add", ["matmul_out", "B"], ["add_out"]),
        helper.make_node("Relu", ["add_out"], ["relu_out"]),
        helper.make_node("Sigmoid", ["relu_out"], ["sigmoid_out"]),
        helper.make_node("Mul", ["sigmoid_out", "scale"], ["mul_out"]),
        helper.make_node("Sub", ["mul_out", "offset"], ["sub_out"]),
        helper.make_node("ReduceSum", ["sub_out", "reduce_axes"], ["reduce_out"], keepdims=1),
        helper.make_node("Transpose", ["sub_out"], ["trans_out"], perm=[1, 0]),
        helper.make_node("Gather", ["trans_out", "gather_idx"], ["gather_out"], axis=0),
        helper.make_node("Shape", ["sub_out"], ["shape_out"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="AllOpsModel",
        inputs=[X],
        outputs=[
            helper.make_tensor_value_info("sub_out", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("reduce_out", TensorProto.FLOAT, [2, 1]),
            helper.make_tensor_value_info("gather_out", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("shape_out", TensorProto.INT64, [2]),
        ],
        initializer=[W, B, scale, offset, gather_indices, reduce_axes],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=11,
        producer_name="all_ops_generator",
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved test model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create all-ops ONNX test model")
    parser.add_argument("--output", default="all_ops_test.onnx", help="Path to save the generated model")
    args = parser.parse_args()

    build_model(os.path.abspath(args.output))


if __name__ == "__main__":
    main()
