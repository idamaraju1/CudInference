#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 16:31:59 2025

@author: aiden
"""
import onnx
from onnx import helper, TensorProto

# Define inputs and outputs
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

# Create weights and biases as initializers
W = helper.make_tensor('W', TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0])  # Identity matrix
b = helper.make_tensor('b', TensorProto.FLOAT, [2], [0.5, -0.5])

# Create a single node (Gemm = linear)
node_def = helper.make_node(
    'Gemm',
    inputs=['input', 'W', 'b'],
    outputs=['output']
)

# Create the graph
graph_def = helper.make_graph(
    [node_def],
    'simple-linear-model',
    [input_tensor],
    [output_tensor],
    [W, b]
)

# Create the model
model_def = helper.make_model(graph_def, producer_name='onnx-example')

# Save the model
onnx.save(model_def, 'simple_model.onnx')

print("ONNX model saved as simple_model.onnx")
