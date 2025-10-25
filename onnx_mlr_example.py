#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 16:31:59 2025

@author: aiden

Batch Multiple Linear Regression Model (ONNX)

Model equation:
    Y = X @ W.T + b

Where:
    X : input batch of shape (batch_size, n_features)
        - Each row represents one sample with all its features.
    W : weight matrix of shape (n_outputs, n_features)
        - Each row corresponds to the weights for one output.
    b : bias vector of shape (n_outputs,)
        - Added to each output across the batch (broadcasted).
    Y : predicted outputs of shape (batch_size, n_outputs)
        - Each row contains the outputs for one sample.

ONNX Implementation:
    1. Transpose W (W -> WT) to align dimensions for MatMul.
    2. MatMul: X @ WT computes linear combination for each output.
    3. Add: bias b is broadcasted and added to the result.
    4. Output Y contains the predicted values for the batch.
    
    
ONNX Diagram
          ┌────────────────────────────┐
          │      Input X               │
          │ shape: [batch, n_features] │
          └─────────┬──────────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │   Transpose W              │
          │ W (n_outputs×n_feat)       │
          │ WT (n_feat×n_outputs)      │
          └─────────┬──────────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │       MatMul               │
          │   X @ WT                   │
          │ shape: [batch, n_outputs]  │
          └─────────┬──────────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │        Add                 │
          │   (X @ WT) + b             │
          │ shape: [batch, n_outputs]  │
          └─────────┬──────────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │        Output Y            │
          │ shape: [batch, n_outputs]  │
          └────────────────────────────┘

See the .pbtxt file output for the human readable version of the onnx file

"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# ========================
# Step 1: Define parameters
# ========================

# 2 outputs, 3 features
W = np.array([[2.0, -1.0, 0.5],    # output 1
              [0.0, 1.0, 2.0]],    # output 2
             dtype=np.float32)     # shape (2,3)
b = np.array([1.0, -1.0], dtype=np.float32)  # bias for each output

# Initializers
W_init = numpy_helper.from_array(W, name='W')
b_init = numpy_helper.from_array(b, name='b')

# Inputs and outputs
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 3])  # batch_size x n_features
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 2])  # batch_size x n_outputs

# ========================
# Step 2: Create nodes
# ========================

# Transpose W: (2,3) -> (3,2)
transpose_node = helper.make_node(
    'Transpose',
    inputs=['W'],
    outputs=['WT'],
    perm=[1,0]
)

# MatMul: X @ WT
matmul_node = helper.make_node(
    'MatMul',
    inputs=['X', 'WT'],
    outputs=['XW^T']
)

# Add bias
add_node = helper.make_node(
    'Add',
    inputs=['XW^T', 'b'],
    outputs=['Y']
)

# ========================
# Step 3: Create graph
# ========================

graph_def = helper.make_graph(
    nodes=[transpose_node, matmul_node, add_node],
    name='MLR',
    inputs=[X],
    outputs=[Y],
    initializer=[W_init, b_init]
)

model_def = helper.make_model(graph_def, producer_name='onnx-mlr')

onnx_file = 'batch_mlr_safe.onnx'
pbtxt_file = 'batch_mlr_safe.pbtxt'

# Binary ONNX
onnx.save(model_def, onnx_file)

# Text (human-readable) ONNX
with open(pbtxt_file, 'w') as f:
    f.write(str(model_def))  # converts protobuf to text

print(f"Saved binary ONNX to '{onnx_file}'")
print(f"Saved human-readable ONNX to '{pbtxt_file}'")


# ========================
# Step 4: Safe Python runner
# ========================

def simple_ort_run_safe(model_path, inputs):
    """Minimal ONNX runner with safe handling of Transpose, MatMul, and Add."""
    model = onnx.load(model_path)
    graph = model.graph

    # Load constants
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    values = {**initializers, **inputs}

    for node in graph.node:
        ins = [values[i] for i in node.input]

        if node.op_type == 'Transpose':
            # Extract perm safely
            perm = next((list(a.ints) for a in node.attribute if a.name == 'perm'), None)

            # Ensure input is at least 2D
            out = np.transpose(ins[0], axes=perm)

        elif node.op_type == 'MatMul':
            out = ins[0] @ ins[1]

        elif node.op_type == 'Add':
            # broadcasting automatically handles bias addition
            out = ins[0] + ins[1]

        else:
            raise NotImplementedError(f"Unsupported op: {node.op_type}")

        values[node.output[0]] = out

    return [values[out.name] for out in graph.output]


# ========================
# Step 5: Generate multiple test cases
# ========================
n_features = 3
n_outputs = 2

num_tests = 5
batch_size = 4
tolerance = 1e-6

print("\n=== Running test cases with validation ===\n")
npassed = 0
for i in range(num_tests):
    X_input = np.random.uniform(-10, 10, size=(batch_size, n_features)).astype(np.float32)
    
    # ONNX runner output
    Y_output = simple_ort_run_safe(onnx_file, {"X": X_input})[0]

    # Ground truth using NumPy
    Y_true = X_input @ W.T + b

    # Validate
    if np.allclose(Y_output, Y_true, atol=tolerance):
        status = "PASS"
        npassed += 1
    else:
        status = "FAIL"

    print(f"Test case {i+1}: {status}")
    print("X =\n", X_input)
    print("ONNX Y =\n", Y_output)
    print("Expected Y =\n", Y_true)
    print("-"*40)
print()
print(f"{npassed}/{num_tests} passed")

    