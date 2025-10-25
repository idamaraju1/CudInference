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

    # ===================================================================
    # 1. Extract all constant initializers from the ONNX graph:
    #      - These are model parameters such as weights (W) and biases (b).
    #      - Convert them from ONNX protobuf format into NumPy arrays.
    #      - Store them in a dictionary keyed by their tensor names.
    # 
    # 2. Merge the initializers dictionary with user-provided inputs:
    #      - Inputs are also provided as a dictionary keyed by tensor names.
    #      - The resulting 'values' dictionary contains all tensors needed
    #        for execution: constants + inputs.
    #
    # 3. Purpose of 'values':
    #      - Acts as the central storage of all tensor values during execution.
    #      - Each node in the graph retrieves its inputs from this dictionary,
    #        computes its outputs, and stores them back in 'values'.
    #      - Enables sequential execution without explicit graph traversal.
    # ===================================================================
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    values = {**initializers, **inputs}


    # ===================================================================
    # Graph Traversal and Node Execution:
    #
    # 1. Sequential Node Processing:
    #      - The graph is executed by iterating over `graph.node` in order.
    #      - Each node represents a single operation (e.g., MatMul, Add, Transpose).
    #
    # 2. Input Retrieval:
    #      - For each node, the inputs are looked up in the 'values' dictionary
    #        using the node's input tensor names.
    #      - This ensures each node has access to both constants (initializers)
    #        and outputs from previously executed nodes.
    #
    # 3. Operation Execution:
    #      - The node performs its operation using the retrieved input arrays.
    #      - Supported operations are implemented in Python (e.g., np.matmul, np.add).
    #
    # 4. Output Storage:
    #      - The computed output(s) of the node are stored back into the 'values'
    #        dictionary under the node's output tensor names.
    #      - This makes them available for any subsequent nodes that depend on them.
    #
    # 5. Purpose of this Traversal:
    #      - By storing all tensor values in 'values' and processing nodes sequentially,
    #        we effectively flatten the graph execution.
    #      - No explicit graph traversal or dependency analysis is required.
    #      - Each node simply reads what it needs and writes its result, maintaining
    #        correct execution order automatically.
    # ===================================================================
    for node in graph.node:
        ins = [values[i] for i in node.input]

        if node.op_type == 'Transpose':
            # ===================================================================
            # Perform the Transpose operation:
            # 1. 'ins[0]' is the input tensor for this node (NumPy array).
            # 2. 'perm' specifies the desired order of axes. (Optional attribute)
            #      - Example 1 (2D tensor):
            #          Input shape: (2, 3)
            #          perm = [1, 0] -> output shape: (3, 2)
            #      - Example 2 (3D tensor):
            #          Input shape: (batch, height, width) = (4, 5, 6)
            #          perm = [0, 2, 1] -> output shape: (4, 6, 5)
            #            Explanation:
            #               axis 0 (batch) stays in place,
            #               axis 1 (height) moves to position 2,
            #               axis 2 (width) moves to position 1
            #      - If perm is None, np.transpose defaults to reversing all axes.
            # 3. np.transpose(arr, axes=perm) rearranges the axes according to 'perm',
            #    producing the transposed output tensor.
            # 4. The result is stored in 'values[node.output[0]]' for subsequent nodes.
            # ===================================================================
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

    