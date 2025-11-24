import onnx

model = onnx.load("/home/dhuppert/Documents/ONNX-GPU-Execution-Engine/model.onnx")
ops = sorted({node.op_type for node in model.graph.node})
print(ops)
