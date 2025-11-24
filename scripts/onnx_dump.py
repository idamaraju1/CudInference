import onnx

model = onnx.load("/home/dhuppert/Documents/ONNX-GPU-Execution-Engine/SmolLM2-135M.onnx")
ops = sorted({node.op_type for node in model.graph.node})
print(ops)
