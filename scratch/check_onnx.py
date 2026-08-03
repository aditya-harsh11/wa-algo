import onnx
model = onnx.load("model_float32.onnx")
for input in model.graph.input:
    print(f"Input Name: {input.name}")
    # Get the shape information
    shape = []
    for dim in input.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append("Dynamic")
    print(f"Input Shape: {shape}")
