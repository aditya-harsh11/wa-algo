import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path):
    # 1. Initialize the logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 2. Create Builder, Network, and Parser
    builder = trt.Builder(logger)
    # EXPLICIT_BATCH is required for parsing ONNX models in modern TensorRT
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    
    # 3. Read and Parse the ONNX model
    print(f"Parsing ONNX file: {onnx_file_path}...")
    if not os.path.exists(onnx_file_path):
        print(f"ERROR: ONNX file {onnx_file_path} not found.")
        return None

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print("ONNX parsed successfully!")

    # 4. Create a Build Configuration
    config = builder.create_builder_config()
    
    # Enable FP16 precision for faster inference (assuming your GPU supports it)
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Modern TensorRT uses memory pools instead of the old max_workspace_size.
    # Set the workspace size to 2GB (adjust up or down depending on your GPU RAM).
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))
    
    # 5. Build and Serialize the Engine
    print("Building TensorRT engine. This may take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return None

    # 6. Save the Engine to a file
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Engine saved successfully to {engine_file_path}")

def main(size):
    # Updated paths to match our newly exported Plus model
    onnx_path = f"weights/twinlitenetplus_{size}.onnx"
    engine_path = f"weights/twinlitenetplus_{size}.engine"

    # Ensure directory exists (if you change the path to include folders)
    save_dir = os.path.dirname(engine_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    build_engine(onnx_path, engine_path)


if __name__ == "__main__":
    main("large")
    main("medium")
    main("small")
    main("nano")