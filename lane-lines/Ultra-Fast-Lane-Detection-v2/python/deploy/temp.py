import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Handles initialization and cleanup
import numpy as np

# Logger to capture TensorRT warnings and errors
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    # Load the TensorRT engine
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def test_engine(engine):
    # TensorRT 8.x API doesn't have 'num_bindings'; uses tensor names and other methods
    num_tensors = engine.num_io_tensors  # Number of input/output tensors

    print(f"Number of tensors: {num_tensors}")
    
    # Iterate over all tensors (input/output)
    for i in range(num_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        
        print(f"Tensor {i}:")
        print(f"  Name: {tensor_name}")
        print(f"  Shape: {tensor_shape}")
        print(f"  Dtype: {tensor_dtype}")
    
    # No need for explicit synchronization
    # cuda.Context.synchronize()  # Commented out to avoid issues

def run_inference(engine, input_shape):
    # Create a context for the engine
    context = engine.create_execution_context()

    # Allocate memory for inputs and outputs
    input_data = np.random.random(input_shape).astype(np.float32)  # Dummy input data
    input_mem = cuda.mem_alloc(input_data.nbytes)

    # Get output tensor sizes based on their shapes from the engine
    loc_row_shape = (1, 200, 72, 4)  # Based on the tensor shape from test
    loc_col_shape = (1, 100, 81, 4)  # Based on the tensor shape from test
    exist_row_shape = (1, 2, 72, 4)  # Based on the tensor shape from test
    exist_col_shape = (1, 2, 81, 4)  # Based on the tensor shape from test
    
    # Allocate memory for output tensors (explicitly casting to int)
    loc_row_mem = cuda.mem_alloc(int(np.prod(loc_row_shape) * np.float32().nbytes))
    loc_col_mem = cuda.mem_alloc(int(np.prod(loc_col_shape) * np.float32().nbytes))
    exist_row_mem = cuda.mem_alloc(int(np.prod(exist_row_shape) * np.float32().nbytes))
    exist_col_mem = cuda.mem_alloc(int(np.prod(exist_col_shape) * np.float32().nbytes))
    
    # Transfer input data to GPU
    cuda.memcpy_htod(input_mem, input_data)

    # Execute inference with the context
    context.execute_v2([int(input_mem), int(loc_row_mem), int(loc_col_mem), int(exist_row_mem), int(exist_col_mem)])
    
    # Synchronize to ensure all GPU operations are complete
    cuda.Context.synchronize()

    # Transfer output from GPU to CPU for verification
    loc_row_output = np.empty(loc_row_shape, dtype=np.float32)
    loc_col_output = np.empty(loc_col_shape, dtype=np.float32)
    exist_row_output = np.empty(exist_row_shape, dtype=np.float32)
    exist_col_output = np.empty(exist_col_shape, dtype=np.float32)

    # Copy the data back from GPU to CPU
    cuda.memcpy_dtoh(loc_row_output, loc_row_mem)
    cuda.memcpy_dtoh(loc_col_output, loc_col_mem)
    cuda.memcpy_dtoh(exist_row_output, exist_row_mem)
    cuda.memcpy_dtoh(exist_col_output, exist_col_mem)
    
    # Print the output data
    print("loc_row output:", loc_row_output)
    print("loc_col output:", loc_col_output)
    print("exist_row output:", exist_row_output)
    print("exist_col output:", exist_col_output)

if __name__ == "__main__":
    engine_path = 'weights/ufldv2_culane_res18.engine'
    
    # Load the engine
    engine = load_engine(engine_path)
    
    # Test engine information and bindings
    test_engine(engine)

    # Define input shape (based on your input tensor information)
    input_shape = (1, 3, 320, 1600)  # Adjust this shape if needed

    # Run inference to check if the engine is working correctly
    run_inference(engine, input_shape)

    # Free any manually allocated resources, if needed
    # pycuda.autoinit automatically handles cleanup
