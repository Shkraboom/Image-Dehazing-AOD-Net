import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    """
    Function for building engine
    """
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MiB

        # Parse model file
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build and return an engine
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)

        return serialized_engine

def load_engine(engine_file_path):
    """
    Function for loading engine
    """
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:

        return runtime.deserialize_cuda_engine(f.read())

def preprocess_image(image_path, input_shape):
    """
    Function for preprocessing image
    """
    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Error: Unable to load image.")

    image = cv2.resize(image, (input_shape[3], input_shape[2]))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # CHW
    image = np.expand_dims(image, axis=0)   # NCHW

    return image

def postprocess_image(output, output_shape, original_shape):
    """
    Function for postprocessing image
    """
    print(f"Output shape before reshape: {output.shape}")
    output = output.reshape(output_shape)

    print(f"Output shape after reshape: {output.shape}")
    output = np.squeeze(output)

    print(f"Output shape after squeeze: {output.shape}")
    output = np.transpose(output, (1, 2, 0))  # HWC
    output = (output * 255).astype(np.uint8)
    output = cv2.resize(output, (original_shape[1], original_shape[0]))

    return output

def do_inference(context, h_input, h_output, d_input, d_output, stream):
    """
    Function for inference
    """
    cuda.memcpy_htod_async(d_input, h_input, stream)

    context.set_input_shape("input", (1, 3, 1024, 1024))
    context.set_tensor_address("input", int(d_input))
    context.set_tensor_address("output", int(d_output))
    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return h_output

def main():
    onnx_file_path = "/path/to/onnx/model.onnx"
    engine_file_path = "/path/to/trt/engine.trt"
    image_path = "/path/to/input/image"
    output_image_path = "/path/to/output/image"

    # Build engine
    build_engine(onnx_file_path, engine_file_path)

    # Load engine
    engine = load_engine(engine_file_path)

    context = engine.create_execution_context()

    input_shape = (1, 3, 1024, 1024)
    output_shape = (1, 3, 1024, 1024)

    # Allocate host and device buffers
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Preprocessing
    input_image = preprocess_image(image_path, input_shape)
    np.copyto(h_input, input_image.ravel())

    # Inference
    output = do_inference(context, h_input, h_output, d_input, d_output, stream)

    original_image = cv2.imread(image_path)
    original_shape = original_image.shape

    # Postprocessing
    print(f"Output shape: {output.shape}")
    output_image = postprocess_image(output, output_shape, original_shape)

    cv2.imwrite(output_image_path, output_image)
    print(f"Output image saved to {output_image_path}")

if __name__ == "__main__":
    main()
