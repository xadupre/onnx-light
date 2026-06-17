import unittest
import numpy as np
import onnxruntime as ort
from onnx_light.ext_test_case import import_or_skip

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
make_test_class = import_or_skip("onnx_light.onnx.backend", "make_test_class")


def onnxruntime_backend(model, *inputs: np.ndarray) -> list[np.ndarray]:
    """
    Runs an ONNX model using ONNXRuntime.

    Args:
        model: The ONNX model (onnx_light.ModelProto) to run
        *inputs: Input arrays for the model

    Returns:
        List of output arrays from the model
    """
    # Serialize the model to bytes
    model_bytes = model.SerializeToString()

    # Create an ONNXRuntime inference session
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

    # Get input names from the session
    input_names = [inp.name for inp in sess.get_inputs()]

    # Create input dictionary
    input_dict = dict(zip(input_names, inputs))

    # Run inference
    outputs = sess.run(None, input_dict)
    return outputs


ORT_EXCLUDE_REGEX = [
    "adagrad",
    "adam",
    "bitcast_bool_to_uint8",
    "bitshift_left_uint16",
    "bitshift_right_uint16",
    "cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN",
    "cast_no_saturate_FLOAT16_to_FLOAT8E5M2",
]

TestOrtBackend = make_test_class(onnxruntime_backend, exclude_regex=ORT_EXCLUDE_REGEX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
