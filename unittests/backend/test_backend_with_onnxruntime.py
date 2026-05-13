import unittest

import numpy as np
import onnxruntime as ort

from onnx_light.backend.test.case import make_test_class
from onnx_light.onnx import defs


# Register schemas needed for tests
def _register_test_schemas():
    """Registers schemas needed for backend tests."""
    # Register Abs schema (since_version=13)
    if not defs.has_schema("Abs"):
        abs_schema = defs.OpSchema("Abs", defs.ONNX_DOMAIN, 13, doc="Absolute value")
        defs.register_schema(abs_schema)


# Register schemas before creating test class
_register_test_schemas()


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


TestOrtBackend = make_test_class(onnxruntime_backend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
