import unittest
import numpy as np
import onnxruntime as ort
from onnx_light.backend.test.case import make_test_class


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


# Exclude test cases that exercise operators in the ``ai.onnx.preview``
# domain — ONNXRuntime does not register that domain, so these models fail
# to load with errors such as "ai.onnx.preview:FlexAttention(-1) is not a
# registered function/op". They are covered by the reference backend tests.
_ORT_EXCLUDE_REGEX = [
    r"^test_cc_flex_attention_",
]

TestOrtBackend = make_test_class(onnxruntime_backend, exclude_regex=_ORT_EXCLUDE_REGEX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
