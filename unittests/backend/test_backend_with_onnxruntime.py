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


# Backend test cases that ONNXRuntime cannot run as-is:
#   * ``test_cc_roialign_max`` — ORT's RoiAlign max-mode implementation does
#     not match the ONNX reference (ORT emits a warning on session creation).
#   * ``test_cc_flex_attention_*`` — ORT does not register the
#     ``ai.onnx.preview`` domain, so these models fail to load with
#     "ai.onnx.preview:FlexAttention(-1) is not a registered function/op".
#   * ``test_cc_adam_*`` — ORT does not register the
#     ``ai.onnx.preview.training`` domain, so these models fail to load with
#     "ai.onnx.preview.training:Adam(-1) is not a registered function/op".
# These cases remain covered by the reference backend tests.
ORT_EXCLUDE_REGEX = [r"^test_cc_roialign_max$", r"^test_cc_flex_attention_", r"^test_cc_adam_"]

TestOrtBackend = make_test_class(onnxruntime_backend, exclude_regex=ORT_EXCLUDE_REGEX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
