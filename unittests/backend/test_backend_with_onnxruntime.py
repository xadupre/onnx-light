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
#   * ``test_cc_roialign_max`` and ``test_cc_roialign_mode_max`` — ORT's
#     RoiAlign max-mode implementation does not match the ONNX reference
#     (ORT emits a warning on session creation).
#   * ``test_cc_flex_attention_*`` — ORT does not register the
#     ``ai.onnx.preview`` domain, so these models fail to load with
#     "ai.onnx.preview:FlexAttention(-1) is not a registered function/op".
#   * ``test_cc_adam_*``, ``test_adam`` and ``test_adam_multiple`` — ORT does
#     not register the ``ai.onnx.preview.training`` domain, so these models
#     fail to load with
#     "ai.onnx.preview.training:Adam(-1) is not a registered function/op".
#   * ``test_cc_binarizer_int64`` — ORT only registers a ``float`` kernel for
#     ``ai.onnx.ml::Binarizer``, so the ``int64`` variant fails with
#     "Could not find an implementation for Binarizer(1) node". The
#     ``float`` variant (``test_cc_binarizer_float``) is still exercised.
#   * ``test_cc_cast_*FLOAT8E4M3*`` and ``test_cc_cast_*FLOAT8E5M2*`` — ORT's
#     CPU EP rejects the float8 dtypes as ``Cast`` operand types
#     ("Type 'tensor(float8eNmM)' of input parameter ... is invalid") and its
#     Python bindings have no numpy mapping for the output type
#     ("No corresponding Numpy type for Tensor Type. Float8E5M2"). The
#     reference backend still exercises these conversions byte-for-byte.
# These cases remain covered by the reference backend tests.
ORT_EXCLUDE_REGEX = [
    r"^test_cc_roialign_max$",
    r"^test_cc_roialign_mode_max$",
    r"^test_cc_flex_attention_",
    r"^test_cc_adam_",
    r"^test_adam$",
    r"^test_adam_multiple$",
    r"^test_cc_binarizer_int64$",
    r"^test_cc_cast_.*FLOAT8E4M3.*$",
    r"^test_cc_cast_.*FLOAT8E5M2.*$",
]

TestOrtBackend = make_test_class(onnxruntime_backend, exclude_regex=ORT_EXCLUDE_REGEX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
