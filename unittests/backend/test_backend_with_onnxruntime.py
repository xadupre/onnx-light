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
#   * ``test_cc_bernoulli``, ``test_cc_bernoulli_double`` and
#     ``test_cc_bernoulli_seed`` — ORT decomposes ``Bernoulli`` into
#     ``RandomUniformLike`` + ``Less``, but its CPU EP has no kernel
#     registered for ``RandomUniformLike(22)`` ("Could not find an
#     implementation for RandomUniformLike(22) node"). The reference backend
#     still exercises these cases.
#   * ``test_cc_binarizer_int64`` — ORT only registers a ``float`` kernel for
#     ``ai.onnx.ml::Binarizer``, so the ``int64`` variant fails with
#     "Could not find an implementation for Binarizer(1) node". The
#     ``float`` variant (``test_cc_binarizer_float``) is still exercised.
#   * ``test_cc_scaler_int64`` — ORT only registers a ``float`` kernel for
#     ``ai.onnx.ml::Scaler``, so the ``int64`` variant fails with
#     "Could not find an implementation for Scaler(1) node". The
#     ``float`` variant (``test_cc_scaler_float``) is still exercised.
#   * ``test_cc_cast_*FLOAT8E4M3*`` and ``test_cc_cast_*FLOAT8E5M2*`` — ORT's
#     CPU EP rejects the float8 dtypes as ``Cast`` operand types
#     ("Type 'tensor(float8eNmM)' of input parameter ... is invalid") and its
#     Python bindings have no numpy mapping for the output type
#     ("No corresponding Numpy type for Tensor Type. Float8E5M2"). The
#     reference backend still exercises these conversions byte-for-byte.
#   * ``test_cc_cast_*UINT4*`` and ``test_cc_cast_*UINT2*`` — ORT's CPU EP
#     rejects ``uint4``/``uint2`` as ``Cast`` operand types
#     ("Type 'tensor(uint4)' of input parameter ... is invalid"). The
#     reference backend still exercises these conversions byte-for-byte.
#   * ``test_cc_cast_*INT4*`` and ``test_cc_cast_*INT2*`` — ORT's CPU EP
#     rejects ``int4``/``int2`` as ``Cast`` operand types
#     ("Type 'tensor(int4)' of input parameter ... is invalid") and its
#     Python bindings have no numpy mapping for the packed output types
#     ("No corresponding Numpy type for Tensor Type. Int4x2"). The
#     reference backend still exercises these conversions byte-for-byte.
#   * ``test_cc_zipmap_*`` — ORT returns Python ``list[dict]`` for ZipMap
#     outputs while the lightweight backend-test carrier materializes ZipMap
#     expected outputs as float tensors containing map values.
#   * ``test_dequantizelinear_int16`` and ``test_dequantizelinear_uint16`` —
#     ORT's CPU EP rejects ``int16``/``uint16`` as ``DequantizeLinear`` input
#     types ("Type 'tensor(int16)' of input parameter ... is invalid"). The
#     reference backend still exercises these cases.
#   * ``test_dequantizelinear_e4m3fn``, ``test_dequantizelinear_e5m2`` and
#     ``test_dequantizelinear_e4m3fn_zero_point`` — ORT's CPU EP rejects
#     ``float8e4m3fn`` / ``float8e5m2`` as ``DequantizeLinear`` input types
#     ("Type 'tensor(float8e4m3fn)' of input parameter (x) of operator
#     (DequantizeLinear) ... is invalid"). The reference backend still
#     exercises these cases.
#   * ``test_quantizelinear_int16`` and ``test_quantizelinear_uint16`` — ORT's
#     CPU EP rejects ``int16``/``uint16`` as ``QuantizeLinear`` ``y_zero_point``
#     types ("Type 'tensor(int16)' of input parameter (y_zero_point) ... is
#     invalid"). The reference backend still exercises these cases.
#   * ``test_cc_svmclassifier_int64_binary`` and
#     ``test_cc_svmregressor_linear`` — ORT's ``ai.onnx.ml`` SVM kernels follow
#     a different scoring/layout convention than the lightweight backend
#     reference kernels for these focused C++ parity cases.
#   * ``test_cc_linearclassifier_int64_binary`` and
#     ``test_cc_linearregressor_single_target`` — focused C++ parity cases for
#     the ``ai.onnx.ml`` Linear* operators; ORT's kernels may apply a
#     different score-expansion convention for binary classifiers.
#   * ``test_cc_treeensembleclassifier_int64_binary`` — focused C++ parity
#     case for ``ai.onnx.ml::TreeEnsembleClassifier``; ORT's binary-classifier
#     kernel applies a different score-expansion convention (threshold-at-zero
#     rather than argmax) that produces a different predicted label for the
#     zero-score sample.
#   * ``test_cc_globallppool_*`` — ORT has no CPU kernel for
#     ``GlobalLpPool(22)`` ("Could not find an implementation for
#     GlobalLpPool(22) node"). The reference backend still exercises these
#     cases.
#   * ``test_cc_dict_vectorizer_*`` — these models declare an
#     ``ai.onnx.ml::DictVectorizer`` input typed as ``map(K, V)``. ORT loads
#     the model with a map-typed input parameter and rejects the tensor
#     placeholder fed by the lightweight backend harness
#     ("input with name: 'x' expected to be of type: 1 but received a
#     tensor"). The reference backend still exercises these cases.
#   * ``test_cc_feature_vectorizer_mixed_dtypes`` — ORT's
#     ``ai.onnx.ml::FeatureVectorizer`` kernel binds the variadic ``T1``
#     type-constraint to a single dtype across all inputs and rejects mixed
#     dtypes at load time ("Type parameter (T1) of Optype (FeatureVectorizer)
#     bound to different types (tensor(int64) and tensor(float))"). The ONNX
#     reference backend still exercises this case.
#   * ``test_cc_simple_rnn_batchwise`` — ORT's CPU ``RNN`` kernel rejects
#     ``layout=1`` at initialization ("Batchwise recurrent operations
#     (layout == 1) are not supported. If you need support create a github
#     issue with justification."). The reference backend still exercises
#     this case.
# These cases remain covered by the reference backend tests.
ORT_EXCLUDE_REGEX = [
    r"^test_cc_roialign_max$",
    r"^test_cc_roialign_mode_max$",
    r"^test_cc_flex_attention_",
    r"^test_cc_adam_",
    r"^test_adam$",
    r"^test_adam_multiple$",
    r"^test_cc_bernoulli$",
    r"^test_cc_bernoulli_double$",
    r"^test_cc_bernoulli_seed$",
    r"^test_cc_binarizer_int64$",
    r"^test_cc_scaler_int64$",
    r"^test_cc_cast_.*FLOAT8E4M3.*$",
    r"^test_cc_cast_.*FLOAT8E5M2.*$",
    r"^test_cc_cast_.*UINT4.*$",
    r"^test_cc_cast_.*UINT2.*$",
    r"^test_cc_cast_.*INT4.*$",
    r"^test_cc_cast_.*INT2.*$",
    r"^test_cc_zipmap_",
    r"^test_dequantizelinear_int16$",
    r"^test_dequantizelinear_uint16$",
    r"^test_dequantizelinear_e4m3fn$",
    r"^test_dequantizelinear_e4m3fn_zero_point$",
    r"^test_dequantizelinear_e5m2$",
    r"^test_quantizelinear_int16$",
    r"^test_quantizelinear_uint16$",
    r"^test_cc_svmclassifier_int64_binary$",
    r"^test_cc_svmregressor_linear$",
    r"^test_cc_linearclassifier_int64_binary$",
    r"^test_cc_linearregressor_single_target$",
    r"^test_cc_treeensembleclassifier_int64_binary$",
    r"^test_cc_globallppool_",
    r"^test_cc_dict_vectorizer_",
    r"^test_cc_feature_vectorizer_mixed_dtypes$",
    r"^test_cc_simple_rnn_batchwise$",
]

TestOrtBackend = make_test_class(onnxruntime_backend, exclude_regex=ORT_EXCLUDE_REGEX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
