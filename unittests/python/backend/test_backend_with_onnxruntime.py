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
#   * ``test_cc_adam_*``, ``test_adam``, ``test_adam_multiple``,
#     ``test_adagrad``, ``test_adagrad_multiple``, ``test_momentum``,
#     ``test_momentum_multiple`` and ``test_nesterov_momentum`` — ORT does
#     not register the ``ai.onnx.preview.training`` domain, so these models
#     fail to load with
#     "ai.onnx.preview.training:Adam(-1) is not a registered function/op".
#   * ``test_cc_bernoulli``, ``test_cc_bernoulli_double`` and
#     ``test_cc_bernoulli_seed`` — ORT decomposes ``Bernoulli`` into
#     ``RandomUniformLike`` + ``Less``, but its CPU EP has no kernel
#     registered for ``RandomUniformLike(22)`` ("Could not find an
#     implementation for RandomUniformLike(22) node"). The reference backend
#     still exercises these cases.
#   * ``test_cc_multinomial``, ``test_cc_multinomial_seeded`` and
#     ``test_cc_multinomial_int64`` — ORT's CPU EP has no kernel registered
#     for ``Multinomial(22)`` ("Could not find an implementation for
#     Multinomial(22) node"). The reference backend still exercises these
#     cases.
#   * ``test_cc_randomnormal*``, ``test_cc_randomnormallike*``,
#     ``test_cc_randomuniform*`` and ``test_cc_randomuniformlike*`` — ORT's
#     CPU EP has no kernel registered for ``RandomNormal(22)``,
#     ``RandomNormalLike(22)``, ``RandomUniform(22)`` or
#     ``RandomUniformLike(22)`` ("Could not find an implementation for
#     RandomNormal(22) node"). The reference backend still exercises these
#     cases.
#   * ``test_training_dropout``, ``test_training_dropout_mask``,
#     ``test_training_dropout_default`` and
#     ``test_training_dropout_default_mask`` — training-mode ``Dropout`` with
#     ``ratio > 0`` selects kept/dropped elements via a runtime-defined RNG,
#     so the expected outputs computed by ``kernel::Dropout`` are not
#     bit-comparable with ORT's CPU kernel even for the same seed. The
#     zero-ratio variants remain deterministic (output equals input, mask is
#     all ones) and are still exercised.
#   * ``test_cc_binarizer_int64`` — ORT only registers a ``float`` kernel for
#     ``ai.onnx.ml::Binarizer``, so the ``int64`` variant fails with
#     "Could not find an implementation for Binarizer(1) node". The
#     ``float`` variant (``test_ai_onnx_ml_binarizer``) is still exercised.
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
#   * ``test_dequantizelinear_e4m3fn``, ``test_dequantizelinear_e5m2``,
#     ``test_dequantizelinear_e4m3fn_zero_point`` and
#     ``test_dequantizelinear_e4m3fn_float16`` — ORT's CPU EP rejects
#     ``float8e4m3fn`` / ``float8e5m2`` as ``DequantizeLinear`` input types
#     ("Type 'tensor(float8e4m3fn)' of input parameter (x) of operator
#     (DequantizeLinear) ... is invalid"). The reference backend still
#     exercises these cases.
#   * ``test_dequantizelinear_uint4``, ``test_dequantizelinear_int4``,
#     ``test_dequantizelinear_uint2``, ``test_dequantizelinear_int2`` and
#     ``test_dequantizelinear_float4e2m1`` — ORT's CPU EP rejects sub-byte
#     dtypes as ``DequantizeLinear`` input types ("Type 'tensor(uint4)' of
#     input parameter (x) of operator (DequantizeLinear) ... is invalid").
#     The reference backend still exercises these cases.
#   * ``test_dequantizelinear_blocked`` — ORT's CPU EP rejects the
#     ``block_size`` attribute on ``DequantizeLinear`` ("This is an invalid
#     model. In Node ... DequantizeLinear ..."). The reference backend still
#     exercises this case.
#   * ``test_quantizelinear_int16`` and ``test_quantizelinear_uint16`` — ORT's
#     CPU EP rejects ``int16``/``uint16`` as ``QuantizeLinear`` ``y_zero_point``
#     types ("Type 'tensor(int16)' of input parameter (y_zero_point) ... is
#     invalid"). The reference backend still exercises these cases.
#   * ``test_quantizelinear_e4m3fn``, ``test_quantizelinear_e5m2``,
#     ``test_quantizelinear_uint4``, ``test_quantizelinear_int4``,
#     ``test_quantizelinear_uint2``, ``test_quantizelinear_int2`` and
#     ``test_quantizelinear_float4e2m1`` — ORT's CPU EP rejects these
#     sub-byte / float8 / float4 types as ``QuantizeLinear`` ``y_zero_point``
#     types ("Type 'tensor(<dtype>)' of input parameter (y_zero_point) ... is
#     invalid"). The reference backend still exercises these cases.
#   * ``test_cc_qlinearmatmul_{2D,3D}_{uint8,int8}_float16`` — ORT's CPU EP
#     rejects FLOAT16 scales for ``QLinearMatMul`` ("Type 'tensor(float16)'
#     of input parameter (a_scale) of operator (QLinearMatMul) ... is
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
#   * ``test_cc_maxroipool_*`` — ORT has no CPU kernel for
#     ``MaxRoiPool(22)`` ("Could not find an implementation for
#     MaxRoiPool(22) node"). The reference backend still exercises these
#     cases.
#   * ``test_cc_scan_zero_trip_count`` — ORT's ``Scan`` implementation
#     unconditionally slices the scan input at ``dim0_offset=0`` even when
#     the scan-input leading dimension is 0, asserting
#     ``dim0_offset < dim0_size`` ("Invalid dim0_offset of 0. Dimension 0
#     is 0"). The reference backend still exercises the zero-trip-count
#     case.
#   * ``test_bitshift_right_uint16`` and ``test_bitshift_left_uint16`` — ORT's
#     CPU EP does not register a ``BitShift`` kernel for ``uint16`` ("Could
#     not find an implementation for BitShift(11) node"); only ``uint8`` /
#     ``uint32`` / ``uint64`` are registered. The reference backend still
#     exercises the ``uint16`` cases.
#   * ``test_bitcast_*`` — ORT's CPU EP does not register a ``BitCast`` kernel
#     ("Could not find an implementation for BitCast(26) node"). The reference
#     backend still exercises these cases.
#   * ``test_cc_dict_vectorizer_*`` and ``test_cc_cast_map_*`` — these models
#     declare an ``ai.onnx.ml::DictVectorizer`` / ``ai.onnx.ml::CastMap``
#     input typed as ``map(K, V)``. ORT loads the model with a map-typed
#     input parameter and rejects the tensor placeholder fed by the
#     lightweight backend harness ("input with name: 'x' expected to be of
# type: 1 but received a tensor"). The reference backend still exercises
#     these cases.
#   * ``test_cc_feature_vectorizer_mixed_dtypes`` — ORT's
#     ``ai.onnx.ml::FeatureVectorizer`` kernel binds the variadic ``T1``
#     type-constraint to a single dtype across all inputs and rejects mixed
#     dtypes at load time ("Type parameter (T1) of Optype (FeatureVectorizer)
#     bound to different types (tensor(int64) and tensor(float))"). The ONNX
#     reference backend still exercises this case.
#   * ``test_cc_simple_rnn_batchwise``, ``test_cc_lstm_batchwise`` and
#     ``test_cc_gru_batchwise`` — ORT's CPU ``RNN``/``LSTM``/``GRU``
#     kernels reject ``layout=1`` at initialization ("Batchwise recurrent
#     operations (layout == 1) are not supported. If you need support
#     create a github issue with justification."). The reference backend
#     still exercises these cases.
#   * ``test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal``
#     and ``test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal``
#     — ORT's CPU ``Attention`` kernel writes ``-FLT_MAX`` (≈ -3.4e38) into the
#     ``qk_matmul_output`` at causal-masked positions while the ONNX reference
#     (and this lightweight backend) writes the mathematical ``-inf`` from the
#     additive causal bias. Both produce the same softmax outputs but differ
#     bit-for-bit in the intermediate ``qk_matmul_output``. The reference
#     backend still exercises these cases.
# These cases remain covered by the reference backend tests.
#   * ``test_cc_top_k_uint64`` — ORT's CPU EP has no ``TopK(11)`` kernel
#     registered for ``uint64`` ("Could not find an implementation for
#     TopK(11) node"). The reference backend still exercises this case.
#   * ``test_cc_prelu_inf`` — ORT's CPU EP ``PRelu`` kernel evaluates
#     ``slope * x`` on both branches of the sign mask and so returns
#     ``NaN`` for ``+inf`` / ``-inf`` inputs (see
#     microsoft/onnxruntime#28732). The reference backend still exercises
#     this case to lock in the ``±inf``-preserving behaviour.
#   * ``test_pow_types_float32_uint32`` and ``test_pow_types_float32_uint64``
#     — ORT has no CPU kernel for ``Pow(13)`` with ``uint32``/``uint64``
#     exponent inputs ("Could not find an implementation for Pow(13) node").
#     The reference backend still exercises these mixed-type coverage cases.
#   * ``test_cc_shape_inference_shape_identity_unsqueeze`` — model
#     intentionally exercises the in-memory INT64 initializer path in
#     ``Graph::SaveShapeValuesFromDataPropagation`` (the regression scenario
#     fixed by microsoft/onnxruntime#28778). ORT versions on the macOS /
#     Windows CI runners predate that fix and abort on the model during
#     graph re-resolution. The reference backend still exercises this case.
#   * ``test_max_int16``, ``test_max_uint16``, ``test_min_int16`` and
#     ``test_min_uint16`` — ORT has no CPU kernel for ``Max(13)`` /
#     ``Min(13)`` with ``int16``/``uint16`` inputs ("Could not find an
#     implementation for Max(13) node" / "Could not find an implementation
#     for Min(13) node"). The reference backend still exercises these cases.
#   * ``test_resize_downsample_scales_linear_align_corners`` and
#     ``test_resize_downsample_scales_cubic_align_corners`` — when
#     downsampling with ``coordinate_transformation_mode="align_corners"``
#     and float ``scales``, ORT's CPU EP computes the output spatial size
#     by rounding ``input_size * scale`` (e.g. ``round(4 * 0.6) = 2`` and
#     ``round(4 * 0.8) = 3``) and then maps coordinates with
#     ``(output_size - 1) / (input_size - 1)``, which lands sampling
#     positions exactly on existing grid points and reproduces the input
#     values verbatim. The ONNX reference implementation in
#     ``onnx/reference/ops/op_resize.py`` instead computes ``align_corners``
#     downsample positions using the float scale directly
#     (``i * (input_size - 1) * scale / (output_size - 1)``), which yields
#     the fractional sample positions baked into the upstream
#     ``test_resize_downsample_scales_*_align_corners`` reference outputs.
#     The reference backend still exercises these cases.
#   * ``test_cc_maxunpool_export_with_output_shape`` — when ``output_shape``
#     differs from the shape inferred from ``kernel_shape``/``strides``/
#     ``pads``, ORT's CPU EP scatters ``X`` into ``output_shape`` directly by
#     reinterpreting ``indices`` as flat offsets into ``output_shape``. The
#     ONNX reference implementation instead scatters into the inferred shape
#     and copies that region into the top-left corner of ``output_shape``
#     (see ``onnx/reference/ops/op_max_unpool.py``). The reference backend
#     still exercises this case.
ORT_EXCLUDE_REGEX = [
    r"^test_cc_roialign_max$",
    r"^test_cc_roialign_mode_max$",
    r"^test_cc_flex_attention_",
    r"^test_cc_image_decoder_",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal$",
    r"^test_cc_adam_",
    r"^test_adam$",
    r"^test_adam_multiple$",
    r"^test_adagrad$",
    r"^test_adagrad_multiple$",
    r"^test_momentum$",
    r"^test_momentum_multiple$",
    r"^test_nesterov_momentum$",
    r"^test_cc_bernoulli$",
    r"^test_cc_bernoulli_double$",
    r"^test_cc_bernoulli_seed$",
    r"^test_cc_multinomial$",
    r"^test_cc_multinomial_seeded$",
    r"^test_cc_multinomial_int64$",
    r"^test_cc_randomnormal$",
    r"^test_cc_randomnormal_double$",
    r"^test_cc_randomnormal_seeded$",
    r"^test_cc_randomnormallike$",
    r"^test_cc_randomnormallike_double$",
    r"^test_cc_randomnormallike_seeded$",
    r"^test_cc_randomuniform$",
    r"^test_cc_randomuniform_double$",
    r"^test_cc_randomuniform_seeded$",
    r"^test_cc_randomuniformlike$",
    r"^test_cc_randomuniformlike_double$",
    r"^test_cc_randomuniformlike_seeded$",
    r"^test_training_dropout$",
    r"^test_training_dropout_mask$",
    r"^test_training_dropout_default$",
    r"^test_training_dropout_default_mask$",
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
    r"^test_dequantizelinear_e4m3fn_float16$",
    r"^test_dequantizelinear_e5m2$",
    r"^test_dequantizelinear_uint4$",
    r"^test_dequantizelinear_int4$",
    r"^test_dequantizelinear_uint2$",
    r"^test_dequantizelinear_int2$",
    r"^test_dequantizelinear_float4e2m1$",
    r"^test_dequantizelinear_blocked$",
    r"^test_quantizelinear_int16$",
    r"^test_quantizelinear_uint16$",
    r"^test_quantizelinear_e4m3fn$",
    r"^test_quantizelinear_e5m2$",
    r"^test_quantizelinear_uint4$",
    r"^test_quantizelinear_int4$",
    r"^test_quantizelinear_uint2$",
    r"^test_quantizelinear_int2$",
    r"^test_quantizelinear_float4e2m1$",
    r"^test_cc_qlinearmatmul_2D_uint8_float16$",
    r"^test_cc_qlinearmatmul_2D_int8_float16$",
    r"^test_cc_qlinearmatmul_3D_uint8_float16$",
    r"^test_cc_qlinearmatmul_3D_int8_float16$",
    r"^test_cc_svmclassifier_int64_binary$",
    r"^test_cc_svmregressor_linear$",
    r"^test_cc_linearclassifier_int64_binary$",
    r"^test_cc_linearregressor_single_target$",
    r"^test_cc_treeensembleclassifier_int64_binary$",
    r"^test_cc_globallppool_",
    r"^test_cc_maxroipool_",
    r"^test_cc_dict_vectorizer_",
    r"^test_cc_cast_map_",
    r"^test_cc_feature_vectorizer_mixed_dtypes$",
    r"^test_cc_simple_rnn_batchwise$",
    r"^test_cc_lstm_batchwise$",
    r"^test_cc_gru_batchwise$",
    r"^test_bitshift_right_uint16$",
    r"^test_bitshift_left_uint16$",
    r"^test_bitcast_",
    r"^test_cc_top_k_uint64$",
    r"^test_cc_scan_zero_trip_count$",
    r"^test_cc_prelu_inf$",
    r"^test_pow_types_float32_uint32$",
    r"^test_pow_types_float32_uint64$",
    r"^test_cc_shape_inference_shape_identity_unsqueeze$",
    r"^test_max_int16$",
    r"^test_max_uint16$",
    r"^test_min_int16$",
    r"^test_min_uint16$",
    r"^test_cc_maxunpool_export_with_output_shape$",
    r"^test_resize_downsample_scales_linear_align_corners$",
    r"^test_resize_downsample_scales_cubic_align_corners$",
    # Range opset 27 cases: ONNX Runtime only guarantees support up to opset 26.
    r"^test_range_float16_type_positive_delta$",
    r"^test_range_bfloat16_type_positive_delta$",
    # LinearAttention is opset 27: ONNX Runtime only guarantees support up to opset 26.
    r"^test_cc_linear_attention_.*$",
    # CausalConvWithState is opset 27: ONNX Runtime only guarantees support up to opset 26.
    r"^test_cc_causal_conv_with_state_.*$",
]

TestOrtBackend = make_test_class(onnxruntime_backend, exclude_regex=ORT_EXCLUDE_REGEX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
