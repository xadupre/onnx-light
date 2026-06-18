import unittest
import numpy as np
import onnxruntime as ort
from onnxruntime.capi._pybind_state import get_all_operator_schema
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


def ort_max_supported_opset() -> int:
    """
    Returns the highest default-domain opset version ONNX Runtime supports.

    Reads the registered operator schemas from ONNX Runtime and takes the
    maximum ``since_version`` over the default ONNX domain (``""``). This lets
    the exclusion list adapt to the installed ONNX Runtime instead of
    hard-coding an opset ceiling.

    Returns:
        The highest default-domain opset version ONNX Runtime supports.
    """
    return max(
        schema.since_version for schema in get_all_operator_schema() if schema.domain == ""
    )


# Opset version at which the cases below were introduced. They are only excluded
# when the installed ONNX Runtime does not yet support that opset.
OPSET_27 = 27

# Exclusions that only apply when ONNX Runtime does not support the given opset.
ORT_OPSET_GATED_EXCLUDE_REGEX = {
    OPSET_27: [
        # Range opset 27 cases.
        r"^test_range_float16_type_positive_delta$",
        r"^test_range_bfloat16_type_positive_delta$",
        # LinearAttention is opset 27.
        r"^test_cc_linear_attention_.*$",
        # CausalConvWithState is opset 27.
        r"^test_cc_causal_conv_with_state_.*$",
    ]
}

ORT_EXCLUDE_REGEX = [
    # ORT/reference parity mismatches in focused C++ cases.
    r"^test_cc_stft_complex_batched$",
    r"^test_cc_image_decoder_",
    # Preview ops/functions are not registered in ORT.
    r"^test_cc_flexattention_",
    # ORT exposes different Attention intermediates than the ONNX reference.
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal(_expanded)?$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal(_expanded)?$",
    r"^test_cc_attention_3d_with_past_and_present_qk_matmul_bias(_expanded)?$",
    r"^test_cc_attention_4d_with_qk_matmul_bias(_expanded)?$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias(_expanded)?$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask(_expanded)?$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask(_expanded)?$",
    r"^test_cc_attention_4d_softcap_neginf_mask(_expanded)?$",
    r"^test_cc_attention_4d_softcap_neginf_mask_poison(_expanded)?$",
    # Preview training ops are not registered in ORT.
    r"^test_cc_adam_",
    r"^test_adam$",
    r"^test_adam_multiple$",
    r"^test_adagrad$",
    r"^test_adagrad_multiple$",
    r"^test_momentum$",
    r"^test_momentum_multiple$",
    r"^test_nesterov_momentum$",
    # Random ops are missing or nondeterministic in ORT.
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
    # ORT only wires float kernels for these ai.onnx.ml cases.
    r"^test_cc_binarizer_int64$",
    r"^test_cc_scaler_int64$",
    # ORT's binary LinearClassifier Z output uses [1-z, z] instead of the spec's [-z, z].
    r"^test_cc_linearclassifier_int64_binary$",
    # ORT returns wrong labels for the binary TreeEnsembleClassifier test case.
    r"^test_cc_treeensembleclassifier_int64_binary$",
    # Low-precision Cast/CastLike dtypes are unsupported in ORT.
    r"^test_cc_cast_.*FLOAT8E4M3.*$",
    r"^test_cc_cast_.*FLOAT8E5M2.*$",
    r"^test_cc_cast_.*FLOAT8E8M0.*$",
    r"^test_cc_cast_.*FLOAT4E2M1.*$",
    r"^test_cast_no_saturate_.*FLOAT8.*$",
    r"^test_cc_castlike_.*FLOAT8E4M3.*$",
    r"^test_cc_castlike_.*FLOAT8E5M2.*$",
    r"^test_cc_castlike_.*FLOAT8E8M0.*$",
    r"^test_cc_castlike_.*FLOAT4E2M1.*$",
    r"^test_castlike_no_saturate_.*FLOAT8.*$",
    r"^test_cc_cast_.*UINT4.*$",
    r"^test_cc_cast_.*UINT2.*$",
    r"^test_cc_cast_.*INT4.*$",
    r"^test_cc_cast_.*INT2.*$",
    r"^test_cc_cast_.*BFLOAT16.*$",
    r"^test_cc_castlike_.*UINT4.*$",
    r"^test_cc_castlike_.*UINT2.*$",
    r"^test_cc_castlike_.*INT4.*$",
    r"^test_cc_castlike_.*INT2.*$",
    r"^test_cc_castlike_.*BFLOAT16.*$",
    # ORT returns ZipMap outputs in a different carrier format.
    r"^test_cc_zipmap_",
    # ORT rejects these QuantizeLinear/DequantizeLinear dtypes or attrs.
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
    # ORT rejects FLOAT16 scales for QLinearMatMul.
    r"^test_cc_qlinearmatmul_2D_uint8_float16$",
    r"^test_cc_qlinearmatmul_2D_int8_float16$",
    r"^test_cc_qlinearmatmul_3D_uint8_float16$",
    r"^test_cc_qlinearmatmul_3D_int8_float16$",
    # ORT is missing kernels for these ops or dtypes.
    r"^test_cc_globallppool_",
    r"^test_cc_maxroipool_",
    # The backend harness cannot feed these map-typed inputs to ORT.
    r"^test_cc_dict_vectorizer_",
    r"^test_cc_cast_map_",
    # ORT rejects these mixed-dtype or batchwise sequence patterns.
    r"^test_cc_feature_vectorizer_mixed_dtypes$",
    # More single-op kernel gaps and focused parity checks.
    r"^test_bitshift_right_uint16$",
    r"^test_bitshift_left_uint16$",
    r"^test_bitcast_",
    r"^test_cc_top_k_uint64$",
    r"^test_pow_types_float32_uint32$",
    r"^test_pow_types_float32_uint64$",
    r"^test_max_int16$",
    r"^test_max_uint16$",
    r"^test_min_int16$",
    r"^test_min_uint16$",
    # dim0_offset < dim0_size was false. Invalid dim0_offset of 0. Dimension 0 is 0
    r"^test_cc_scan_zero_trip_count$",
    # ORT CPU does not register int16/int64 kernels for Relu(14).
    r"^test_cc_relu_int16$",
    r"^test_cc_relu_int64$",
    # ORT CPU does not register these bfloat16 kernels.
    r"^test_cc_(abs|add|ceil|div|elu|equal|erf|exp|floor|gelu_default|greater|greater_or_equal|isnan|less|less_or_equal|log|mul|neg|reciprocal|relu|sigmoid|sign|softplus|softsign|sqrt|sub|tanh)_bfloat16$",
    r"^test_mod_mixed_sign_bfloat16$",
    r"^test_cc_mod_bfloat16_fmod$",
    r"^test_cc_pow_types_bfloat16_float32$",
    # ORT diverges from the reference on MaxUnpool and on align_corners
    # Resize downsample cases where scale * input_width is fractional:
    # ONNX reference / onnx-light use (scale * input_width - 1) in the
    # denominator, while ORT uses (output_width_int - 1).
    r"^test_cc_maxunpool_export_with_output_shape$",
    r"^test_resize_downsample_scales_linear_align_corners$",
    r"^test_resize_downsample_scales_cubic_align_corners$",
    # ORT IRFFT mishandles the ``inverse=1, onesided=1`` combination.
    r"^test_cc_dft_irfft(_opset19|_roundtrip|_roundtrip_opset19)?$",
    # ORT does not support Optional loop-carried state in this graph structure.
    r"^test_cc_loop16_seq_none$",
    # ORT does not support these Sequence/Optional graph patterns.
    r"^test_cc_identity_sequence$",
    r"^test_cc_identity_opt$",
    r"^test_cc_if_seq$",
    r"^test_cc_if_opt$",
    # ORT rejects the empty-name encoding of the optional ``axes`` input.
    r"^test_cc_squeeze_empty_axes_name$",
    # ORT does not support batchwise recurrent operations (layout == 1).
    r"^test_cc_gru_batchwise$",
    r"^test_cc_lstm_batchwise$",
    r"^test_cc_simple_rnn_batchwise$",
]

# Add opset-gated exclusions only for opset versions ONNX Runtime cannot load yet.
_ORT_MAX_OPSET = ort_max_supported_opset()
for _opset, _patterns in ORT_OPSET_GATED_EXCLUDE_REGEX.items():
    if _ORT_MAX_OPSET < _opset:
        ORT_EXCLUDE_REGEX.extend(_patterns)

TestOrtBackend = make_test_class(onnxruntime_backend, exclude_regex=ORT_EXCLUDE_REGEX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
