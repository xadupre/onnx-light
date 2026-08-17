import platform
import re
import sys
import unittest
import numpy as np
from onnx_light.ext_test_case import import_or_skip, InferenceSessionAllTypes

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
make_test_class = import_or_skip("onnx_light.onnx.backend", "make_test_class")
ORT_MAX_IR_VERSION = 13


def onnxruntime_backend(model, *inputs: np.ndarray) -> list[np.ndarray]:
    """
    Runs an ONNX model using ONNXRuntime with support for all dtypes.

    Args:
        model: The ONNX model (onnx_light.ModelProto) to run
        *inputs: Input arrays for the model

    Returns:
        List of output arrays from the model
    """
    if model.ir_version > ORT_MAX_IR_VERSION:
        raise unittest.SkipTest(
            f"model IR version {model.ir_version} exceeds "
            f"onnxruntime maximum {ORT_MAX_IR_VERSION}"
        )

    sess = InferenceSessionAllTypes(model)

    # Get input names and create feed dict
    input_names = [inp.name for inp in sess._sess.get_inputs()]
    input_dict = dict(zip(input_names, inputs))

    # Run inference
    outputs = sess.run(None, input_dict)
    return outputs


# Highest default-domain (``ai.onnx``) opset version each ONNX Runtime release can
# actually *load*. The registered operator schemas advertise the next,
# still-under-development opset before the runtime is able to load a model
# targeting it (e.g. ONNX Runtime 1.28 bundles ONNX 1.22, whose schemas report
# ``since_version`` 27, yet ``InferenceSession`` rejects opset 27 with "Current
# official support for domain ai.onnx is till opset 26"). A static mapping keeps
# the exclusion list accurate without loading every backend test model.
#
# Values follow the ONNX Runtime <-> ONNX opset compatibility table
# (https://onnxruntime.ai/docs/reference/compatibility.html) and, for the newest
# releases, the ONNX version each runtime bundles (cmake/deps.txt).
_ORT_VERSION_TO_MAX_OPSET = {
    (1, 17): 20,
    (1, 18): 21,
    (1, 19): 21,
    (1, 20): 21,
    (1, 21): 22,
    (1, 22): 22,
    (1, 23): 23,
    (1, 24): 24,
    (1, 25): 26,
    (1, 26): 26,
    (1, 27): 26,
    (1, 28): 26,
}


def ort_max_supported_opset() -> int:
    """
    Returns the highest default-domain opset version ONNX Runtime can load.

    Looks up the installed ``onnxruntime`` ``(major, minor)`` version in
    ``_ORT_VERSION_TO_MAX_OPSET``. This mirrors the official ONNX Runtime <-> ONNX
    opset compatibility table and, unlike the registered operator schemas
    (``get_all_operator_schema``), reflects the opset the runtime can actually
    load rather than the next still-under-development opset it merely registers
    schemas for.

    For a version newer than the last tabulated entry, the ceiling of the highest
    known version is used (never runs cases the runtime may not load); for an
    older-than-tabulated version, the lowest known ceiling is used.

    Returns:
        The highest default-domain opset version ONNX Runtime can load.
    """
    import onnxruntime

    parts = re.findall(r"\d+", onnxruntime.__version__)
    major, minor = int(parts[0]), int(parts[1])
    key = (major, minor)
    if key in _ORT_VERSION_TO_MAX_OPSET:
        return _ORT_VERSION_TO_MAX_OPSET[key]

    known = sorted(_ORT_VERSION_TO_MAX_OPSET)
    if key < known[0]:
        return _ORT_VERSION_TO_MAX_OPSET[known[0]]
    lower = max(version for version in known if version <= key)
    return _ORT_VERSION_TO_MAX_OPSET[lower]


# Opset version at which the cases below were introduced. They are only excluded
# when the installed ONNX Runtime does not yet support that opset.
OPSET_27 = 27
OPSET_28 = 28

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
    ],
    OPSET_28: [
        # Celu-28 adds float16/bfloat16 support; these test cases target opset 28.
        r"^test_cc_celu_float16$",
        r"^test_cc_celu_bfloat16$",
        # SwiGLU is opset 28 (onnx#8202).
        r"^test_cc_swiglu.*$",
    ],
}

ORT_EXCLUDE_REGEX = [
    # ORT/reference parity mismatches in focused C++ cases.
    r"^test_cc_stft_complex_batched$",
    r"^test_cc_image_decoder_",
    # Preview ops/functions are not registered in ORT.
    r"^test_cc_flexattention_",
    # Light-only ai.rt ops are not registered in ORT.
    r"^test_cc_delayedinitializer_",
    # ORT exposes different Attention intermediates than the ONNX reference.
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal$",
    r"^test_cc_attention_3d_with_past_and_present_qk_matmul_bias$",
    r"^test_cc_attention_4d_with_qk_matmul_bias$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask$",
    r"^test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask$",
    r"^test_cc_attention_4d_softcap_neginf_mask$",
    r"^test_cc_attention_4d_softcap_neginf_mask_poison$",
    r"^test_cc_attention_23_boolmask_fullymasked_row_nan_robustness$",
    r"^test_cc_attention_causal_boolmask_nan_robustness$",
    r"^test_cc_attention_23_fullymasked_qk_matmul_output_mode3_zero$",
    r"^test_cc_attention_24_fullymasked_qk_matmul_output_mode3_zero$",
    r"^test_cc_attention_4d_causal_nonpad_attn_mask_composition$",
    r"^test_cc_attention_4d_causal_nonpad_batch_prefill$",
    r"^test_cc_attention_4d_causal_nonpad_continued_prefill$",
    r"^test_cc_attention_4d_causal_nonpad_negative_offset_structural_empty$",
    r"^test_cc_attention_4d_gqa_causal_nonpad_decode$",
    r"^test_cc_attention_4d_gqa_causal_nonpad_decode_fp16$",
    # ORT does not yet implement the opset-24 offset-aware (bottom-right)
    # causal frontier for an external KV cache (``nonpad_kv_seqlen`` without
    # ``past_key``); see ONNX PR #8068.
    r"^test_cc_attention_4d_causal_nonpad_kv_continued_prefill$",
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
    # ORT returns ZipMap outputs in a different carrier format.
    r"^test_cc_zipmap_",
    # ORT only supports scalar/1-element zero points for MatMulInteger.
    r"^test_cc_matmulinteger_per_col_b_zp$",
    r"^test_cc_matmulinteger_per_row_a_zp$",
    # ORT rejects FLOAT16 scales for QLinearMatMul.
    r"^test_cc_qlinearmatmul_2D_uint8_float16$",
    r"^test_cc_qlinearmatmul_2D_int8_float16$",
    r"^test_cc_qlinearmatmul_3D_uint8_float16$",
    r"^test_cc_qlinearmatmul_3D_int8_float16$",
    # ORT 1.27 still mishandles the opset-18 ceil_mode+count_include_pad
    # AveragePool tail windows tracked by microsoft/onnxruntime#29629.
    r"^test_cc_averagepool_18_ceil_count_include_pad_1d$",
    r"^test_cc_averagepool_18_ceil_count_include_pad_2d$",
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
    # TopK(sorted=0) is valid, but ORT may return the top-k set in a different
    # order because the schema leaves it undefined.
    r"^test_cc_top_k_not_sorted$",
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
    # ...
    r"e2m1.*",
    r"e4m3.*",
    r"e5m2.*",
    r"float8.*",
    r"quantizelinear_u?int2.*",
    r"quantizelinear_u?int4.*",
    # ...
    r"E2M1.*",
    r"E4M3.*",
    r"E5M2.*",
    r"FLOAT8.*",
    r"to_BFLOAT16.*",
    r"to_U?INT[24].*",
    r"castlike_U?INT[24].*",
    r"cast_U?INT[24].*",
    r"to_STRING",
    r"prelu_inf.*",
    r"sequence.*",
    # ONNX Runtime's Where kernel does not implement these dtypes.
    r"^test_cc_where_(bool|int8|int16|uint16|uint32|uint64)$",
]

# Exclusions that only apply on a given platform. On macOS/arm64 ONNX Runtime
# saturates out-of-range floating-point values when casting to ``UINT16`` (for
# example ``-1.5 -> 0``), whereas onnx-light — like ONNX Runtime on x86 —
# applies the modular wrap the reference kernel defines (``-1.5 -> 65535``).
# Out-of-range float-to-integer conversion is undefined behaviour, so these
# cases only agree with ONNX Runtime off macOS; keep the coverage everywhere
# else and drop just these here.
_MACOS_ORT_EXCLUDE_REGEX = [r"^test_cc_cast(like)?_(FLOAT|FLOAT16|BFLOAT16)_to_UINT16$"]
if sys.platform == "darwin":
    ORT_EXCLUDE_REGEX.extend(_MACOS_ORT_EXCLUDE_REGEX)

# Add opset-gated exclusions only for opset versions ONNX Runtime cannot load yet.
_ORT_MAX_OPSET = ort_max_supported_opset()
for _opset, _patterns in ORT_OPSET_GATED_EXCLUDE_REGEX.items():
    if _ORT_MAX_OPSET < _opset:
        ORT_EXCLUDE_REGEX.extend(_patterns)

# On Apple silicon (arm64), ONNX Runtime saturates out-of-range float -> UINT16
# casts to 0 (the AArch64 ``fcvtzu`` instruction saturates), whereas onnx-light
# and ORT on x86-64 produce the well-defined modular result (e.g. ``-1.5`` casts
# to ``65535``). Converting an out-of-range floating-point value to an unsigned
# integer is undefined behaviour in C/C++ and ONNX leaves it unspecified, so
# exclude these float-family -> UINT16 parity checks on macOS only.
if platform.system() == "Darwin":
    ORT_EXCLUDE_REGEX.append(r"^test_cc_cast(like)?_(FLOAT|FLOAT16|BFLOAT16)_to_UINT16$")

TestOrtBackend = make_test_class(onnxruntime_backend, exclude_regex=ORT_EXCLUDE_REGEX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
