import unittest
import numpy as np
import onnxruntime as ort
from onnxruntime.capi._pybind_state import get_all_operator_schema
from onnx_light.ext_test_case import import_or_skip

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
make_test_class = import_or_skip("onnx_light.onnx.backend", "make_test_class")

# Import ORT's C++ bindings for IOBinding workaround
try:
    from onnxruntime.capi import _pybind_state as C
    _HAS_ORT_CPP_API = True
except ImportError:
    _HAS_ORT_CPP_API = False


# Mapping from NumPy dtype to ONNX TensorProto data type
_NUMPY_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE = {
    np.dtype('float32'): 1,  # FLOAT
    np.dtype('uint8'): 2,  # UINT8
    np.dtype('int8'): 3,  # INT8
    np.dtype('uint16'): 4,  # UINT16
    np.dtype('int16'): 5,  # INT16
    np.dtype('int32'): 6,  # INT32
    np.dtype('int64'): 7,  # INT64
    np.dtype('bool'): 9,  # BOOL
    np.dtype('float64'): 11,  # DOUBLE
    np.dtype('uint32'): 12,  # UINT32
    np.dtype('uint64'): 13,  # UINT64
}

# Special handling for dtypes that need to be viewed as uint types
# Map from ONNX dtype number to (numpy view dtype, onnx tensor element type)
_SPECIAL_DTYPE_MAPPINGS = {
    # FLOAT16: view as uint16
    10: (np.dtype('uint16'), 10),
    # BFLOAT16: view as uint16  
    16: (np.dtype('uint16'), 16),
    # FLOAT8E4M3FN: view as uint8
    17: (np.dtype('uint8'), 17),
    # FLOAT8E5M2: view as uint8
    19: (np.dtype('uint8'), 19),
    # INT4: view as uint8
    21: (np.dtype('uint8'), 21),
    # INT2: view as uint8
    22: (np.dtype('uint8'), 22),
    # UINT4: view as uint8 
    29: (np.dtype('uint8'), 29),
    # UINT2: view as uint8
    30: (np.dtype('uint8'), 30),
    # FLOAT8E4M3FNUZ: view as uint8
    18: (np.dtype('uint8'), 18),
    # FLOAT8E5M2FNUZ: view as uint8
    20: (np.dtype('uint8'), 20),
    # FLOAT4E2M1: view as uint8
    31: (np.dtype('uint8'), 31),
    # FLOAT8E8M0FNU: view as uint8
    32: (np.dtype('uint8'), 32),
}


def _get_onnx_tensor_element_type_from_array(arr: np.ndarray) -> int:
    """
    Gets the ONNX tensor element type from a numpy array.
    
    For standard dtypes, uses the dtype directly.
    For special dtypes (FLOAT16, BFLOAT16, FLOAT8, etc.), infers from the array's
    dtype attribute if it has an 'onnx_dtype' annotation, otherwise returns None.
    """
    # Check if array has ONNX dtype annotation (used by ml_dtypes)
    if hasattr(arr.dtype, 'num'):
        # ml_dtypes assigns special dtype numbers
        if arr.dtype.num in _SPECIAL_DTYPE_MAPPINGS:
            return _SPECIAL_DTYPE_MAPPINGS[arr.dtype.num][1]
    
    # Check standard dtypes
    if arr.dtype in _NUMPY_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE:
        return _NUMPY_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE[arr.dtype]
    
    # Try to detect from dtype name for ml_dtypes
    dtype_name = arr.dtype.name.lower()
    if 'float16' in dtype_name or 'half' in dtype_name:
        return 10
    elif 'bfloat16' in dtype_name:
        return 16
    elif 'float8e4m3fn' in dtype_name and 'uz' not in dtype_name:
        return 17
    elif 'float8e4m3fnuz' in dtype_name:
        return 18
    elif 'float8e5m2' in dtype_name and 'uz' not in dtype_name:
        return 19
    elif 'float8e5m2fnuz' in dtype_name:
        return 20
    elif 'int4' in dtype_name:
        return 21
    elif 'int2' in dtype_name:
        return 22
    elif 'uint4' in dtype_name:
        return 29
    elif 'uint2' in dtype_name:
        return 30
    elif 'float4e2m1' in dtype_name:
        return 31
    elif 'float8e8m0' in dtype_name:
        return 32
    
    return None


def onnxruntime_backend(model, *inputs: np.ndarray) -> list[np.ndarray]:
    """
    Runs an ONNX model using ONNXRuntime.
    
    Uses IOBinding with raw memory buffers to support dtypes that NumPy/ORT
    don't natively handle (FLOAT8, BFLOAT16, INT2, INT4, etc.).

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

    # Get input and output metadata
    input_metas = sess.get_inputs()
    output_metas = sess.get_outputs()

    # Check if we need special dtype handling
    needs_iobinding = False
    if _HAS_ORT_CPP_API:
        for inp in inputs:
            onnx_dtype = _get_onnx_tensor_element_type_from_array(inp)
            if onnx_dtype and onnx_dtype in _SPECIAL_DTYPE_MAPPINGS:
                needs_iobinding = True
                break

    # Use IOBinding for special dtypes, standard run otherwise
    if needs_iobinding and _HAS_ORT_CPP_API:
        # Use C++ IOBinding API to bypass NumPy dtype conversion
        io_binding = sess.io_binding()
        
        for i, (meta, inp) in enumerate(zip(input_metas, inputs)):
            onnx_dtype = _get_onnx_tensor_element_type_from_array(inp)
            
            if onnx_dtype and onnx_dtype in _SPECIAL_DTYPE_MAPPINGS:
                # Use raw buffer binding for special dtypes
                view_dtype, tensor_type = _SPECIAL_DTYPE_MAPPINGS[onnx_dtype]
                
                # View the array as the compatible dtype for buffer access
                buffer_view = inp.view(view_dtype)
                
                # Create OrtValue from raw buffer with explicit dtype
                device = C.OrtDevice(C.OrtDevice.cpu(), C.OrtDevice.default_memory(), 0)
                ortvalue = C.OrtValue.ortvalue_from_numpy(buffer_view, device)
                
                # Bind the input
                io_binding.bind_ortvalue_input(meta.name, ortvalue)
            else:
                # Standard dtype, use normal binding
                io_binding.bind_cpu_input(meta.name, inp)
        
        # Bind outputs
        for meta in output_metas:
            io_binding.bind_output(meta.name)
        
        # Run with IOBinding
        sess.run_with_iobinding(io_binding)
        
        # Get outputs
        outputs = io_binding.get_outputs()
        return [out.numpy() for out in outputs]
    else:
        # Standard path for normal dtypes
        input_names = [inp.name for inp in input_metas]
        input_dict = dict(zip(input_names, inputs))
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
    # ORT returns ZipMap outputs in a different carrier format.
    r"^test_cc_zipmap_",
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
