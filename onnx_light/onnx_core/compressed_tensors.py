"""Adapts explicitly supported compressed-tensors weight checkpoint layouts.

The boundary is an in-memory mapping of NumPy arrays, not a checkpoint loader.
No PyTorch or compressed-tensors import is required.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import math
import struct

import numpy

from onnx_light import onnx
import onnx_light.onnx.helper as oh
from . import quantization

UPSTREAM_REVISION = "eb058d630f7344099459b6795b1e3eeffdda4b40"
DEFAULT_MAX_BYTES = 1 << 30


@dataclass
class CompressedTensorsWeights:
    """Carries one module's weight tensors and explicit QuantizationArgs.

    Tensor names are local suffixes such as ``weight_packed``, not file paths.
    Activation configuration and model-level target selection are not accepted.
    """

    compression_format: str
    quantization_args: dict
    tensors: Mapping
    upstream_revision: str = UPSTREAM_REVISION


def _require(condition, message):
    if not condition:
        raise ValueError(f"compressed-tensors: {message}")


def _array(tensors, name, dtype, shape=None):
    _require(name in tensors, f"missing {name}.")
    value = tensors[name]
    _require(isinstance(value, numpy.ndarray), f"{name} must be a loaded NumPy array.")
    _require(value.dtype == numpy.dtype(dtype), f"{name} must have dtype {numpy.dtype(dtype)}.")
    if shape is not None:
        _require(value.shape == shape, f"{name} must have shape {shape}, got {value.shape}.")
    return value


def _metadata(checkpoint, max_bytes):
    _require(
        isinstance(checkpoint, CompressedTensorsWeights), "expected CompressedTensorsWeights."
    )
    _require(checkpoint.upstream_revision == UPSTREAM_REVISION, "unsupported upstream revision.")
    _require(
        isinstance(checkpoint.compression_format, str), "compression_format must be a string."
    )
    _require(type(max_bytes) is int and max_bytes > 0, "max_bytes must be a positive integer.")
    args = checkpoint.quantization_args
    _require(isinstance(args, dict), "quantization_args must be a QuantizationArgs dictionary.")
    _require(
        all(key in args for key in ("num_bits", "type", "symmetric", "strategy")),
        "explicit num_bits, type, symmetric and strategy are required.",
    )
    _require(type(args["num_bits"]) is int, "num_bits must be an integer.")
    _require(type(args["symmetric"]) is bool, "symmetric must be a boolean.")
    _require(
        set(args)
        <= {
            "num_bits",
            "type",
            "symmetric",
            "strategy",
            "group_size",
            "block_structure",
            "dynamic",
            "actorder",
            "scale_dtype",
            "zp_dtype",
            "observer",
            "observer_kwargs",
        },
        "unknown QuantizationArgs fields; pass resolved weight arguments only.",
    )
    _require(
        isinstance(args["type"], str) and isinstance(args["strategy"], str),
        "type and strategy must be strings.",
    )
    _require(args.get("dynamic", False) is False, "dynamic weight quantization is unsupported.")
    _require(
        args.get("actorder") is None or args.get("actorder") is False,
        "activation ordering/g_idx is unsupported.",
    )
    _require(args.get("block_structure") is None, "block_structure is unsupported.")
    tensors = checkpoint.tensors
    _require(isinstance(tensors, Mapping), "tensors must be a mapping of loaded arrays.")
    total = 0
    for name, value in tensors.items():
        _require(isinstance(name, str), "tensor names must be strings.")
        _require(isinstance(value, numpy.ndarray), f"{name} must be a loaded NumPy array.")
        total += value.nbytes
        _require(total <= max_bytes, "checkpoint exceeds max_bytes.")
    return args, tensors


def _shape(rows, columns, max_bytes):
    _require(rows > 0 and columns > 0, "weights must be a nonempty matrix [N,K].")
    _require(rows * columns <= max_bytes // 4, "logical FLOAT32 weights exceed max_bytes.")
    return rows, columns


def _keys(tensors, required, optional=()):
    _require(
        set(required) <= set(tensors) <= set(required) | set(optional),
        f"expected tensors {sorted(required)} with optional {sorted(optional)}; "
        f"got {sorted(tensors)}.",
    )


def _integer_geometry(args, shape):
    bits = args["num_bits"]
    _require(args["type"] == "int" and 1 <= bits <= 8, "pack-quantized requires int, 1--8 bits.")
    rows, columns = shape
    strategy = args["strategy"]
    _require(strategy in ("tensor", "channel", "group"), "unsupported integer strategy.")
    if strategy == "group":
        group = args.get("group_size")
        _require(type(group) is int and group > 0, "group_size must be a positive integer.")
        _require(
            columns % group == 0,
            "K must be divisible by group_size; partial groups are unsupported.",
        )
    else:
        _require(args.get("group_size") in (None, -1), "group_size conflicts with strategy.")
        group = columns
    return bits, group, (1,) if strategy == "tensor" else (rows, columns // group)


def _integer_plan(args, shape, profile=None, reconstruction_dtype=None):
    bits, group, _ = _integer_geometry(args, shape)
    rows, columns = shape
    _require(
        (rows * columns if args["strategy"] == "tensor" else group) <= 0xFFFFFFFF,
        "portable affine blocks must contain at most UINT32_MAX codes.",
    )
    block_size = rows * columns if args["strategy"] == "tensor" else group
    plan = quantization.make_quantization_plan(
        quantization.QuantizationFormat.INT8, rows * columns, block_size
    )
    if profile is not None:
        plan.format = profile
    if reconstruction_dtype is not None:
        plan.reconstruction_dtype = reconstruction_dtype
    run = plan.run(0)
    layout = run.layout
    layout.bits = bits
    run.layout = layout
    plan.set_run(0, run)
    return plan, [block_size] * (rows * columns // block_size)


def _pack_codes(codes, bits):
    """Packs portable LSB-first codes without a floating-point conversion."""
    data = bytearray((len(codes) * bits + 7) // 8)
    mask = (1 << bits) - 1
    for index, code in enumerate(codes):
        value = (int(code) & mask) << ((index * bits) % 8)
        byte = index * bits // 8
        data[byte] |= value & 255
        if value > 255:
            data[byte + 1] |= value >> 8
    return data


def _unpack_codes(data, count, bits):
    values = numpy.empty(count, dtype=numpy.int8)
    for index in range(count):
        byte, shift = divmod(index * bits, 8)
        value = data[byte] >> shift
        if shift + bits > 8:
            value |= data[byte + 1] << (8 - shift)
        value &= (1 << bits) - 1
        values[index] = value - (1 << bits) if value >= (1 << (bits - 1)) else value
    return values


def _dtype_metadata(args, scale, zero):
    for key, expected in (("scale_dtype", scale), ("zp_dtype", zero)):
        value = args.get(key)
        _require(
            value is None or value in (expected, f"torch.{expected}"),
            f"{key} must be {expected} or null for this adapter.",
        )


def _unpack_int32(packed, count, bits, axis=1):
    """Unpacks upstream offset-binary words, including cross-word codes."""
    words = packed.T if axis == 0 else packed
    result = numpy.empty((words.shape[0], count), dtype=numpy.int8)
    remainder = count * bits % 32
    for row_index, row in enumerate(words):
        if remainder:
            _require(
                (int(row[-1]) & 0xFFFFFFFF) >> remainder == 0,
                "nonzero int32 padding bits are unsupported.",
            )
        for index in range(count):
            word, shift = divmod(index * bits, 32)
            code = (int(row[word]) & 0xFFFFFFFF) >> shift
            if shift + bits > 32:
                code |= (int(row[word + 1]) & 0xFFFFFFFF) << (32 - shift)
            result[row_index, index] = (code & ((1 << bits) - 1)) - (1 << (bits - 1))
    return result.T if axis == 0 else result


def _pack_int32(codes, bits, axis=1):
    values = codes.T if axis == 0 else codes
    rows, columns = values.shape
    result = numpy.zeros((rows, (columns * bits + 31) // 32), dtype=numpy.uint32)
    for row_index, row in enumerate(values):
        for index, code in enumerate(row):
            word, shift = divmod(index * bits, 32)
            value = int(code) + (1 << (bits - 1))
            result[row_index, word] |= numpy.uint32((value << shift) & 0xFFFFFFFF)
            if shift + bits > 32:
                result[row_index, word + 1] |= numpy.uint32(value >> (32 - shift))
    result = result.view(numpy.int32)
    return numpy.ascontiguousarray(result.T if axis == 0 else result)


def _finite_affine(codes, scale, zero):
    distance = max(abs(int(codes.min()) - int(zero)), abs(int(codes.max()) - int(zero)))
    with numpy.errstate(over="ignore"):
        bound = numpy.float32(distance) * numpy.float32(scale)
    _require(numpy.isfinite(bound), "affine codes/scales overflow FLOAT32 reconstruction.")


def _integer_import(args, tensors, max_bytes):
    required = {"weight_packed", "weight_shape", "weight_scale"}
    _keys(tensors, required if args["symmetric"] else required | {"weight_zero_point"})
    shape_tensor = _array(tensors, "weight_shape", numpy.int64, (2,))
    shape = _shape(*(int(dim) for dim in shape_tensor), max_bytes)
    bits, _, parameter_shape = _integer_geometry(args, shape)
    _dtype_metadata(args, "float32", "int8")
    packed = _array(
        tensors, "weight_packed", numpy.int32, (shape[0], (shape[1] * bits + 31) // 32)
    )
    scales = _array(tensors, "weight_scale", numpy.float32, parameter_shape)
    _require(
        numpy.all(numpy.isfinite(scales)) and numpy.all(scales > 0),
        "weight_scale must be finite and positive.",
    )
    # Every block carries three doubles in the portable affine schema.
    blocks = scales.size
    block_count = shape[0] * shape[1] // blocks
    _require(
        1 + blocks * (24 + (block_count * bits + 7) // 8) <= max_bytes,
        "encoded affine payload exceeds max_bytes.",
    )
    if args["symmetric"]:
        zeros = numpy.zeros(parameter_shape, dtype=numpy.int8)
    elif args["strategy"] == "tensor":
        zeros = _array(tensors, "weight_zero_point", numpy.int8, parameter_shape)
    else:
        zeros_packed = _array(
            tensors,
            "weight_zero_point",
            numpy.int32,
            ((shape[0] * bits + 31) // 32, parameter_shape[1]),
        )
        zeros = _unpack_int32(zeros_packed, shape[0], bits, axis=0)
    _require(
        numpy.all(zeros >= -(1 << (bits - 1))) and numpy.all(zeros < (1 << (bits - 1))),
        "zero points exceed the signed code range.",
    )
    codes = _unpack_int32(packed, shape[1], bits)
    plan, counts = _integer_plan(args, shape)
    payload = bytearray(b"\0")
    start = 0
    for count, scale, zero in zip(counts, scales.flat, zeros.flat):
        block_codes = codes.flat[start : start + count]
        _finite_affine(block_codes, scale, zero)
        payload.extend(struct.pack("<ddd", float(scale), int(zero), 0))
        payload.extend(_pack_codes(block_codes, bits))
        start += count
    return onnx.EncodedValueProto(
        struct_type=quantization.make_quantization_type(plan),
        logical_type=oh.make_tensor_type_proto(onnx.TensorProto.FLOAT, list(shape)),
        raw_data=bytes(payload),
    )


_MX_FORMATS = {
    "mxfp4-pack-quantized": (4, 32, "mxfp4", "Mxfp4Weights"),
    "mxfp6-pack-quantized": (6, 32, "mxfp6_e2m3", "Mxfp6E2m3Weights"),
    "mxfp8-quantized": (8, 32, "mxfp8", "Mxfp8Weights"),
    "nvfp4-pack-quantized": (4, 16, "nvfp4", "Nvfp4Weights"),
}


def _mx_metadata(compression_format, args):
    _require(
        compression_format in _MX_FORMATS,
        f"unsupported compression format {compression_format!r}; "
        "ordinary FP8 schema adaptation is separate from MXFP8.",
    )
    bits, group, codec, class_name = _MX_FORMATS[compression_format]
    nvfp4 = codec == "nvfp4"
    _require(
        args["type"] == "float" and args["num_bits"] == bits and args["symmetric"],
        f"{compression_format} requires symmetric float{bits} weights.",
    )
    _require(
        args["strategy"] == ("tensor_group" if nvfp4 else "group")
        and type(args.get("group_size")) is int
        and args["group_size"] == group,
        f"{compression_format} requires axis-1 groups of {group}.",
    )
    dtype = "float8_e4m3fn" if nvfp4 else "uint8"
    _dtype_metadata(args, dtype, dtype)
    return bits, group, codec, class_name


def _runtime_tensor(name, array, dtype):
    return quantization.runtime.tensor_from_numpy(
        name,
        dtype,
        list(array.shape),
        numpy.ascontiguousarray(array).view(numpy.uint8).reshape(-1),
    )


def _mx_import(compression_format, args, tensors, max_bytes):
    import ml_dtypes

    bits, group, codec, class_name = _mx_metadata(compression_format, args)
    weight_name = "weight" if bits == 8 else "weight_packed"
    required = {weight_name, "weight_scale"}
    if codec == "nvfp4":
        required.add("weight_global_scale")
    _keys(tensors, required)
    dtype = ml_dtypes.float8_e4m3fn if bits == 8 else numpy.uint8
    packed = _array(tensors, weight_name, dtype)
    _require(packed.ndim == 2, f"{weight_name} must be a matrix.")
    _require(packed.shape[1] * 8 % bits == 0, "invalid packed column extent.")
    shape = _shape(packed.shape[0], packed.shape[1] * 8 // bits, max_bytes)
    _require(shape[1] % group == 0, f"K must be divisible by {group}.")
    nvfp4 = codec == "nvfp4"
    scale_dtype = ml_dtypes.float8_e4m3fn if nvfp4 else numpy.uint8
    scales = _array(tensors, "weight_scale", scale_dtype, (shape[0], shape[1] // group))
    if nvfp4:
        global_scale = _array(tensors, "weight_global_scale", numpy.float32, (1,))
        _require(
            numpy.isfinite(global_scale[0]) and global_scale[0] > 0,
            "weight_global_scale must be finite and positive.",
        )
        _require(
            numpy.all(numpy.isfinite(scales)) and numpy.all(scales > 0),
            "NVFP4 local scales must be finite and positive.",
        )
    else:
        _require(numpy.all(scales != 255), "E8M0 scale byte 255 is unsupported.")
    if bits == 8:
        _require(numpy.all(numpy.isfinite(packed)), "MXFP8 weight codes must be finite E4M3FN.")
    weights = getattr(quantization, class_name)()
    setattr(
        weights,
        weight_name,
        _runtime_tensor(
            weight_name,
            packed,
            onnx.TensorProto.FLOAT8E4M3FN if bits == 8 else onnx.TensorProto.UINT8,
        ),
    )
    weights.weight_scale = _runtime_tensor(
        "weight_scale", scales, onnx.TensorProto.FLOAT8E4M3FN if nvfp4 else onnx.TensorProto.UINT8
    )
    if nvfp4:
        weights.weight_global_scale = float(global_scale[0])
    return getattr(quantization, f"import_{codec}_weights")(weights)


def import_compressed_tensors_weights(checkpoint, *, max_bytes=DEFAULT_MAX_BYTES):
    """Imports loaded checkpoint weights without dequantizing or requantizing.

    Rejects unsupported metadata and representations rather than substituting a
    portable numerical profile. Returns a self-contained EncodedValueProto with
    logical FLOAT32 weights. All accepted code/parameter mappings are lossless.
    """
    args, tensors = _metadata(checkpoint, max_bytes)
    if checkpoint.compression_format == "pack-quantized":
        return _integer_import(args, tensors, max_bytes)
    return _mx_import(checkpoint.compression_format, args, tensors, max_bytes)


def _encoded_shape(value, max_bytes):
    _require(isinstance(value, onnx.EncodedValueProto), "export requires EncodedValueProto.")
    _require(
        value.data_location == onnx.TensorProto.DEFAULT and not value.external_data,
        "export requires inline storage without external_data references.",
    )
    _require(
        not value.has_parameter_ref(), "materialize shared quantization parameters before export."
    )
    _require(len(value.raw_data) <= max_bytes, "encoded payload exceeds max_bytes.")
    logical = value.logical_type
    _require(
        logical.has_tensor_type() and logical.tensor_type.elem_type == onnx.TensorProto.FLOAT,
        "export requires logical FLOAT32 weights.",
    )
    dims = logical.tensor_type.shape.dim
    _require(
        len(dims) == 2 and all(dim.has_dim_value() for dim in dims),
        "export requires a concrete matrix shape.",
    )
    return _shape(*(int(dim.dim_value) for dim in dims), max_bytes)


def _integer_export(value, args, shape, max_bytes):
    bits, _, parameter_shape = _integer_geometry(args, shape)
    _dtype_metadata(args, "float32", "int8")
    parameter_count = math.prod(parameter_shape)
    _require(parameter_count * 24 < max_bytes, "affine parameters exceed max_bytes.")
    count = shape[0] * shape[1] // parameter_count
    expected_bytes = 1 + parameter_count * (24 + (count * bits + 7) // 8)
    data = value.raw_data
    _require(
        len(data) == expected_bytes and data[0] == 0, "invalid affine payload extent/header."
    )
    name = value.struct_type.name
    _require(
        isinstance(name, str)
        and name.startswith(("onnx_light.quantization.v1/", "onnx_light.quantization.v2/")),
        "expected an inline portable affine schema; catalogue references are unsupported.",
    )
    profile = quantization.parse_quantization_format(name.split("/", 1)[1])
    reconstruction = onnx.TensorProto.FLOAT if ".v2/" in name else onnx.TensorProto.DOUBLE
    plan, counts = _integer_plan(args, shape, profile, reconstruction)
    expected_type = quantization.make_quantization_type(plan)
    _require(
        value.struct_type.SerializeToString() == expected_type.SerializeToString(),
        "affine schema does not match the requested bits/grouping; "
        "transforms, unsigned codes, codebooks and outliers are unsupported.",
    )
    scales = numpy.empty(parameter_shape, dtype=numpy.float32)
    zeros = numpy.empty(parameter_shape, dtype=numpy.int8)
    codes = numpy.empty(shape[0] * shape[1], dtype=numpy.int8)
    position, start = 1, 0
    for index, count in enumerate(counts):
        scale, zero, offset = struct.unpack_from("<ddd", data, position)
        _require(
            numpy.isfinite(scale) and 0 < scale <= numpy.finfo(numpy.float32).max,
            "scale must be finite, positive and representable as FLOAT32.",
        )
        _require(
            float(numpy.float32(scale)) == scale,
            "scale conversion to FLOAT32 would not be lossless.",
        )
        _require(
            numpy.isfinite(zero)
            and zero.is_integer()
            and -(1 << (bits - 1)) <= zero < (1 << (bits - 1)),
            "invalid affine zero point.",
        )
        _require(offset == 0, "nonzero offsets cannot be exported losslessly.")
        _require(
            not args["symmetric"] or zero == 0, "symmetric export requires zero zero_points."
        )
        scales.flat[index], zeros.flat[index] = scale, int(zero)
        position += 24
        size = (count * bits + 7) // 8
        chunk = data[position : position + size]
        remainder = count * bits % 8
        _require(not remainder or chunk[-1] >> remainder == 0, "nonzero portable padding bits.")
        block_codes = _unpack_codes(chunk, count, bits)
        _finite_affine(block_codes, scale, zero)
        codes[start : start + count] = block_codes
        position += size
        start += count
    tensors = {
        "weight_packed": _pack_int32(codes.reshape(shape), bits),
        "weight_shape": numpy.array(shape, dtype=numpy.int64),
        "weight_scale": scales,
    }
    if not args["symmetric"]:
        tensors["weight_zero_point"] = (
            zeros if args["strategy"] == "tensor" else _pack_int32(zeros, bits, axis=0)
        )
    return tensors


def _mx_export(value, compression_format, args):
    import ml_dtypes

    bits, _, codec, _ = _mx_metadata(compression_format, args)
    weights = getattr(quantization, f"export_{codec}_weights")(value)
    weight_name = "weight" if bits == 8 else "weight_packed"

    def array(tensor, dtype):
        return (
            quantization.runtime.tensor_to_numpy(tensor)
            .view(dtype)
            .reshape(tuple(tensor.shape))
            .copy()
        )

    nvfp4 = codec == "nvfp4"
    tensors = {
        weight_name: array(
            getattr(weights, weight_name), ml_dtypes.float8_e4m3fn if bits == 8 else numpy.uint8
        ),
        "weight_scale": array(
            weights.weight_scale, ml_dtypes.float8_e4m3fn if nvfp4 else numpy.uint8
        ),
    }
    if nvfp4:
        tensors["weight_global_scale"] = numpy.array(
            [weights.weight_global_scale], dtype=numpy.float32
        )
    return tensors


def export_compressed_tensors_weights(
    value,
    *,
    compression_format,
    quantization_args,
    upstream_revision=UPSTREAM_REVISION,
    max_bytes=DEFAULT_MAX_BYTES,
):
    """Exports codes and parameters with an explicitly requested checkpoint schema.

    Returns:
        CompressedTensorsWeights with owned arrays. Rejects any conversion that
        would round codes or parameters. Logical reconstruction is FLOAT32;
        upstream decompressor output dtypes are not inferred from profile names.
    """
    _require(
        isinstance(quantization_args, dict),
        "quantization_args must be a QuantizationArgs dictionary.",
    )
    checkpoint = CompressedTensorsWeights(
        compression_format, dict(quantization_args), {}, upstream_revision
    )
    args, _ = _metadata(checkpoint, max_bytes)
    shape = _encoded_shape(value, max_bytes)
    if compression_format == "pack-quantized":
        checkpoint.tensors = _integer_export(value, args, shape, max_bytes)
    else:
        _mx_metadata(compression_format, args)
        checkpoint.tensors = _mx_export(value, compression_format, args)
    _metadata(checkpoint, max_bytes)
    return checkpoint


__all__ = [
    "UPSTREAM_REVISION",
    "CompressedTensorsWeights",
    "export_compressed_tensors_weights",
    "import_compressed_tensors_weights",
]
