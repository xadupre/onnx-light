# source: https://github.com/onnx/onnx/blob/main/onnx/numpy_helper.py
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import ml_dtypes as _ml_dtypes
import numpy as np
import numpy.typing as npt

from . import MapProto, OptionalProto, SequenceProto, TensorProto
from . import helper

if TYPE_CHECKING:
    from collections.abc import Sequence

_VALID_OPTIONAL_DTYPES = frozenset(
    {
        int(OptionalProto.UNDEFINED),
        int(OptionalProto.TENSOR),
        int(OptionalProto.SPARSE_TENSOR),
        int(OptionalProto.SEQUENCE),
        int(OptionalProto.MAP),
        int(OptionalProto.OPTIONAL),
    }
)


def to_float8e8m0(x: np.ndarray, saturate: bool = True, round_mode: str = "up") -> np.ndarray:
    """Converts a float32 NumPy array to float8e8m0 representation.

    If the input is not a float32 array, it will be cast to one first.

    Args:
        x: Input array to convert.
        saturate: Whether to saturate at max/min float8e8m0 value.
        round_mode: "nearest", "up", or "down".

    Returns:
        Array of ml_dtypes.float8_e8m0fnu values.
    """
    x_f32 = np.asarray(x, dtype=np.float32)
    f_bits = x_f32.view(np.uint32)

    # Extract exponent bits
    exponent = (f_bits >> 23) & 0xFF
    exponent = exponent.astype(np.uint16)  # use uint16 to prevent overflow during computation

    # Identify NaN or Inf
    special_mask = exponent == 0xFF  # noqa: PLR2004
    output = np.zeros_like(exponent, dtype=np.uint8)
    output[special_mask] = 0xFF  # Preserve NaN/Inf as max exponent

    # Process normal numbers
    normal_mask = ~special_mask

    if round_mode == "nearest":
        # Get guard, round, sticky, and least significant bits
        g = ((f_bits & 0x400000) > 0).astype(np.uint8)
        r = ((f_bits & 0x200000) > 0).astype(np.uint8)
        s = ((f_bits & 0x1FFFFF) > 0).astype(np.uint8)
        lsb = (exponent > 0).astype(np.uint8)

        round_up = (g == 1) & ((r == 1) | (s == 1) | (lsb == 1))

        increment = np.zeros_like(exponent)
        increment[round_up & normal_mask] = 1

        if saturate:
            max_mask = (exponent == 0xFE) & round_up & normal_mask  # noqa: PLR2004
            increment[max_mask] = 0  # Don't overflow past max value

        exponent += increment

    elif round_mode == "up":
        has_fraction = (f_bits & 0x7FFFFF) > 0
        round_up = has_fraction & normal_mask

        if saturate:
            max_mask = (exponent == 0xFE) & round_up  # noqa: PLR2004
            round_up[max_mask] = False

        exponent += round_up.astype(np.uint16)

    elif round_mode == "down":
        pass  # No rounding needed

    else:
        raise ValueError(f"Unsupported rounding mode: {round_mode}")

    # Clip exponent to uint8 range
    exponent = exponent.astype(np.uint8)

    output[normal_mask] = exponent[normal_mask]

    return output.view(_ml_dtypes.float8_e8m0fnu)  # type: ignore[name-defined]


def _unpack_4bit(data: npt.NDArray[np.uint8], dims: Sequence[int]) -> npt.NDArray[np.uint8]:
    """Converts a packed uint4 array to unpacked uint4 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    result = np.empty([data.size * 2], dtype=data.dtype)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    if result.size == np.prod(dims, dtype=np.int64) + 1:
        # handle single-element padding due to odd number of elements
        result = result[:-1]
    result.resize(dims, refcheck=False)
    return result


def _pack_4bitx2(array: np.ndarray) -> npt.NDArray[np.uint8]:
    """Converts a numpy array to flatten, packed int4/uint4.

    Elements must be in the correct range.
    """
    # Create a 1D copy
    array_flat = array.ravel().view(np.uint8).copy()
    size = array.size
    odd_sized = size % 2 == 1
    if odd_sized:
        array_flat.resize([size + 1], refcheck=False)
    array_flat &= 0x0F
    array_flat[1::2] <<= 4
    return array_flat[0::2] | array_flat[1::2]


def _unpack_2bit(data: npt.NDArray[np.uint8], dims: Sequence[int]) -> npt.NDArray[np.uint8]:
    """Converts a packed uint2 array to unpacked uint2 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    result = np.empty([data.size * 4], dtype=data.dtype)
    result[0::4] = data & 0x03
    result[1::4] = (data >> 2) & 0x03
    result[2::4] = (data >> 4) & 0x03
    result[3::4] = (data >> 6) & 0x03
    if result.size > np.prod(dims, dtype=np.int64):
        # handle padding due to non-multiple of 4 elements
        result = result[: np.prod(dims, dtype=np.int64)]
    result.resize(dims, refcheck=False)
    return result


def _pack_2bitx4(array: np.ndarray) -> npt.NDArray[np.uint8]:
    """Converts a numpy array to flatten, packed int2/uint2.

    Elements must be in the correct range.
    """
    # Create a 1D copy
    array_flat = array.ravel().view(np.uint8).copy()
    size = array.size
    pad_len = size % 4
    if pad_len:
        array_flat.resize([size + (4 - pad_len)], refcheck=False)
    array_flat &= 0x03
    array_flat[1::4] <<= 2
    array_flat[2::4] <<= 4
    array_flat[3::4] <<= 6
    return array_flat[0::4] | array_flat[1::4] | array_flat[2::4] | array_flat[3::4]


def _load_external_data_for_tensor(tensor: TensorProto, base_dir: str) -> None:
    """Loads data from an external file into tensor.raw_data.

    Args:
        tensor: a TensorProto object whose external_data field describes the file.
        base_dir: directory that contains the external data file.
    """
    location = ""
    offset: int | None = None
    length: int | None = None
    for entry in tensor.external_data:
        key = str(entry.key)
        value = str(entry.value)
        if key == "location":
            location = value
        elif key == "offset":
            offset = int(value)
        elif key == "length":
            length = int(value)

    data_path = os.path.join(base_dir, location)
    with open(data_path, "rb") as data_file:
        if offset is not None:
            data_file.seek(offset)
        tensor.raw_data = data_file.read(length) if length is not None else data_file.read()


def to_array(tensor: TensorProto, base_dir: str = "") -> np.ndarray:  # noqa: PLR0911
    """Converts a TensorProto object to a numpy array.

    This function uses ml_dtypes if the dtype is not a native numpy dtype.

    Args:
        tensor: a TensorProto object.
        base_dir: if external tensor exists, base_dir can help to find the path to it.

    Returns:
        The converted numpy array.
    """
    if tensor.data_type == TensorProto.UNDEFINED:
        raise TypeError("The element type in the input tensor is UNDEFINED.")

    tensor_dtype = tensor.data_type
    np_dtype = helper.tensor_dtype_to_np_dtype(tensor_dtype)
    storage_np_dtype = helper.tensor_dtype_to_np_dtype(
        helper.tensor_dtype_to_storage_tensor_dtype(tensor_dtype)
    )
    storage_field = helper.tensor_dtype_to_field(tensor_dtype)
    dims = list(tensor.dims)

    if tensor.data_type == TensorProto.STRING:
        utf8_strings = getattr(tensor, storage_field)
        ss = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in utf8_strings]
        return np.asarray(ss).astype(np_dtype).reshape(dims)

    # Load raw data from external tensor if it exists
    if int(tensor.data_location) == int(TensorProto.EXTERNAL):
        _load_external_data_for_tensor(tensor, base_dir)

    if len(tensor.raw_data) > 0:
        # Raw bytes support: using frombuffer.
        raw_data = bytes(tensor.raw_data)
        if sys.byteorder == "big":
            # Convert endian from little to big
            raw_data = np.frombuffer(raw_data, dtype=np_dtype).byteswap().tobytes()

        if tensor_dtype in {TensorProto.INT4, TensorProto.UINT4, TensorProto.FLOAT4E2M1}:
            data = np.frombuffer(raw_data, dtype=np.uint8)
            return _unpack_4bit(data, dims).view(np_dtype)

        if tensor_dtype in {TensorProto.UINT2, TensorProto.INT2}:
            data = np.frombuffer(raw_data, dtype=np.uint8)
            return _unpack_2bit(data, dims).view(np_dtype)

        return np.frombuffer(raw_data, dtype=np_dtype).reshape(dims)

    if tensor_dtype in {
        TensorProto.BFLOAT16,
        TensorProto.FLOAT16,
        TensorProto.INT16,
        TensorProto.UINT16,
    }:
        return (
            np.array(tensor.int32_data, dtype=np.int32)
            .view(np.uint32)
            .astype(np.uint16)
            .reshape(dims)
            .view(np_dtype)
        )

    if tensor_dtype in {
        TensorProto.FLOAT8E4M3FN,
        TensorProto.FLOAT8E4M3FNUZ,
        TensorProto.FLOAT8E5M2,
        TensorProto.FLOAT8E5M2FNUZ,
        TensorProto.FLOAT8E8M0,
        TensorProto.BOOL,
    }:
        return (
            np.array(tensor.int32_data, dtype=np.int32)
            .view(np.uint32)
            .astype(np.uint8)
            .view(np_dtype)
            .reshape(dims)
        )

    if tensor_dtype in {TensorProto.UINT4, TensorProto.INT4, TensorProto.FLOAT4E2M1}:
        data = np.array(tensor.int32_data, dtype=np.int32).view(np.uint32).astype(np.uint8)
        return _unpack_4bit(data, dims).view(np_dtype)

    if tensor_dtype in {TensorProto.UINT2, TensorProto.INT2}:
        data = np.array(tensor.int32_data, dtype=np.int32).view(np.uint32).astype(np.uint8)
        return _unpack_2bit(data, dims).view(np_dtype)

    data = getattr(tensor, storage_field)
    if tensor_dtype in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
        return np.array(data, dtype=storage_np_dtype).view(dtype=np_dtype).reshape(dims)

    return np.asarray(data, dtype=storage_np_dtype).astype(np_dtype).reshape(dims)


def tobytes_little_endian(array: np.ndarray) -> bytes:
    """Converts an array into bytes in little endian byte order.

    Args:
        array: a numpy array.

    Returns:
        Byte representation of passed array in little endian byte order.
    """
    if array.dtype.byteorder == ">" or (sys.byteorder == "big" and array.dtype.byteorder == "="):
        # Ensure that the bytes will be in little-endian byte-order.
        array = array.astype(array.dtype.newbyteorder("<"))

    return array.tobytes()


def from_array(array: np.ndarray, /, name: str | None = None) -> TensorProto:
    """Converts a numpy array into a TensorProto.

    Args:
        array: a numpy array.
        name: (optional) the name of the tensor.

    Returns:
        The converted TensorProto.
    """
    tensor = TensorProto()
    tensor.dims.extend(array.shape)
    if name:
        tensor.name = name
    if array.dtype == object or np.issubdtype(array.dtype, np.str_):
        # Special care for strings.
        tensor.data_type = TensorProto.STRING
        flat_array = array.flatten()
        string_data = []
        for e in flat_array:
            if isinstance(e, str):
                string_data.append(e.encode("utf-8"))
            elif isinstance(e, bytes):
                string_data.append(e)
            else:
                raise NotImplementedError(
                    f"Unrecognized object in the object array, expect a string, or array of "
                    f"bytes: {type(e)}"
                )
        tensor.string_data = string_data
        return tensor

    dtype = helper.np_dtype_to_tensor_dtype(array.dtype)
    if dtype in {TensorProto.INT4, TensorProto.UINT4, TensorProto.FLOAT4E2M1}:
        # Pack the array into int4
        array = _pack_4bitx2(array)

    if dtype in {TensorProto.UINT2, TensorProto.INT2}:
        # Pack the array into int2
        array = _pack_2bitx4(array)

    tensor.raw_data = tobytes_little_endian(array)
    tensor.data_type = dtype  # type: ignore[assignment]
    return tensor


def to_list(sequence: SequenceProto) -> list[Any]:
    """Converts a SequenceProto to a Python list.

    Args:
        sequence: a SequenceProto object.

    Returns:
        The converted list.
    """
    elem_type = sequence.elem_type
    if elem_type == SequenceProto.TENSOR:
        return [to_array(v) for v in sequence.tensor_values]
    if elem_type == SequenceProto.SPARSE_TENSOR:
        return [to_array(v) for v in sequence.sparse_tensor_values]  # type: ignore[arg-type]
    if elem_type == SequenceProto.SEQUENCE:
        return [to_list(v) for v in sequence.sequence_values]
    if elem_type == SequenceProto.MAP:
        return [to_dict(v) for v in sequence.map_values]
    raise TypeError("The element type in the input sequence is not supported.")


def from_list(lst: list[Any], name: str | None = None, dtype: int | None = None) -> SequenceProto:
    """Converts a Python list into a SequenceProto.

    Args:
        lst: a Python list.
        name: (optional) the name of the sequence.
        dtype: (optional) type of element in the input list, used for specifying
            sequence values when converting an empty list.

    Returns:
        The converted SequenceProto.
    """
    sequence = SequenceProto()
    if name:
        sequence.name = name

    if dtype is not None:
        elem_type = dtype
    elif len(lst) > 0:
        first_elem = lst[0]
        if isinstance(first_elem, dict):
            elem_type = SequenceProto.MAP
        elif isinstance(first_elem, list):
            elem_type = SequenceProto.SEQUENCE
        else:
            elem_type = SequenceProto.TENSOR
    else:
        # if empty input list and no dtype specified
        # choose sequence of tensors on default
        elem_type = SequenceProto.TENSOR
    sequence.elem_type = elem_type

    if (len(lst) > 0) and not all(isinstance(elem, type(lst[0])) for elem in lst):
        raise TypeError(
            "The element type in the input list is not the same "
            "for all elements and therefore is not supported as a sequence."
        )

    if elem_type == SequenceProto.TENSOR:
        for tensor in lst:
            sequence.tensor_values.extend([from_array(np.asarray(tensor))])
    elif elem_type == SequenceProto.SEQUENCE:
        for seq in lst:
            sequence.sequence_values.extend([from_list(seq)])
    elif elem_type == SequenceProto.MAP:
        for mapping in lst:
            sequence.map_values.extend([from_dict(mapping)])
    else:
        raise TypeError(
            "The element type in the input list is not a tensor, "
            "sequence, or map and is not supported."
        )
    return sequence


def to_dict(map_proto: MapProto) -> dict[Any, Any]:
    """Converts a MapProto to a Python dictionary.

    Args:
        map_proto: a MapProto object.

    Returns:
        The converted dictionary.
    """
    key_list: list[Any] = []
    if map_proto.key_type == TensorProto.STRING:
        key_list = [str(k) for k in map_proto.string_keys]
    else:
        key_list = list(map_proto.keys)

    value_list = to_list(map_proto.values)
    if len(key_list) != len(value_list):
        raise IndexError(
            f"Length of keys and values for MapProto (map name: {map_proto.name}) are not the "
            f"same."
        )
    return dict(zip(key_list, value_list, strict=False))


def from_dict(dict_: dict[Any, Any], name: str | None = None) -> MapProto:
    """Converts a Python dictionary into a MapProto.

    Args:
        dict_: Python dictionary.
        name: (optional) the name of the map.

    Returns:
        The converted MapProto.
    """
    map_proto = MapProto()
    if name:
        map_proto.name = name
    if not dict_:
        raise ValueError("Cannot convert an empty dictionary to MapProto.")
    keys = list(dict_)
    raw_key_type = np.result_type(keys[0])
    key_type = helper.np_dtype_to_tensor_dtype(raw_key_type)

    valid_key_int_types = {
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    }

    if not (all(np.result_type(key) == raw_key_type for key in keys)):
        raise TypeError(
            "The key type in the input dictionary is not the same "
            "for all keys and therefore is not valid as a map."
        )

    values = list(dict_.values())
    raw_value_type = np.result_type(values[0])
    if not all(np.result_type(val) == raw_value_type for val in values):
        raise TypeError(
            "The value type in the input dictionary is not the same "
            "for all values and therefore is not valid as a map."
        )

    value_seq = from_list(values)

    map_proto.key_type = key_type  # type: ignore[assignment]
    if key_type == TensorProto.STRING:
        map_proto.string_keys.extend(keys)
    elif key_type in valid_key_int_types:
        map_proto.keys.extend(keys)
    else:
        raise TypeError(f"Unsupported map key type: {key_type}")
    map_proto.values.CopyFrom(value_seq)
    return map_proto


def to_optional(optional: OptionalProto) -> Any | None:
    """Converts an OptionalProto to a Python optional.

    Args:
        optional: an OptionalProto object.

    Returns:
        The converted optional value, or None for UNDEFINED.
    """
    elem_type = optional.elem_type
    if elem_type == OptionalProto.UNDEFINED:
        return None
    if elem_type == OptionalProto.TENSOR:
        return to_array(optional.tensor_value)
    if elem_type == OptionalProto.SPARSE_TENSOR:
        return to_array(optional.sparse_tensor_value)  # type: ignore[arg-type]
    if elem_type == OptionalProto.SEQUENCE:
        return to_list(optional.sequence_value)
    if elem_type == OptionalProto.MAP:
        return to_dict(optional.map_value)
    if elem_type == OptionalProto.OPTIONAL:
        return to_optional(optional.optional_value)
    raise TypeError("The element type in the input optional is not supported.")


def from_optional(
    opt: Any | None, name: str | None = None, dtype: int | None = None
) -> OptionalProto:
    """Converts an optional value into an OptionalProto.

    Args:
        opt: a Python optional.
        name: (optional) the name of the optional.
        dtype: (optional) type of element in the input, used for specifying optional values
            when converting empty None. dtype must be a valid OptionalProto.DataType value.

    Returns:
        The converted OptionalProto.
    """
    optional = OptionalProto()
    if name:
        optional.name = name

    if dtype is not None:
        if dtype not in _VALID_OPTIONAL_DTYPES:
            raise TypeError(f"{dtype} must be a valid OptionalProto.DataType.")
        elem_type = dtype
    elif isinstance(opt, dict):
        elem_type = OptionalProto.MAP
    elif isinstance(opt, list):
        elem_type = OptionalProto.SEQUENCE
    elif opt is None:
        elem_type = OptionalProto.UNDEFINED
    else:
        elem_type = OptionalProto.TENSOR

    optional.elem_type = elem_type

    if opt is not None:
        if elem_type == OptionalProto.TENSOR:
            optional.tensor_value.CopyFrom(from_array(opt))
        elif elem_type == OptionalProto.SEQUENCE:
            optional.sequence_value.CopyFrom(from_list(opt))
        elif elem_type == OptionalProto.MAP:
            optional.map_value.CopyFrom(from_dict(opt))
        else:
            raise TypeError(
                "The element type in the input is not a tensor, "
                "sequence, or map and is not supported."
            )
    return optional


def create_random_int(input_shape: tuple[int, ...], dtype: np.dtype, seed: int = 1) -> np.ndarray:
    """Creates a random integer array for backend/test/case/node.

    Args:
        input_shape: The shape for the returned integer array.
        dtype: The NumPy data type for the returned integer array.
        seed: The seed for np.random.

    Returns:
        Random integer array.
    """
    np.random.seed(seed)
    if dtype in (
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ):
        # the range of np.random.randint is int32; set a fixed boundary if overflow
        end = min(np.iinfo(dtype).max, np.iinfo(np.int32).max)
        start = max(np.iinfo(dtype).min, np.iinfo(np.int32).min)
        return np.random.randint(start, end, size=input_shape).astype(dtype)
    raise TypeError(f"{dtype} is not supported by create_random_int.")


def saturate_cast(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Performs saturate-cast of a numpy array to the target dtype.

    Values outside the representable range of the target dtype are clamped to the
    maximum or minimum representable value of that dtype.

    Args:
        x: Input array.
        dtype: Target numpy dtype.

    Returns:
        The clamped and cast array.
    """
    if np.issubdtype(dtype, np.integer) or dtype in (
        _ml_dtypes.int4,  # type: ignore[name-defined]
        _ml_dtypes.uint4,  # type: ignore[name-defined]
        _ml_dtypes.int2,  # type: ignore[name-defined]
        _ml_dtypes.uint2,  # type: ignore[name-defined]
    ):
        info = _ml_dtypes.iinfo(dtype)  # type: ignore[name-defined]
        x = np.round(x)
    else:
        info = _ml_dtypes.finfo(dtype)  # type: ignore[name-defined]

    return np.clip(x, info.min, info.max).astype(dtype)  # type: ignore[no-any-return]
