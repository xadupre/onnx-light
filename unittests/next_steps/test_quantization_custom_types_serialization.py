import base64
import json
import unittest

FLOAT = 1
UINT8 = 2
INT32 = 6
INT64 = 7
BOOL = 9
FLOAT16 = 10
UINT4 = 21
INT4 = 22


def _encode(value):
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"bytes"}:
            return base64.b64decode(value["bytes"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _round_trip(value):
    payload = json.dumps(_encode(value), separators=(",", ":"), sort_keys=True).encode("utf-8")
    parsed = _decode(json.loads(payload))
    assert (
        json.dumps(_encode(parsed), separators=(",", ":"), sort_keys=True).encode("utf-8")
        == payload
    )
    return parsed


def _constant(name, data_type, value, dims=()):
    return {
        "name": name,
        "constant": {"data_type": data_type, "dims": list(dims), "value": value},
    }


def _array(name, element_type, dimension):
    return {
        "name": name,
        "type": {"array": {"element_type": element_type, "dimension": dimension}},
    }


def _structure(name, fields):
    return {"name": name, "structure": {"fields": fields}}


def _constants(struct_type):
    return {
        field["name"]: field["constant"]["value"]
        for field in struct_type["structure"]["fields"]
        if "constant" in field
    }


def _make_specialized_linear():
    return {
        "storage_type": INT4,
        "bits": 4,
        "symmetric": True,
        "scale_float": 0.125,
        "zero_point": 0,
        "axis": -1,
    }


def _make_custom_linear():
    return _structure(
        "LINEAR",
        [
            _array("values", INT4, 128),
            _constant("storage_type", INT32, INT4),
            _constant("bits", INT32, 4),
            _constant("symmetric", BOOL, True),
            _constant("scale_float", FLOAT, 0.125),
            _constant("zero_point", INT64, 0),
            _constant("axis", INT32, -1),
        ],
    )


def _make_specialized_codebook():
    return {
        "scale_float": 0.5,
        "codebook_data": [-1.0, -0.5, 0.5, 1.0],
        "packed_count": 2,
        "packed_bytes": 1,
        "num_codebooks": 1,
        "codebook_size": 4,
        "vector_size": 1,
        "index_bits": 2,
    }


def _make_custom_codebook():
    specialized = _make_specialized_codebook()
    return _structure(
        "CODEBOOK",
        [
            _array("indices", UINT8, specialized["packed_bytes"]),
            _constant("scale_float", FLOAT, specialized["scale_float"]),
            _constant(
                "codebook_data",
                FLOAT,
                specialized["codebook_data"],
                (len(specialized["codebook_data"]),),
            ),
            *[
                _constant(name, INT32, specialized[name])
                for name in (
                    "packed_count",
                    "packed_bytes",
                    "num_codebooks",
                    "codebook_size",
                    "vector_size",
                    "index_bits",
                )
            ],
        ],
    )


def _make_specialized_tiling():
    return {
        "tile_shape": [128],
        "axes": [1],
        "elem_quant": [_make_specialized_linear()],
        "perm": [1, 0],
    }


def _make_custom_tiling():
    specialized = _make_specialized_tiling()
    return {
        "struct_types": [
            _make_custom_linear(),
            _structure(
                "TILED",
                [
                    _array("tiles", {"type_index": 0}, 32),
                    _constant("tile_shape", INT64, specialized["tile_shape"], (1,)),
                    _constant("axes", INT32, specialized["axes"], (1,)),
                    _constant("perm", INT32, specialized["perm"], (2,)),
                ],
            ),
        ]
    }


def _make_specialized_stq1_0(block_count=2):
    return {
        "data_type": FLOAT,
        "structured_block": {
            "block_layout": {
                "block_size": 256,
                "bytes_per_block": 42,
                "fields": [
                    {"role": 6, "bit_offset": 0, "bit_width": 4, "count": 64, "data_type": 0},
                    {"role": 1, "bit_offset": 256, "bit_width": 1, "count": 64, "data_type": 0},
                    {
                        "role": 2,
                        "bit_offset": 320,
                        "bit_width": 16,
                        "count": 1,
                        "data_type": FLOAT16,
                    },
                ],
            },
            "codebook_data": [float(index % 3 - 1) for index in range(128)],
            "codebook_vector_size": 4,
            "index_formula": [{"field": 6, "multiplier": 1}, {"field": 1, "multiplier": 16}],
            "scatter": {"group_size": 64, "vector_size": 4, "stride": 16},
            "block_count": block_count,
        },
    }


def _make_custom_stq1_0(block_count=2):
    specialized = _make_specialized_stq1_0(block_count)
    structured_block = specialized["structured_block"]
    layout = structured_block["block_layout"]
    return {
        "struct_types": [
            _structure(
                "STQ1_0_BLOCK_256",
                [
                    _array("code", UINT4, 64),
                    _array("sign", UINT8, 8),
                    _array("scale", FLOAT16, 1),
                    _constant("field_roles", INT32, [6, 1, 2], (3,)),
                    _constant("sign_bit_width", INT32, 1),
                    _constant("sign_count", INT32, 64),
                    _constant("block_size", INT32, layout["block_size"]),
                    _constant("bytes_per_block", INT32, layout["bytes_per_block"]),
                ],
            ),
            _structure(
                "STQ1_0",
                [
                    _array("blocks", {"type_index": 0}, block_count),
                    _constant("codebook", FLOAT, structured_block["codebook_data"], (32, 4)),
                    _constant(
                        "codebook_vector_size", INT32, structured_block["codebook_vector_size"]
                    ),
                    _constant("index_fields", INT32, [6, 1], (2,)),
                    _constant("index_multipliers", INT32, [1, 16], (2,)),
                    _constant("scatter_group_size", INT32, 64),
                    _constant("scatter_vector_size", INT32, 4),
                    _constant("scatter_stride", INT32, 16),
                    _constant("data_type", INT32, FLOAT),
                ],
            ),
        ]
    }


def _make_quantized_tensor(quantization, raw_data=None, dims=(16, 8)):
    if raw_data is None:
        raw_data = bytes(range(64))
    return {
        "dims": list(dims),
        "raw_data": raw_data,
        "n_bytes": len(raw_data),
        "quantization": quantization,
        "name": "weight",
        "doc_string": "Quantized test weight.",
    }


def _make_struct_value(struct_types, type_index, raw_data=None, logical_dims=(16, 8)):
    struct_types[type_index]["decoder"] = {
        "output": {"elem_type": FLOAT, "dims": list(logical_dims)}
    }
    if raw_data is None:
        raw_data = bytes(range(64))
    return {
        "model": {"struct_types": struct_types},
        "value": {
            "type": type_index,
            "dims": [],
            "raw_data": raw_data,
            "name": "weight",
            "doc_string": "Quantized test weight.",
        },
    }


def _varint(value):
    encoded = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            encoded.append(chunk | 0x80)
        else:
            encoded.append(chunk)
            return bytes(encoded)


def _tag(field_number, wire_type):
    return _varint((field_number << 3) | wire_type)


def _length_delimited(field_number, payload):
    return _tag(field_number, 2) + _varint(len(payload)) + payload


def _serialize_quantized_tensor(tensor):
    """Serializes a QuantizedTensorProto to the onnx-light wire format."""
    chunks = []
    for dim in tensor["dims"]:
        chunks.append(_tag(1, 0) + _varint(dim))
    chunks.append(_length_delimited(2, tensor["raw_data"]))
    chunks.append(_tag(3, 0) + _varint(tensor["n_bytes"]))
    chunks.append(_length_delimited(6, tensor["name"].encode("utf-8")))
    chunks.append(_length_delimited(7, tensor["doc_string"].encode("utf-8")))
    return b"".join(chunks)


def _serialize_struct_value(value):
    """Serializes a StructProto value (custom-type form) to the onnx-light wire format."""
    chunks = [_tag(1, 0) + _varint(value["type"])]
    for dim in value["dims"]:
        chunks.append(_tag(2, 0) + _varint(dim))
    chunks.append(_length_delimited(3, value["raw_data"]))
    chunks.append(_length_delimited(4, value["name"].encode("utf-8")))
    chunks.append(_length_delimited(5, value["doc_string"].encode("utf-8")))
    return b"".join(chunks)


def _serialized_raw_data_size(field_number, raw_data, serialized):
    """Returns the payload byte count of the raw_data field embedded in a serialized message."""
    segment = _length_delimited(field_number, raw_data)
    assert segment in serialized
    return len(segment) - len(_tag(field_number, 2)) - len(_varint(len(raw_data)))


class TestQuantizationCustomTypesSerialization(unittest.TestCase):
    def test_linear_uniform_with_and_without_custom_types(self):
        specialized = _round_trip(_make_specialized_linear())
        custom = _round_trip(_make_custom_linear())

        self.assertEqual(specialized, _constants(custom))
        self.assertEqual(custom["structure"]["fields"][0], _array("values", INT4, 128))

    def test_codebook_with_and_without_custom_types(self):
        specialized = _make_specialized_codebook()
        custom = _make_custom_codebook()

        specialized = _round_trip(specialized)
        custom = _round_trip(custom)

        self.assertEqual(specialized, _constants(custom))
        self.assertEqual(
            custom["structure"]["fields"][0],
            _array("indices", UINT8, specialized["packed_bytes"]),
        )

    def test_tiling_with_and_without_custom_types(self):
        specialized = _make_specialized_tiling()
        custom = _make_custom_tiling()

        specialized = _round_trip(specialized)
        custom = _round_trip(custom)
        tiled = custom["struct_types"][1]

        self.assertEqual(
            {key: specialized[key] for key in ("tile_shape", "axes", "perm")}, _constants(tiled)
        )
        self.assertEqual(specialized["elem_quant"][0], _constants(custom["struct_types"][0]))
        self.assertEqual(tiled["structure"]["fields"][0], _array("tiles", {"type_index": 0}, 32))

    def _assert_quantized_tensor_round_trip(
        self,
        specialized_quantization,
        custom_types,
        custom_type_index,
        custom_type_name,
        raw_data=None,
        logical_dims=(16, 8),
    ):
        quantized_tensor = _round_trip(
            _make_quantized_tensor(specialized_quantization, raw_data, logical_dims)
        )
        struct_value = _round_trip(
            _make_struct_value(custom_types, custom_type_index, raw_data, logical_dims)
        )
        value = struct_value["value"]
        selected_type = struct_value["model"]["struct_types"][value["type"]]

        self.assertEqual(quantized_tensor["raw_data"], value["raw_data"])
        self.assertEqual(quantized_tensor["n_bytes"], len(value["raw_data"]))
        self.assertEqual(quantized_tensor["dims"], selected_type["decoder"]["output"]["dims"])
        self.assertEqual(selected_type["decoder"]["output"]["elem_type"], FLOAT)
        self.assertEqual(quantized_tensor["name"], value["name"])
        self.assertEqual(quantized_tensor["doc_string"], value["doc_string"])
        self.assertEqual(selected_type["name"], custom_type_name)
        return quantized_tensor, struct_value

    def test_quantized_tensor_linear_with_and_without_custom_types(self):
        quantized_tensor, struct_value = self._assert_quantized_tensor_round_trip(
            _make_specialized_linear(), [_make_custom_linear()], 0, "LINEAR"
        )
        self.assertEqual(
            quantized_tensor["quantization"], _constants(struct_value["model"]["struct_types"][0])
        )

    def test_quantized_tensor_codebook_with_and_without_custom_types(self):
        quantized_tensor, struct_value = self._assert_quantized_tensor_round_trip(
            _make_specialized_codebook(), [_make_custom_codebook()], 0, "CODEBOOK"
        )
        self.assertEqual(
            quantized_tensor["quantization"], _constants(struct_value["model"]["struct_types"][0])
        )

    def test_quantized_tensor_tiling_with_and_without_custom_types(self):
        custom = _make_custom_tiling()
        quantized_tensor, struct_value = self._assert_quantized_tensor_round_trip(
            _make_specialized_tiling(), custom["struct_types"], 1, "TILED"
        )
        quantization = quantized_tensor["quantization"]
        model_types = struct_value["model"]["struct_types"]
        self.assertEqual(
            {key: quantization[key] for key in ("tile_shape", "axes", "perm")},
            _constants(model_types[1]),
        )
        self.assertEqual(quantization["elem_quant"][0], _constants(model_types[0]))

    def _assert_stq1_0_equivalent(self, specialized, custom):
        structured_block = specialized["structured_block"]
        layout = structured_block["block_layout"]
        block_type, stq_type = custom["struct_types"]
        physical_fields = block_type["structure"]["fields"][:3]
        block_constants = _constants(block_type)
        stq_constants = _constants(stq_type)

        self.assertEqual(
            physical_fields,
            [_array("code", UINT4, 64), _array("sign", UINT8, 8), _array("scale", FLOAT16, 1)],
        )
        self.assertEqual(block_constants["field_roles"], [6, 1, 2])
        self.assertEqual(block_constants["sign_bit_width"], 1)
        self.assertEqual(block_constants["sign_count"], 64)
        self.assertEqual(block_constants["block_size"], layout["block_size"])
        self.assertEqual(block_constants["bytes_per_block"], layout["bytes_per_block"])
        self.assertEqual(stq_constants["codebook"], structured_block["codebook_data"])
        self.assertEqual(
            stq_constants["codebook_vector_size"], structured_block["codebook_vector_size"]
        )
        self.assertEqual(
            stq_constants["index_fields"],
            [term["field"] for term in structured_block["index_formula"]],
        )
        self.assertEqual(
            stq_constants["index_multipliers"],
            [term["multiplier"] for term in structured_block["index_formula"]],
        )
        self.assertEqual(
            stq_constants["scatter_group_size"], structured_block["scatter"]["group_size"]
        )
        self.assertEqual(
            stq_constants["scatter_vector_size"], structured_block["scatter"]["vector_size"]
        )
        self.assertEqual(stq_constants["scatter_stride"], structured_block["scatter"]["stride"])
        self.assertEqual(stq_constants["data_type"], specialized["data_type"])
        self.assertEqual(
            stq_type["structure"]["fields"][0],
            _array("blocks", {"type_index": 0}, structured_block["block_count"]),
        )

    def test_stq1_0_with_and_without_custom_types(self):
        specialized = _round_trip(_make_specialized_stq1_0())
        custom = _round_trip(_make_custom_stq1_0())

        self._assert_stq1_0_equivalent(specialized, custom)

    def test_quantized_tensor_stq1_0_with_and_without_custom_types(self):
        specialized = _make_specialized_stq1_0()
        custom = _make_custom_stq1_0()
        block_count = specialized["structured_block"]["block_count"]
        bytes_per_block = specialized["structured_block"]["block_layout"]["bytes_per_block"]
        raw_data = bytes(index % 256 for index in range(block_count * bytes_per_block))

        quantized_tensor, struct_value = self._assert_quantized_tensor_round_trip(
            specialized,
            custom["struct_types"],
            1,
            "STQ1_0",
            raw_data=raw_data,
            logical_dims=(block_count * 256,),
        )

        self.assertEqual(quantized_tensor["n_bytes"], block_count * bytes_per_block)
        self._assert_stq1_0_equivalent(
            quantized_tensor["quantization"],
            {"struct_types": struct_value["model"]["struct_types"]},
        )

    def _assert_serialized_size_matches_buffer(
        self, specialized_quantization, custom_types, custom_type_index, raw_data
    ):
        tensor = _round_trip(_make_quantized_tensor(specialized_quantization, raw_data))
        struct_value = _round_trip(_make_struct_value(custom_types, custom_type_index, raw_data))
        value = struct_value["value"]

        serialized_tensor = _serialize_quantized_tensor(tensor)
        serialized_value = _serialize_struct_value(value)

        # The declared byte size matches the buffer that actually gets serialized.
        self.assertEqual(tensor["n_bytes"], len(raw_data))
        # The serialized raw_data segment carries exactly the buffer size, with and
        # without custom types.
        self.assertEqual(
            _serialized_raw_data_size(2, tensor["raw_data"], serialized_tensor), len(raw_data)
        )
        self.assertEqual(
            _serialized_raw_data_size(3, value["raw_data"], serialized_value), len(raw_data)
        )

    def test_quantized_tensor_serialized_size_matches_buffer(self):
        self._assert_serialized_size_matches_buffer(
            _make_specialized_linear(), [_make_custom_linear()], 0, bytes(range(64))
        )
        self._assert_serialized_size_matches_buffer(
            _make_specialized_codebook(),
            [_make_custom_codebook()],
            0,
            bytes(index % 256 for index in range(200)),
        )

    def test_quantized_tensor_stq1_0_serialized_size_matches_buffer(self):
        specialized = _make_specialized_stq1_0()
        custom = _make_custom_stq1_0()
        block_count = specialized["structured_block"]["block_count"]
        bytes_per_block = specialized["structured_block"]["block_layout"]["bytes_per_block"]
        raw_data = bytes(index % 256 for index in range(block_count * bytes_per_block))

        self._assert_serialized_size_matches_buffer(
            specialized, custom["struct_types"], 1, raw_data
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
