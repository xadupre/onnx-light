"""Pure-Python textproto (protobuf text format) parser and serializer.

This module implements a self-contained reader and writer for the protobuf
*text format* (``.textproto``) on top of the onnx-light proto objects. It does
not depend on the ``protobuf`` runtime; instead it drives the serialization and
parsing from a static description of the ONNX message schema (field name, type
and cardinality) that mirrors ``onnx.proto``.

The two entry points are :func:`serialize_to_textproto`, which converts a proto
object into its textual representation, and :func:`parse_from_textproto`, which
populates a proto object from such a representation. Both are used by
:mod:`onnx_light.onnx_proto._io_helper` to extend ``load``/``save`` with the
``"textproto"`` format.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

from ..onnx_py._onnxpyprotoop import (  # type: ignore
    AttributeProto,
    DeviceConfigurationProto,
    FunctionProto,
    GraphProto,
    IntIntListEntryProto,
    MapProto,
    ModelProto,
    NodeDeviceConfigurationProto,
    NodeProto,
    OperatorSetIdProto,
    OptionalProto,
    SequenceProto,
    ShardedDimProto,
    ShardingSpecProto,
    SimpleShardedDimProto,
    SparseTensorProto,
    StringStringEntryProto,
    TensorAnnotation,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
)

# Field kinds. Scalar kinds are plain strings; message and enum kinds are
# represented as ``("msg", cls)`` and ``("enum", cls)`` tuples.
_INT = "int"
_FLOAT = "float"
_BOOL = "bool"
_STR = "str"
_BYTES = "bytes"


def _msg(cls: type) -> tuple[str, type]:
    """Returns the field descriptor for a singular/repeated message field."""
    return ("msg", cls)


def _enum(cls: type) -> tuple[str, type]:
    """Returns the field descriptor for an enum field."""
    return ("enum", cls)


# Schema: message class -> list of (field_name, kind, repeated).
# The field order matches the proto field-number order used by ``onnx.proto``.
_SCHEMA: Optional[dict[type, list[tuple[str, Any, bool]]]] = None


def _build_schema() -> dict[type, list[tuple[str, Any, bool]]]:
    """Builds and caches the ONNX message schema used for (de)serialization."""
    global _SCHEMA
    if _SCHEMA is not None:
        return _SCHEMA

    string_entry = StringStringEntryProto
    schema: dict[type, list[tuple[str, Any, bool]]] = {
        StringStringEntryProto: [("key", _STR, False), ("value", _STR, False)],
        IntIntListEntryProto: [("key", _INT, False), ("value", _INT, True)],
        TensorAnnotation: [
            ("tensor_name", _STR, False),
            ("quant_parameter_tensor_names", _msg(string_entry), True),
        ],
        DeviceConfigurationProto: [
            ("name", _STR, False),
            ("num_devices", _INT, False),
            ("device", _STR, True),
        ],
        SimpleShardedDimProto: [
            ("dim_value", _INT, False),
            ("dim_param", _STR, False),
            ("num_shards", _INT, False),
        ],
        ShardedDimProto: [
            ("axis", _INT, False),
            ("simple_sharding", _msg(SimpleShardedDimProto), True),
        ],
        ShardingSpecProto: [
            ("tensor_name", _STR, False),
            ("device", _INT, True),
            ("index_to_device_group_map", _msg(IntIntListEntryProto), True),
            ("sharded_dim", _msg(ShardedDimProto), True),
        ],
        NodeDeviceConfigurationProto: [
            ("configuration_id", _STR, False),
            ("sharding_spec", _msg(ShardingSpecProto), True),
            ("pipeline_stage", _INT, False),
        ],
        OperatorSetIdProto: [("domain", _STR, False), ("version", _INT, False)],
        TensorShapeProto.Dimension: [
            ("dim_value", _INT, False),
            ("dim_param", _STR, False),
            ("denotation", _STR, False),
        ],
        TensorShapeProto: [("dim", _msg(TensorShapeProto.Dimension), True)],
        TensorProto: [
            ("dims", _INT, True),
            ("data_type", _INT, False),
            ("float_data", _FLOAT, True),
            ("int32_data", _INT, True),
            ("string_data", _BYTES, True),
            ("int64_data", _INT, True),
            ("name", _STR, False),
            ("raw_data", _BYTES, False),
            ("double_data", _FLOAT, True),
            ("uint64_data", _INT, True),
            ("doc_string", _STR, False),
            ("external_data", _msg(string_entry), True),
            ("data_location", _enum(TensorProto.DataLocation), False),
            ("metadata_props", _msg(string_entry), True),
        ],
        SparseTensorProto: [
            ("values", _msg(TensorProto), False),
            ("indices", _msg(TensorProto), False),
            ("dims", _INT, True),
        ],
        TypeProto.Tensor: [("elem_type", _INT, False), ("shape", _msg(TensorShapeProto), False)],
        TypeProto.Sequence: [("elem_type", _msg(TypeProto), False)],
        TypeProto.Map: [("key_type", _INT, False), ("value_type", _msg(TypeProto), False)],
        TypeProto.Optional: [("elem_type", _msg(TypeProto), False)],
        TypeProto.SparseTensor: [
            ("elem_type", _INT, False),
            ("shape", _msg(TensorShapeProto), False),
        ],
        TypeProto: [
            ("tensor_type", _msg(TypeProto.Tensor), False),
            ("sequence_type", _msg(TypeProto.Sequence), False),
            ("map_type", _msg(TypeProto.Map), False),
            ("denotation", _STR, False),
            ("sparse_tensor_type", _msg(TypeProto.SparseTensor), False),
            ("optional_type", _msg(TypeProto.Optional), False),
        ],
        ValueInfoProto: [
            ("name", _STR, False),
            ("type", _msg(TypeProto), False),
            ("doc_string", _STR, False),
            ("metadata_props", _msg(string_entry), True),
        ],
        AttributeProto: [
            ("name", _STR, False),
            ("ref_attr_name", _STR, False),
            ("doc_string", _STR, False),
            ("type", _enum(AttributeProto.AttributeType), False),
            ("f", _FLOAT, False),
            ("i", _INT, False),
            ("s", _BYTES, False),
            ("t", _msg(TensorProto), False),
            ("g", _msg(GraphProto), False),
            ("sparse_tensor", _msg(SparseTensorProto), False),
            ("tp", _msg(TypeProto), False),
            ("floats", _FLOAT, True),
            ("ints", _INT, True),
            ("strings", _BYTES, True),
            ("tensors", _msg(TensorProto), True),
            ("graphs", _msg(GraphProto), True),
            ("sparse_tensors", _msg(SparseTensorProto), True),
            ("type_protos", _msg(TypeProto), True),
        ],
        NodeProto: [
            ("input", _STR, True),
            ("output", _STR, True),
            ("name", _STR, False),
            ("op_type", _STR, False),
            ("attribute", _msg(AttributeProto), True),
            ("doc_string", _STR, False),
            ("domain", _STR, False),
            ("overload", _STR, False),
            ("metadata_props", _msg(string_entry), True),
            ("device_configurations", _msg(NodeDeviceConfigurationProto), True),
        ],
        GraphProto: [
            ("node", _msg(NodeProto), True),
            ("name", _STR, False),
            ("initializer", _msg(TensorProto), True),
            ("doc_string", _STR, False),
            ("input", _msg(ValueInfoProto), True),
            ("output", _msg(ValueInfoProto), True),
            ("value_info", _msg(ValueInfoProto), True),
            ("quantization_annotation", _msg(TensorAnnotation), True),
            ("sparse_initializer", _msg(SparseTensorProto), True),
            ("metadata_props", _msg(string_entry), True),
        ],
        FunctionProto: [
            ("name", _STR, False),
            ("input", _STR, True),
            ("output", _STR, True),
            ("attribute", _STR, True),
            ("node", _msg(NodeProto), True),
            ("doc_string", _STR, False),
            ("opset_import", _msg(OperatorSetIdProto), True),
            ("domain", _STR, False),
            ("attribute_proto", _msg(AttributeProto), True),
            ("value_info", _msg(ValueInfoProto), True),
            ("overload", _STR, False),
            ("metadata_props", _msg(string_entry), True),
        ],
        ModelProto: [
            ("ir_version", _INT, False),
            ("producer_name", _STR, False),
            ("producer_version", _STR, False),
            ("domain", _STR, False),
            ("model_version", _INT, False),
            ("doc_string", _STR, False),
            ("graph", _msg(GraphProto), False),
            ("opset_import", _msg(OperatorSetIdProto), True),
            ("metadata_props", _msg(string_entry), True),
            ("functions", _msg(FunctionProto), True),
            ("configuration", _msg(DeviceConfigurationProto), True),
        ],
        SequenceProto: [
            ("name", _STR, False),
            ("elem_type", _INT, False),
            ("tensor_values", _msg(TensorProto), True),
            ("sparse_tensor_values", _msg(SparseTensorProto), True),
            ("sequence_values", _msg(SequenceProto), True),
            ("map_values", _msg(MapProto), True),
            ("optional_values", _msg(OptionalProto), True),
        ],
        MapProto: [
            ("name", _STR, False),
            ("key_type", _INT, False),
            ("keys", _INT, True),
            ("string_keys", _BYTES, True),
            ("values", _msg(SequenceProto), False),
        ],
        OptionalProto: [
            ("name", _STR, False),
            ("elem_type", _INT, False),
            ("tensor_value", _msg(TensorProto), False),
            ("sparse_tensor_value", _msg(SparseTensorProto), False),
            ("sequence_value", _msg(SequenceProto), False),
            ("map_value", _msg(MapProto), False),
            ("optional_value", _msg(OptionalProto), False),
        ],
    }
    # Index fields by name for fast lookup during parsing.
    _SCHEMA = schema
    return schema


def _field_map(cls: type) -> dict[str, tuple[str, Any, bool]]:
    """Returns a mapping ``field_name -> (name, kind, repeated)`` for *cls*."""
    schema = _build_schema()
    fields = schema.get(cls)
    if fields is None:
        raise TypeError(f"No textproto schema is registered for type {cls!r}.")
    return {name: (name, kind, repeated) for name, kind, repeated in fields}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _format_float(value: float) -> str:
    """Formats a floating-point value the way the protobuf text format does."""
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def _escape_bytes(data: bytes) -> str:
    """Returns a double-quoted, C-escaped representation of *data*."""
    out = ['"']
    for byte in data:
        if byte == 0x0A:
            out.append("\\n")
        elif byte == 0x0D:
            out.append("\\r")
        elif byte == 0x09:
            out.append("\\t")
        elif byte == 0x5C:
            out.append("\\\\")
        elif byte == 0x27:
            out.append("\\'")
        elif byte == 0x22:
            out.append('\\"')
        elif 0x20 <= byte < 0x7F:
            out.append(chr(byte))
        else:
            out.append("\\%03o" % byte)
    out.append('"')
    return "".join(out)


def _to_bytes(value: Any) -> bytes:
    """Returns the raw bytes of a string/bytes field value.

    Repeated string and bytes fields expose their elements as a binding-level
    ``String`` object rather than a Python ``str``/``bytes``; ``decode('latin-1')``
    recovers the original bytes one-to-one.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return value.decode("latin-1").encode("latin-1")


def _format_scalar(kind: Any, value: Any) -> str:
    """Formats a single scalar *value* of the given *kind* as textproto."""
    if kind in (_STR, _BYTES):
        return _escape_bytes(_to_bytes(value))
    if kind == _INT:
        return str(int(value))
    if kind == _FLOAT:
        return _format_float(value)
    if kind == _BOOL:
        return "true" if value else "false"
    if isinstance(kind, tuple) and kind[0] == "enum":
        enum_cls = kind[1]
        int_value = int(value)
        try:
            return enum_cls(int_value).name
        except ValueError:
            return str(int_value)
    raise TypeError(f"Unsupported scalar kind {kind!r}.")


def _scalar_is_set(obj: Any, name: str, kind: Any, value: Any) -> bool:
    """Returns True when a singular scalar field should be serialized."""
    has = getattr(obj, "has_" + name, None)
    if callable(has):
        return bool(has())
    if kind in (_STR, _BYTES):
        return len(value) > 0
    if kind == _FLOAT:
        return float(value) != 0.0
    if kind == _BOOL:
        return bool(value)
    # ints and enums.
    return int(value) != 0


def _write_message(obj: Any, indent: int, lines: list[str]) -> None:
    """Appends the textproto lines of *obj* (without a header) to *lines*."""
    schema = _build_schema()
    pad = "  " * indent
    for name, kind, repeated in schema[type(obj)]:
        is_message = isinstance(kind, tuple) and kind[0] == "msg"
        if repeated:
            container = getattr(obj, name)
            count = len(container)
            if count == 0:
                continue
            for i in range(count):
                item = container[i]
                if is_message:
                    lines.append(f"{pad}{name} {{")
                    _write_message(item, indent + 1, lines)
                    lines.append(f"{pad}}}")
                else:
                    lines.append(f"{pad}{name}: {_format_scalar(kind, item)}")
        elif is_message:
            has = getattr(obj, "has_" + name, None)
            if callable(has) and not has():
                continue
            child = getattr(obj, name)
            lines.append(f"{pad}{name} {{")
            _write_message(child, indent + 1, lines)
            lines.append(f"{pad}}}")
        else:
            value = getattr(obj, name)
            if not _scalar_is_set(obj, name, kind, value):
                continue
            lines.append(f"{pad}{name}: {_format_scalar(kind, value)}")


def serialize_to_textproto(proto: Any) -> str:
    """Serializes a proto object to its protobuf text-format representation.

    Args:
        proto: A proto object whose type is part of the ONNX schema (for
            example ``ModelProto``, ``GraphProto`` or ``TensorProto``).

    Returns:
        The textproto representation, terminated by a trailing newline.

    Raises:
        TypeError: If *proto* is not a known ONNX message type.
    """
    schema = _build_schema()
    if type(proto) not in schema:
        raise TypeError(f"Cannot serialize {type(proto)!r} to textproto.")
    lines: list[str] = []
    _write_message(proto, 0, lines)
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<comment>\#[^\n]*)
    | (?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
    | (?P<symbol>[:{}\[\],;<>])
    | (?P<number>[-+]?(?:0[xX][0-9a-fA-F]+|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?))
    | (?P<name>[A-Za-z_][A-Za-z0-9_.]*)
    """,
    re.VERBOSE,
)

_SIMPLE_ESCAPES = {
    ord("a"): 0x07,
    ord("b"): 0x08,
    ord("f"): 0x0C,
    ord("n"): 0x0A,
    ord("r"): 0x0D,
    ord("t"): 0x09,
    ord("v"): 0x0B,
    ord("\\"): 0x5C,
    ord("'"): 0x27,
    ord('"'): 0x22,
    ord("?"): 0x3F,
}


class _Token:
    """A single lexical token of a textproto document."""

    __slots__ = ("kind", "text", "value")

    def __init__(self, kind: str, text: str, value: Any = None) -> None:
        self.kind = kind
        self.text = text
        self.value = value


def _unescape_string(literal: str) -> bytes:
    """Decodes a quoted string *literal* (including quotes) into raw bytes."""
    inner = literal[1:-1]
    out = bytearray()
    i = 0
    length = len(inner)
    while i < length:
        ch = inner[i]
        if ch != "\\":
            out += ch.encode("utf-8")
            i += 1
            continue
        i += 1
        if i >= length:
            out.append(0x5C)
            break
        esc = inner[i]
        code = ord(esc)
        if code in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[code])
            i += 1
        elif esc in ("x", "X"):
            i += 1
            hex_digits = ""
            while i < length and len(hex_digits) < 2 and inner[i] in "0123456789abcdefABCDEF":
                hex_digits += inner[i]
                i += 1
            if not hex_digits:
                raise ValueError(f"Invalid \\x escape in string literal {literal!r}.")
            out.append(int(hex_digits, 16) & 0xFF)
        elif esc in "01234567":
            oct_digits = ""
            while i < length and len(oct_digits) < 3 and inner[i] in "01234567":
                oct_digits += inner[i]
                i += 1
            out.append(int(oct_digits, 8) & 0xFF)
        else:
            out += esc.encode("utf-8")
            i += 1
    return bytes(out)


def _tokenize(text: str) -> list[_Token]:
    """Splits a textproto document into a list of tokens."""
    tokens: list[_Token] = []
    pos = 0
    length = len(text)
    while pos < length:
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise ValueError(f"Unexpected character {text[pos]!r} at position {pos}.")
        pos = match.end()
        kind = match.lastgroup
        assert kind is not None
        if kind in ("ws", "comment"):
            continue
        token_text = match.group()
        if kind == "string":
            tokens.append(_Token("string", token_text, _unescape_string(token_text)))
        else:
            tokens.append(_Token(kind, token_text))
    return tokens


class _Reader:
    """Recursive-descent reader that fills proto objects from tokens."""

    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Optional[_Token]:
        """Returns the current token without consuming it, or None at the end."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> _Token:
        """Consumes and returns the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def parse_message(self, obj: Any, closing: Optional[str]) -> None:
        """Parses fields into *obj* until *closing* (or end of input)."""
        fields = _field_map(type(obj))
        while True:
            token = self._peek()
            if token is None:
                if closing is not None:
                    raise ValueError(f"Expected {closing!r} but reached end of input.")
                return
            if token.kind == "symbol" and token.text in (";", ","):
                self._advance()
                continue
            if closing is not None and token.kind == "symbol" and token.text == closing:
                self._advance()
                return
            if token.kind != "name":
                raise ValueError(f"Expected a field name but found {token.text!r}.")
            self._advance()
            spec = fields.get(token.text)
            self._consume_field(obj, token.text, spec)

    def _consume_field(self, obj: Any, name: str, spec: Optional[tuple[str, Any, bool]]) -> None:
        """Consumes ``: value`` / ``{ ... }`` / ``[ ... ]`` for one field."""
        token = self._peek()
        if token is not None and token.kind == "symbol" and token.text == ":":
            self._advance()
            token = self._peek()
        if token is not None and token.kind == "symbol" and token.text == "[":
            self._advance()
            while True:
                nxt = self._peek()
                if nxt is None:
                    raise ValueError("Unterminated '[' list in textproto input.")
                if nxt.kind == "symbol" and nxt.text == "]":
                    self._advance()
                    break
                self._consume_single(obj, name, spec)
                sep = self._peek()
                if sep is not None and sep.kind == "symbol" and sep.text == ",":
                    self._advance()
        else:
            self._consume_single(obj, name, spec)

    def _consume_single(self, obj: Any, name: str, spec: Optional[tuple[str, Any, bool]]) -> None:
        """Consumes a single message block or scalar value for one field."""
        token = self._peek()
        if token is None:
            raise ValueError(f"Expected a value for field {name!r}.")
        if token.kind == "symbol" and token.text in ("{", "<"):
            closing = "}" if token.text == "{" else ">"
            self._advance()
            if spec is None or not (isinstance(spec[1], tuple) and spec[1][0] == "msg"):
                self._skip_message(closing)
                return
            _, kind, repeated = spec
            child = getattr(obj, name).add() if repeated else getattr(obj, name)
            self.parse_message(child, closing)
            return
        raw = self._read_scalar()
        if spec is None:
            return
        _, kind, repeated = spec
        value = _convert_scalar(kind, raw)
        if repeated:
            getattr(obj, name).append(value)
        else:
            setattr(obj, name, value)

    def _read_scalar(self) -> tuple[str, Any]:
        """Reads a scalar token, concatenating adjacent string literals."""
        token = self._peek()
        if token is None:
            raise ValueError("Expected a scalar value but reached end of input.")
        if token.kind == "string":
            buffer = bytearray()
            while True:
                nxt = self._peek()
                if nxt is None or nxt.kind != "string":
                    break
                buffer += self._advance().value
            return ("bytes", bytes(buffer))
        self._advance()
        return ("token", token.text)

    def _skip_message(self, closing: str) -> None:
        """Skips a balanced message block for an unknown field."""
        depth = 1
        while depth > 0:
            token = self._peek()
            if token is None:
                raise ValueError("Unterminated message block in textproto input.")
            self._advance()
            if token.kind == "symbol" and token.text in ("{", "<"):
                depth += 1
            elif token.kind == "symbol" and token.text in ("}", ">"):
                depth -= 1


def _convert_scalar(kind: Any, raw: tuple[str, Any]) -> Any:
    """Converts a raw scalar token into a Python value for the given *kind*."""
    raw_kind, raw_value = raw
    if kind == _STR:
        if raw_kind != "bytes":
            raise ValueError("Expected a quoted string value.")
        return raw_value.decode("utf-8")
    if kind == _BYTES:
        if raw_kind != "bytes":
            raise ValueError("Expected a quoted bytes value.")
        return raw_value
    if kind == _INT:
        return int(raw_value, 0) if raw_kind == "token" else int.from_bytes(raw_value, "big")
    if kind == _FLOAT:
        text = raw_value if raw_kind == "token" else raw_value.decode("utf-8")
        return _parse_float(text)
    if kind == _BOOL:
        return _parse_bool(raw_value if raw_kind == "token" else raw_value.decode("utf-8"))
    if isinstance(kind, tuple) and kind[0] == "enum":
        return _parse_enum(
            kind[1], raw_value if raw_kind == "token" else raw_value.decode("utf-8")
        )
    raise TypeError(f"Unsupported scalar kind {kind!r}.")


def _parse_float(text: str) -> float:
    """Parses a textproto float literal, including ``inf``/``nan`` forms."""
    lowered = text.lower().rstrip("f")
    if lowered in ("inf", "+inf", "infinity", "+infinity"):
        return math.inf
    if lowered in ("-inf", "-infinity"):
        return -math.inf
    if lowered in ("nan", "+nan", "-nan"):
        return math.nan
    return float(text.rstrip("fF"))


def _parse_bool(text: str) -> bool:
    """Parses a textproto boolean literal."""
    if text in ("true", "True", "t", "1"):
        return True
    if text in ("false", "False", "f", "0"):
        return False
    raise ValueError(f"Invalid boolean literal {text!r}.")


def _parse_enum(enum_cls: Any, text: str) -> int:
    """Parses an enum value given as a name or an integer literal."""
    members = enum_cls.__members__
    if text in members:
        return int(members[text].value)
    try:
        return int(text, 0)
    except ValueError as exc:
        raise ValueError(f"Invalid enum value {text!r} for {enum_cls!r}.") from exc


def parse_from_textproto(text: str, proto: Any) -> Any:
    """Populates *proto* from a protobuf text-format string and returns it.

    Args:
        text: The textproto document to parse.
        proto: A freshly created proto object of a known ONNX message type;
            parsed fields are merged into it.

    Returns:
        The populated *proto* object (returned for convenience).

    Raises:
        TypeError: If *proto* is not a known ONNX message type.
        ValueError: If the input is not valid textproto.
    """
    schema = _build_schema()
    if type(proto) not in schema:
        raise TypeError(f"Cannot parse textproto into {type(proto)!r}.")
    tokens = _tokenize(text)
    _Reader(tokens).parse_message(proto, None)
    return proto
