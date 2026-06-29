"""Helpers to accept proto-like objects in onnx_light wrapper APIs."""

from __future__ import annotations

from typing import Any, TypeVar

_ProtoT = TypeVar("_ProtoT")


def coerce_proto(proto: Any, cls: type[_ProtoT]) -> _ProtoT:
    """Returns *proto* as an ``onnx_light`` proto of type *cls*."""

    if isinstance(proto, cls):
        return proto
    if not hasattr(proto, "SerializeToString"):
        raise TypeError(
            f"Expected {cls.__name__} or a proto-like object exposing "
            f"SerializeToString(), got {type(proto).__name__}."
        )

    try:
        serialized = proto.SerializeToString()
    except Exception as exc:
        raise TypeError(f"Unable to serialize {type(proto).__name__} as {cls.__name__}.") from exc

    if not isinstance(serialized, (bytes, bytearray, memoryview)):
        raise TypeError(f"SerializeToString() on {type(proto).__name__} did not return bytes.")

    converted = cls()
    try:
        converted.ParseFromString(bytes(serialized))
    except Exception as exc:
        raise TypeError(f"Unable to parse {type(proto).__name__} as {cls.__name__}.") from exc
    return converted


def matches_proto_class(proto: Any, cls: type[Any]) -> bool:
    """Returns whether *proto* is an instance of *cls* or has the same proto class name."""

    return isinstance(proto, cls) or type(proto).__name__ == cls.__name__


def copy_proto_back(dst: Any, src: Any) -> None:
    """Copies the serialized contents of *src* back into *dst* when possible."""

    if dst is src:
        return
    if not hasattr(dst, "Clear") or not hasattr(dst, "ParseFromString"):
        return
    dst.Clear()
    dst.ParseFromString(src.SerializeToString())
