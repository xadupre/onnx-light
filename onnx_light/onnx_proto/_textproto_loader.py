"""Loads ONNX models stored in protobuf text format (``.textproto``, etc.).

Text-format protobuf files (``.textproto``, ``pbtxt``, ``prototxt``) are
human-readable representations of the binary protobuf encoding.  onnx-light
itself only has a binary protobuf parser, so this module provides a bridge:

1. It uses ``google.protobuf`` (available as a transitive dependency of
   ``onnxruntime``) together with the pre-compiled ONNX proto descriptor
   bundled as ``_onnx_proto2.desc`` to parse the text-format file into a
   Google-protobuf ``ModelProto`` object.
2. It then serialises that object to binary protobuf and returns the raw bytes
   so the caller can hand them to onnx-light's own binary parser.

The ONNX ``.proto`` file that produced ``_onnx_proto2.desc`` uses *proto2*
syntax (the same syntax as the canonical ``onnx/onnx.proto``).  Proto2 does
**not** use packed encoding for repeated scalar fields by default, which keeps
the binary output compatible with onnx-light's lightweight parser.
"""

from __future__ import annotations

import contextlib
import importlib
import pathlib

_DESCRIPTOR_PATH = pathlib.Path(__file__).with_name("_onnx_proto2.desc")

# Cached state: descriptor bytes and the google-protobuf ModelProto class.
_pb_ModelProto: type | None = None


def _get_pb_model_proto_class() -> type:
    """Returns the google-protobuf ``ModelProto`` class, loading the ONNX
    descriptor on the first call.

    Raises:
        ImportError: When ``google.protobuf`` is not available.
        RuntimeError: When the bundled descriptor cannot be loaded.
    """
    global _pb_ModelProto  # noqa: PLW0603
    if _pb_ModelProto is not None:
        return _pb_ModelProto

    try:
        descriptor_pb2 = importlib.import_module("google.protobuf.descriptor_pb2")
        descriptor_pool_mod = importlib.import_module("google.protobuf.descriptor_pool")
        message_factory_mod = importlib.import_module("google.protobuf.message_factory")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Loading '.textproto' files requires the 'google-protobuf' package. "
            "Install it with 'pip install protobuf' or install 'onnxruntime' which "
            "brings it in as a transitive dependency."
        ) from exc

    descriptor_bytes = _DESCRIPTOR_PATH.read_bytes()
    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(descriptor_bytes)

    pool = descriptor_pool_mod.Default()
    with contextlib.suppress(TypeError):
        pool.Add(fds.file[0])

    msg_classes = message_factory_mod.GetMessages(fds.file, pool=pool)
    pb_class = msg_classes.get("onnx.ModelProto")
    if pb_class is None:
        raise RuntimeError(
            "Failed to find 'onnx.ModelProto' in the bundled ONNX proto descriptor. "
            f"Descriptor file: {_DESCRIPTOR_PATH}"
        )

    _pb_ModelProto = pb_class
    return pb_class


def load_text_proto(path: str | pathlib.Path) -> bytes:
    """Reads an ONNX model stored in protobuf text format and returns the
    equivalent binary protobuf bytes.

    Args:
        path: Path to the ``.textproto`` (or ``.pbtxt`` / ``.prototxt``) file.

    Returns:
        Binary protobuf bytes that can be passed to onnx-light's binary parser.

    Raises:
        ImportError: When ``google.protobuf`` is not available.
        ValueError: When the file cannot be parsed as an ONNX ``ModelProto``.
    """
    try:
        text_format_mod = importlib.import_module("google.protobuf.text_format")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Loading '.textproto' files requires the 'google-protobuf' package. "
            "Install it with 'pip install protobuf' or install 'onnxruntime' which "
            "brings it in as a transitive dependency."
        ) from exc

    pb_ModelProto = _get_pb_model_proto_class()
    text_content = pathlib.Path(path).read_text(encoding="utf-8")

    try:
        proto_obj = text_format_mod.Parse(text_content, pb_ModelProto())
    except Exception as exc:
        raise ValueError(
            f"Failed to parse '{path}' as an ONNX ModelProto in text format: {exc}"
        ) from exc

    return proto_obj.SerializeToString()
