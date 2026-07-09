"""Compatibility shim for :mod:`onnx.external_data_helper`.

Re-exports the helpers from :mod:`onnx_light.onnx_lib.external_data_helper` and
adds :class:`ExternalDataInfo` which ir-py uses to inspect external-data
metadata on a :class:`~onnx_light.onnx.TensorProto`.
"""

from __future__ import annotations

import warnings

from ..onnx_lib.external_data_helper import (  # noqa: F401
    convert_model_to_external_data,
    load_external_data_for_model,
    set_external_data,
    uses_external_data,
)
from ..onnx_proto._numpy_helper import (  # noqa: F401
    _load_external_data_for_tensor as load_external_data_for_tensor,
)
from . import TensorProto

# Security: whitelist of spec-defined external-data keys (GHSA-3jf9-582g-jjmq).
# Unknown keys are warned and ignored, preventing attribute injection (CWE-915).
_ALLOWED_EXTERNAL_DATA_KEYS: frozenset[str] = frozenset(
    {"location", "offset", "length", "checksum", "basepath"}
)


class ExternalDataInfo:
    """Parses external-data metadata from a TensorProto.

    Attributes:
        location: Relative path to the external data file.
        offset: Byte offset into the file, or ``None``.
        length: Number of bytes to read, or ``None``.
        checksum: Optional checksum string, or ``None``.
        basepath: Optional base path override, or ``None``.
    """

    def __init__(self, tensor: TensorProto) -> None:
        self.location: str = ""
        self.offset: int | None = None
        self.length: int | None = None
        self.checksum: str | None = None
        self.basepath: str | None = None

        unknown_keys: set[str] = set()
        for entry in tensor.external_data:
            key = entry.key
            if key not in _ALLOWED_EXTERNAL_DATA_KEYS:
                unknown_keys.add(key[:100] + ("..." if len(key) > 100 else ""))
                continue
            if key == "location":
                self.location = entry.value
            elif key == "offset":
                self.offset = int(entry.value)
            elif key == "length":
                self.length = int(entry.value)
            elif key == "checksum":
                self.checksum = entry.value
            elif key == "basepath":
                self.basepath = entry.value

        if unknown_keys:
            warnings.warn(
                f"Ignoring unknown external data key(s) {sorted(unknown_keys)!r} "
                f"for tensor {tensor.name!r}. "
                f"Allowed keys: {sorted(_ALLOWED_EXTERNAL_DATA_KEYS)}",
                stacklevel=2,
            )

        if self.offset is not None and self.offset < 0:
            raise ValueError(
                f"External data offset must be non-negative, got {self.offset} "
                f"for tensor {tensor.name!r}."
            )

        if self.length is not None and self.length < 0:
            raise ValueError(
                f"External data length must be non-negative, got {self.length} "
                f"for tensor {tensor.name!r}."
            )
