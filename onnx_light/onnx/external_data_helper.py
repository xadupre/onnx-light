"""Compatibility shim for :mod:`onnx.external_data_helper`.

Re-exports the helpers from :mod:`onnx_light.onnx_lib.external_data_helper` and
adds :class:`ExternalDataInfo` which ir-py uses to inspect external-data
metadata on a :class:`~onnx_light.onnx.TensorProto`.
"""

from __future__ import annotations

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


class ExternalDataInfo:
    """Parses external-data metadata from a TensorProto.

    Attributes:
        location: Relative path to the external data file.
        offset: Byte offset into the file, or ``None``.
        length: Number of bytes to read, or ``None``.
    """

    def __init__(self, tensor: TensorProto) -> None:
        self.location: str = ""
        self.offset: int | None = None
        self.length: int | None = None

        for entry in tensor.external_data:
            if entry.key == "location":
                self.location = entry.value
            elif entry.key == "offset":
                self.offset = int(entry.value)
            elif entry.key == "length":
                self.length = int(entry.value)
