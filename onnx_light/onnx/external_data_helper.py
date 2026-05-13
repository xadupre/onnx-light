# source: https://github.com/onnx/onnx/blob/main/onnx/external_data_helper.py
from __future__ import annotations

import os

from . import TensorProto

_ALLOWED_EXTERNAL_DATA_KEYS = frozenset({"location", "offset", "length", "checksum", "basepath"})


class ExternalDataInfo:
    """Parses the external_data entries of a TensorProto into typed attributes."""

    def __init__(self, tensor: TensorProto) -> None:
        self.location = ""
        self.offset: int | None = None
        self.length: int | None = None
        self.checksum: str | None = None
        self.basepath = ""

        for entry in tensor.external_data:
            key = str(entry.key)
            value = str(entry.value)
            if key in _ALLOWED_EXTERNAL_DATA_KEYS:
                setattr(self, key, value)

        if self.offset is not None:
            self.offset = int(self.offset)
            if self.offset < 0:
                raise ValueError(
                    f"External data offset must be non-negative, got {self.offset} "
                    f"for tensor {tensor.name!r}"
                )

        if self.length is not None:
            self.length = int(self.length)
            if self.length < 0:
                raise ValueError(
                    f"External data length must be non-negative, got {self.length} "
                    f"for tensor {tensor.name!r}"
                )


def uses_external_data(tensor: TensorProto) -> bool:
    """Returns True if the tensor stores its data in an external file.

    Args:
        tensor: a TensorProto object.

    Returns:
        True if the tensor uses external data storage.
    """
    return int(tensor.data_location) == int(TensorProto.EXTERNAL)


def load_external_data_for_tensor(tensor: TensorProto, base_dir: str) -> None:
    """Loads data from an external file into tensor.raw_data.

    Args:
        tensor: a TensorProto object whose external_data field describes the file.
        base_dir: directory that contains the external data file.
    """
    info = ExternalDataInfo(tensor)
    data_path = os.path.join(base_dir, info.location)
    with open(data_path, "rb") as data_file:
        file_size = os.fstat(data_file.fileno()).st_size

        if info.offset is not None:
            if info.offset > file_size:
                raise ValueError(
                    f"External data offset ({info.offset}) exceeds file size "
                    f"({file_size}) for tensor {tensor.name!r}"
                )
            data_file.seek(info.offset)

        if info.length is not None:
            read_start = info.offset if info.offset is not None else 0
            available = file_size - read_start
            if info.length > available:
                raise ValueError(
                    f"External data length ({info.length}) exceeds available data "
                    f"({available} bytes from offset {read_start}) "
                    f"for tensor {tensor.name!r}"
                )
            tensor.raw_data = data_file.read(info.length)
        else:
            tensor.raw_data = data_file.read()
