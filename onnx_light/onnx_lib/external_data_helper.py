"""Helpers to convert tensors to/from external data, mirroring the
:mod:`onnx.external_data_helper` API on top of onnx-light's
:class:`~onnx_light.onnx.ModelProto`.

The heavy lifting (model traversal, marking tensors as EXTERNAL, loading
external bytes back into ``raw_data``) is performed in C++ by the
``convert_model_to_external_data`` / ``load_external_data_for_model``
bindings exposed by :mod:`onnx_light.onnx_proto._onnxpy`.  This module is a
thin Python shim providing the upstream-compatible signatures plus the small
``set_external_data`` / ``uses_external_data`` helpers.
"""

from __future__ import annotations

from ..onnx_proto._onnxpy import (  # type: ignore
    convert_model_to_external_data as _convert_model_to_external_data,
    load_external_data_for_model as _load_external_data_for_model,
)
from . import ModelProto, TensorProto

__all__ = [
    "convert_model_to_external_data",
    "load_external_data_for_model",
    "set_external_data",
    "uses_external_data",
]


def uses_external_data(tensor: TensorProto) -> bool:
    """Returns ``True`` if the tensor stores its bytes in an external file."""
    return tensor.has_external_data() or tensor.data_location == TensorProto.EXTERNAL


def set_external_data(
    tensor: TensorProto,
    location: str,
    offset: int | None = None,
    length: int | None = None,
    checksum: str | None = None,
    basepath: str | None = None,
) -> None:
    """Marks *tensor* as EXTERNAL and records the given external-data entries.

    The tensor must currently carry inline ``raw_data``; the bytes are left
    untouched so they can be written out by a subsequent ``save`` call.
    """
    if not tensor.raw_data:
        raise ValueError(
            f"Tensor {tensor.name!r} does not have raw_data. "
            "Cannot set external data for this tensor."
        )
    tensor.external_data.clear()
    tensor.data_location = TensorProto.EXTERNAL
    entries = {
        "location": location,
        "offset": int(offset) if offset is not None else None,
        "length": int(length) if length is not None else None,
        "checksum": checksum,
        "basepath": basepath,
    }
    for key, value in entries.items():
        if value is None:
            continue
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = str(value)


def convert_model_to_external_data(
    model: ModelProto,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
) -> None:
    """Marks every initializer tensor with raw data above ``size_threshold``
    as external data.  The actual bytes are not written; they remain in
    ``raw_data`` and are flushed to disk by the next call to
    :func:`onnx_light.onnx.save`.

    :param model: model to modify in place.
    :param all_tensors_to_one_file: when ``True`` (default), every qualifying
        tensor points at the same external file (``location`` or a generated
        ``<uuid>.data`` name).  When ``False``, each tensor is given its own
        file named after the tensor.
    :param location: relative path of the external data file.  Must be
        relative to the model file.  Ignored when
        ``all_tensors_to_one_file=False``.
    :param size_threshold: only tensors whose ``raw_data`` size is greater
        than or equal to ``size_threshold`` bytes are moved to external
        storage.  Set to ``0`` to externalize every tensor with raw data.
    :param convert_attribute: when ``True``, also externalize tensors stored
        inside node attributes (``AttributeProto.t`` and
        ``AttributeProto.tensors``).
    :raises ValueError: if ``location`` is an absolute path.
    :raises FileExistsError: if ``location`` already exists on disk.
    """
    _convert_model_to_external_data(
        model,
        all_tensors_to_one_file,
        "" if location is None else location,
        size_threshold,
        convert_attribute,
    )


def load_external_data_for_model(model: ModelProto, base_dir: str) -> None:
    """Loads external tensor bytes into *model* in place.

    For every tensor whose data lives in an external file, this function
    reads the bytes from ``base_dir`` into the tensor's ``raw_data`` and
    resets ``data_location`` to ``DEFAULT``, dropping the ``external_data``
    entries.

    :param model: model whose external tensors are loaded in place.
    :param base_dir: directory that contains the external data files
        referenced by the tensors.
    """
    _load_external_data_for_model(model, base_dir)
