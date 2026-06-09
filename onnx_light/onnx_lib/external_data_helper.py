"""Helpers to convert tensors to/from external data, mirroring the
:mod:`onnx.external_data_helper` API on top of onnx-light's
:class:`~onnx_light.onnx.ModelProto`.

Two functions are provided:

* :func:`convert_model_to_external_data` marks tensors whose ``raw_data`` is
  large enough as EXTERNAL and records the target file in their
  ``external_data`` metadata.  Calling :func:`onnx_light.onnx.save` after this
  function writes the raw bytes to the chosen external file.
* :func:`load_external_data_for_model` loads the bytes referenced by each
  tensor's ``external_data`` back into its ``raw_data`` field, removing the
  external reference.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Iterable, Iterator

from . import AttributeProto, FunctionProto, GraphProto, ModelProto, TensorProto

__all__ = [
    "convert_model_to_external_data",
    "load_external_data_for_model",
    "set_external_data",
    "uses_external_data",
]


_FILENAME_PATTERN = re.compile(r'^[^<>:;,?"*|/]+$')


def _is_valid_filename(filename: str) -> bool:
    """Returns ``True`` if *filename* contains no path-separator or otherwise
    illegal characters."""
    return bool(_FILENAME_PATTERN.match(filename))


def _iter_attribute_graphs(attribute: AttributeProto) -> Iterator[GraphProto]:
    if attribute.type == AttributeProto.GRAPH and attribute.has_g():
        yield attribute.g
    elif attribute.type == AttributeProto.GRAPHS:
        for i in range(len(attribute.graphs)):
            yield attribute.graphs[i]


def _iter_initializer_tensors_from_graph(graph: GraphProto) -> Iterator[TensorProto]:
    for i in range(len(graph.initializer)):
        yield graph.initializer[i]
    for i in range(len(graph.node)):
        node = graph.node[i]
        for j in range(len(node.attribute)):
            attribute = node.attribute[j]
            for sub_graph in _iter_attribute_graphs(attribute):
                yield from _iter_initializer_tensors_from_graph(sub_graph)


def _iter_attribute_tensors_from_graph(
    graph_or_function: GraphProto | FunctionProto,
) -> Iterator[TensorProto]:
    for i in range(len(graph_or_function.node)):
        node = graph_or_function.node[i]
        for j in range(len(node.attribute)):
            attribute = node.attribute[j]
            if attribute.has_t():
                yield attribute.t
            for k in range(len(attribute.tensors)):
                yield attribute.tensors[k]
            for sub_graph in _iter_attribute_graphs(attribute):
                yield from _iter_attribute_tensors_from_graph(sub_graph)


def _iter_initializer_tensors(model: ModelProto) -> Iterator[TensorProto]:
    if model.has_graph():
        yield from _iter_initializer_tensors_from_graph(model.graph)


def _iter_attribute_tensors(model: ModelProto) -> Iterator[TensorProto]:
    if model.has_graph():
        yield from _iter_attribute_tensors_from_graph(model.graph)
    if model.has_functions():
        for i in range(len(model.functions)):
            yield from _iter_attribute_tensors_from_graph(model.functions[i])


def _iter_all_tensors(model: ModelProto) -> Iterator[TensorProto]:
    yield from _iter_initializer_tensors(model)
    yield from _iter_attribute_tensors(model)


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
    if convert_attribute:
        tensors: Iterable[TensorProto] = _iter_all_tensors(model)
    else:
        tensors = _iter_initializer_tensors(model)

    if all_tensors_to_one_file:
        file_name = f"{uuid.uuid1()}.data"
        if location:
            if os.path.isabs(location):
                raise ValueError(
                    "location must be a relative path that is relative to the model path."
                )
            if os.path.exists(location):
                raise FileExistsError(f"External data file exists in {location}.")
            file_name = location
        for tensor in tensors:
            if tensor.raw_data and len(tensor.raw_data) >= size_threshold:
                set_external_data(tensor, file_name)
    else:
        for tensor in tensors:
            if tensor.raw_data and len(tensor.raw_data) >= size_threshold:
                tensor_location = tensor.name
                if not tensor_location or not _is_valid_filename(tensor_location):
                    tensor_location = str(uuid.uuid1())
                set_external_data(tensor, tensor_location)


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
    for tensor in _iter_all_tensors(model):
        if not uses_external_data(tensor):
            continue
        tensor.load_external_data(base_dir)
        tensor.data_location = TensorProto.DEFAULT
        tensor.external_data.clear()
