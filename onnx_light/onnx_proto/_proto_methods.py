"""Convenience methods bolted onto the ONNX proto classes.

The classes ``NodeProto``, ``GraphProto``, ``FunctionProto`` and ``ModelProto``
are defined in C++ and exposed via nanobind, so we cannot edit their source.
This module instead augments them at import time with small, self-contained
helpers that mirror the spirit of ``onnx.helper.make_*`` while operating
directly on an existing proto instance. They make node/graph/model creation
look closer to what users typically write by hand::

    node = graph.add_node("Add", ["a", "b"], ["c"], axis=1)
    graph.add_input("a", TensorProto.FLOAT, [None, 3])
    graph.add_initializer("w", np.zeros((3, 3), dtype=np.float32))
    model.add_opset("ai.onnx.ml", 4)
    model.add_metadata("author", "me")

All methods reuse the existing :mod:`onnx_light.onnx_proto._helper` factories
so the behaviour stays in sync with ``make_node``, ``make_attribute``,
``make_tensor`` and friends.
"""

from typing import Any, Sequence

import numpy as np

from ..onnx_py._onnxpyprotoop import (  # type: ignore
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    StringStringEntryProto,
    TensorProto,
    ValueInfoProto,
)
from . import _helper as helper
from . import _numpy_helper as numpy_helper

# ---------------------------------------------------------------------------
# NodeProto
# ---------------------------------------------------------------------------


def _node_set_attribute(
    self: NodeProto,
    name: str,
    value: Any,
    attr_type: int | None = None,
    doc_string: str | None = None,
) -> AttributeProto:
    """Sets attribute ``name`` of *self* to ``value``.

    If an attribute with the same name already exists it is replaced in place,
    otherwise a new one is appended. The attribute type is inferred from
    ``value`` exactly like :func:`onnx_light.onnx.helper.make_attribute`; pass
    ``attr_type`` to disambiguate (e.g. for empty iterables).

    Returns:
        The :class:`AttributeProto` that was added or updated.
    """
    new_attr = helper.make_attribute(name, value, doc_string=doc_string, attr_type=attr_type)
    for i in range(len(self.attribute)):
        if self.attribute[i].name == name:
            self.attribute[i].CopyFrom(new_attr)
            return self.attribute[i]
    self.attribute.append(new_attr)
    return self.attribute[len(self.attribute) - 1]


NodeProto.set_attribute = _node_set_attribute  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Shared input/output/node helpers for GraphProto and FunctionProto
# ---------------------------------------------------------------------------


def _make_value_info(
    name_or_proto: str | ValueInfoProto,
    elem_type: int | None,
    shape: Sequence[int | str | None] | None,
    doc_string: str | None,
) -> ValueInfoProto:
    if isinstance(name_or_proto, ValueInfoProto):
        if elem_type is not None or shape is not None:
            raise ValueError("elem_type and shape must be None when a ValueInfoProto is passed.")
        return name_or_proto
    if elem_type is None:
        vi = ValueInfoProto()
        vi.name = name_or_proto
        if doc_string:
            vi.doc_string = doc_string
        return vi
    return helper.make_tensor_value_info(
        name_or_proto, elem_type, shape, doc_string=doc_string or ""
    )


def _graph_add_input(
    self: GraphProto,
    name_or_proto: str | ValueInfoProto,
    elem_type: int | None = None,
    shape: Sequence[int | str | None] | None = None,
    doc_string: str | None = None,
) -> ValueInfoProto:
    """Appends a new input to *self* and returns it.

    ``name_or_proto`` may be either an already-built :class:`ValueInfoProto`
    or a plain string. In the latter case ``elem_type`` (and optionally
    ``shape``) describes a tensor input and is forwarded to
    :func:`onnx_light.onnx.helper.make_tensor_value_info`.
    """
    self.input.append(_make_value_info(name_or_proto, elem_type, shape, doc_string))
    return self.input[len(self.input) - 1]


def _graph_add_output(
    self: GraphProto,
    name_or_proto: str | ValueInfoProto,
    elem_type: int | None = None,
    shape: Sequence[int | str | None] | None = None,
    doc_string: str | None = None,
) -> ValueInfoProto:
    """Appends a new output to *self* and returns it. See :meth:`add_input`."""
    self.output.append(_make_value_info(name_or_proto, elem_type, shape, doc_string))
    return self.output[len(self.output) - 1]


def _graph_add_value_info(
    self: GraphProto,
    name_or_proto: str | ValueInfoProto,
    elem_type: int | None = None,
    shape: Sequence[int | str | None] | None = None,
    doc_string: str | None = None,
) -> ValueInfoProto:
    """Appends a new intermediate value_info to *self* and returns it."""
    self.value_info.append(_make_value_info(name_or_proto, elem_type, shape, doc_string))
    return self.value_info[len(self.value_info) - 1]


def _graph_add_initializer(
    self: GraphProto, name_or_proto: str | TensorProto, array: "np.ndarray | None" = None
) -> TensorProto:
    """Appends a new initializer to *self* and returns it.

    Accepts either an already-built :class:`TensorProto` or a ``(name, array)``
    pair, in which case the tensor is built from ``array`` using
    :func:`onnx_light.onnx.numpy_helper.from_array`.
    """
    if isinstance(name_or_proto, TensorProto):
        if array is not None:
            raise ValueError("array must be None when a TensorProto is passed.")
        self.initializer.append(name_or_proto)
    else:
        if array is None:
            raise ValueError("array is required when a name is passed.")
        tensor = numpy_helper.from_array(np.asarray(array), name=name_or_proto)
        self.initializer.append(tensor)
    return self.initializer[len(self.initializer) - 1]


def _add_node_common(
    repeated_node,
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str | None = None,
    doc_string: str | None = None,
    domain: str | None = None,
    overload: str | None = None,
    **attrs: Any,
) -> NodeProto:
    node = helper.make_node(
        op_type,
        inputs,
        outputs,
        name=name,
        doc_string=doc_string,
        domain=domain,
        overload=overload,
        **attrs,
    )
    repeated_node.append(node)
    return repeated_node[len(repeated_node) - 1]


def _graph_add_node(
    self: GraphProto,
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str | None = None,
    doc_string: str | None = None,
    domain: str | None = None,
    overload: str | None = None,
    **attrs: Any,
) -> NodeProto:
    """Builds a :class:`NodeProto` with :func:`make_node` and appends it.

    Any extra keyword arguments are forwarded to ``make_node`` as node
    attributes, matching ``onnx.helper.make_node``'s ``**kwargs`` API.

    Returns:
        The newly appended :class:`NodeProto`.
    """
    return _add_node_common(
        self.node,
        op_type,
        inputs,
        outputs,
        name=name,
        doc_string=doc_string,
        domain=domain,
        overload=overload,
        **attrs,
    )


GraphProto.add_input = _graph_add_input  # type: ignore[attr-defined]
GraphProto.add_output = _graph_add_output  # type: ignore[attr-defined]
GraphProto.add_value_info = _graph_add_value_info  # type: ignore[attr-defined]
GraphProto.add_initializer = _graph_add_initializer  # type: ignore[attr-defined]
GraphProto.add_node = _graph_add_node  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# FunctionProto
# ---------------------------------------------------------------------------


def _function_add_input(self: FunctionProto, name: str) -> str:
    """Appends an input name to *self*."""
    self.input.extend([name])
    return name


def _function_add_output(self: FunctionProto, name: str) -> str:
    """Appends an output name to *self*."""
    self.output.extend([name])
    return name


def _function_add_node(
    self: FunctionProto,
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str | None = None,
    doc_string: str | None = None,
    domain: str | None = None,
    overload: str | None = None,
    **attrs: Any,
) -> NodeProto:
    """Builds a :class:`NodeProto` with :func:`make_node` and appends it."""
    return _add_node_common(
        self.node,
        op_type,
        inputs,
        outputs,
        name=name,
        doc_string=doc_string,
        domain=domain,
        overload=overload,
        **attrs,
    )


def _function_add_opset(self: FunctionProto, domain: str, version: int) -> OperatorSetIdProto:
    """Appends an opset import ``(domain, version)`` to *self*."""
    self.opset_import.append(helper.make_opsetid(domain, version))
    return self.opset_import[len(self.opset_import) - 1]


FunctionProto.add_input = _function_add_input  # type: ignore[attr-defined]
FunctionProto.add_output = _function_add_output  # type: ignore[attr-defined]
FunctionProto.add_node = _function_add_node  # type: ignore[attr-defined]
FunctionProto.add_opset = _function_add_opset  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ModelProto
# ---------------------------------------------------------------------------


def _model_add_function(self: ModelProto, function: FunctionProto) -> FunctionProto:
    """Appends a :class:`FunctionProto` to *self* and returns it."""
    if not isinstance(function, FunctionProto):
        raise TypeError(f"function must be a FunctionProto, got {type(function).__name__}.")
    self.functions.append(function)
    return self.functions[len(self.functions) - 1]


def _model_add_opset(self: ModelProto, domain: str, version: int) -> OperatorSetIdProto:
    """Appends an opset import ``(domain, version)`` to *self* and returns it."""
    self.opset_import.append(helper.make_opsetid(domain, version))
    return self.opset_import[len(self.opset_import) - 1]


def _metadata_add(repeated, key: str, value: str) -> StringStringEntryProto:
    for i in range(len(repeated)):
        if repeated[i].key == key:
            repeated[i].value = value
            return repeated[i]
    entry = StringStringEntryProto()
    entry.key = key
    entry.value = value
    repeated.append(entry)
    return repeated[len(repeated) - 1]


def _model_add_metadata(self: ModelProto, key: str, value: str) -> StringStringEntryProto:
    """Sets metadata property ``key`` to ``value`` on *self*.

    If an entry with the same ``key`` already exists it is updated in place;
    otherwise a new :class:`StringStringEntryProto` is appended. Returns the
    affected entry.
    """
    return _metadata_add(self.metadata_props, key, value)


ModelProto.add_function = _model_add_function  # type: ignore[attr-defined]
ModelProto.add_opset = _model_add_opset  # type: ignore[attr-defined]
ModelProto.add_metadata = _model_add_metadata  # type: ignore[attr-defined]


# The same metadata helper is useful on the other protos that carry
# ``metadata_props``; expose it consistently.
for _cls in (NodeProto, GraphProto, FunctionProto):

    def _add_metadata(self, key: str, value: str, _cls=_cls) -> StringStringEntryProto:
        """Sets metadata property ``key`` to ``value`` on *self*."""
        return _metadata_add(self.metadata_props, key, value)

    _cls.add_metadata = _add_metadata  # type: ignore[attr-defined]

del _cls
