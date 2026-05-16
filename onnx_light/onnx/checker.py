# source: https://github.com/onnx/onnx/blob/main/onnx/checker.py
"""Checker helpers for onnx_light."""

from __future__ import annotations

from .onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

_checker = _C.checker

#: Raised when a model or proto fails validation.
ValidationError = _checker.ValidationError


def check_model(model: _C.ModelProto) -> None:
    """Checks a model and raises checker.ValidationError on invalid content.

    Returns:
        None.
    """
    _checker.check_model(model)


def check_attribute(attribute: _C.AttributeProto) -> None:
    """Checks an attribute and raises checker.ValidationError on invalid content.

    Raises:
        ValidationError: If the attribute is invalid.
    """
    oneof = [
        attribute.has_f(),
        attribute.has_i(),
        attribute.has_s(),
        attribute.has_t(),
        attribute.has_sparse_tensor(),
        attribute.has_floats(),
        attribute.has_ints(),
        attribute.has_strings(),
        attribute.has_tensors(),
        attribute.has_sparse_tensors(),
    ]
    if not any(oneof):
        raise ValidationError(f"The attribute {attribute.name!r} has no value: {attribute}")
    if sum(int(i) for i in oneof) != 1:
        raise ValidationError(
            f"The attribute {attribute.name!r} has more than one value: {attribute}"
        )


def check_sparse_tensor(sparse_tensor: _C.SparseTensorProto) -> None:
    """Checks a sparse tensor and raises checker.ValidationError on invalid content.

    Raises:
        ValidationError: If the sparse tensor is invalid.
    """
    dims = tuple(sparse_tensor.dims)
    if len(dims) != 2:
        raise ValidationError(f"Only 2D sparse tensors are allowed: {dims}")


def check_graph(graph: _C.GraphProto) -> None:
    """Checks a graph and raises checker.ValidationError on invalid content.

    Raises:
        ValidationError: If the graph is invalid.
    """
    if not isinstance(graph, _C.GraphProto):
        raise ValidationError(f"Expected a GraphProto, got {type(graph)}")


def check_function_call_cycles(model: _C.ModelProto) -> None:
    """Checks for cycles in model-local function call graph.

    Raises:
        ValidationError: If the model contains cyclic function references.
    """
    _checker.check_function_call_cycles(model)
