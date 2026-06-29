# source: https://github.com/onnx/onnx/blob/main/onnx/checker.py
"""Checker helpers for onnx_light."""

from __future__ import annotations

from ..onnx_py import _onnxpyprotoop as _P, _onnxpyprotolib as _C  # type: ignore
from ._proto_compat import coerce_proto

_checker = _C.checker  # type: ignore

#: Raised when a model or proto fails validation.
ValidationError = _checker.ValidationError


def check_model(
    model,  # type: ignore
    *,
    full_check: bool = False,
    skip_opset_compatibility_check: bool = False,
    check_custom_domain: bool = False,
) -> None:
    """Checks a model and raises checker.ValidationError on invalid content.

    Validates the model's IR structure, including topological ordering of nodes,
    SSA form, schema compliance, and metadata consistency.

    Raises:
        ValidationError: If the model fails validation.
    """
    model = coerce_proto(model, _P.ModelProto)
    _checker.check_model(
        model,
        full_check=full_check,
        skip_opset_compatibility_check=skip_opset_compatibility_check,
        check_custom_domain=check_custom_domain,
    )


def check_attribute(attribute) -> None:  # type: ignore
    """Checks an attribute and raises checker.ValidationError on invalid content.

    Raises:
        ValidationError: If the attribute is invalid.
    """
    attribute = coerce_proto(attribute, _P.AttributeProto)
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
    if sum(oneof) != 1:
        raise ValidationError(
            f"The attribute {attribute.name!r} has more than one value: {attribute}"
        )


def check_sparse_tensor(sparse_tensor) -> None:  # type: ignore
    """Checks a sparse tensor and raises checker.ValidationError on invalid content.

    Raises:
        ValidationError: If the sparse tensor is invalid.
    """
    sparse_tensor = coerce_proto(sparse_tensor, _P.SparseTensorProto)
    if len(sparse_tensor.dims) != 2:
        raise ValidationError(f"Only 2D sparse tensors are allowed: {tuple(sparse_tensor.dims)}")


def check_graph(graph) -> None:  # type: ignore
    """Checks a graph and raises checker.ValidationError on invalid content.

    Raises:
        ValidationError: If the graph is invalid.
    """
    graph = coerce_proto(graph, _P.GraphProto)
    if not isinstance(graph, _P.GraphProto):  # type: ignore
        raise ValidationError(f"Expected a GraphProto, got {type(graph)}")


def check_function_call_cycles(model) -> None:  # type: ignore
    """Checks for cycles in model-local function call graph.

    Raises:
        ValidationError: If the model contains cyclic function references.
    """
    model = coerce_proto(model, _P.ModelProto)
    _checker.check_function_call_cycles(model)
