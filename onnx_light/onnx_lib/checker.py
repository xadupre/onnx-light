# source: https://github.com/onnx/onnx/blob/main/onnx/checker.py
"""Checker helpers for onnx_light."""

from __future__ import annotations

from ..onnx_py import _onnxpyprotoop as _P, _onnxpyprotolib as _C  # type: ignore

_checker = _C.checker  # type: ignore

#: Raised when a model or proto fails validation.
ValidationError = _checker.ValidationError


def check_model(
    model: _C.ModelProto,  # type: ignore
    *,
    full_check: bool = False,
    skip_opset_compatibility_check: bool = False,
    check_custom_domain: bool = False,
    skip_external_data_location_check: bool = False,
) -> None:
    """Checks a model and raises checker.ValidationError on invalid content.

    Validates the model's IR structure, including topological ordering of nodes,
    SSA form, schema compliance, and metadata consistency.

    When ``skip_external_data_location_check`` is True, the validation of
    external tensor data locations is skipped, bypassing errors such as an empty
    or unsafe location. This is not recommended and must never be the default:
    it disables safety checks that guard against unsafe external data paths.

    Raises:
        ValidationError: If the model fails validation.
    """
    _checker.check_model(
        model,
        full_check=full_check,
        skip_opset_compatibility_check=skip_opset_compatibility_check,
        check_custom_domain=check_custom_domain,
        skip_external_data_location_check=skip_external_data_location_check,
    )


def check_attribute(attribute: _C.AttributeProto) -> None:  # type: ignore
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
    if sum(oneof) != 1:
        raise ValidationError(
            f"The attribute {attribute.name!r} has more than one value: {attribute}"
        )


def check_sparse_tensor(sparse_tensor: _P.SparseTensorProto) -> None:  # type: ignore
    """Checks a sparse tensor and raises checker.ValidationError on invalid content.

    Raises:
        ValidationError: If the sparse tensor is invalid.
    """
    if len(sparse_tensor.dims) != 2:
        raise ValidationError(f"Only 2D sparse tensors are allowed: {tuple(sparse_tensor.dims)}")


def check_graph(graph: _P.GraphProto) -> None:  # type: ignore
    """Checks a graph and raises checker.ValidationError on invalid content.

    Raises:
        ValidationError: If the graph is invalid.
    """
    if not isinstance(graph, _P.GraphProto):  # type: ignore
        raise ValidationError(f"Expected a GraphProto, got {type(graph)}")


def check_function_call_cycles(model: _P.ModelProto) -> None:  # type: ignore
    """Checks for cycles in model-local function call graph.

    Raises:
        ValidationError: If the model contains cyclic function references.
    """
    _checker.check_function_call_cycles(model)
