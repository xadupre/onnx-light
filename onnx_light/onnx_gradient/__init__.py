"""Reverse-mode automatic differentiation for ONNX graphs.

This module provides two functions that compute gradient FunctionProtos from
ONNX graph descriptions:

- :func:`gradient_of_nodes` – takes a list of :class:`NodeProto` and metadata
  (inputs, initializers, xs, y, zs) and returns a :class:`FunctionProto`
  encoding the gradient computation.
- :func:`gradient_of_function` – takes an existing :class:`FunctionProto`
  together with xs, y, zs and returns the gradient :class:`FunctionProto`.

The returned FunctionProto has:

* **inputs**: *xs* values, then *zs* values, then ``"dy"`` (the incoming
  gradient of *y*; pass ``ones_like(y)`` for a scalar loss).
* **outputs**: one gradient tensor per element of *xs*, named ``"grad_<x>"``.

Supported forward operators
---------------------------
Conv, MatMul, Gemm, Add, Sub, Mul, Div, Neg, Identity, Relu, Sigmoid, Tanh,
ReduceSum, ReduceMean, Reshape, Transpose.

Example: linear regression gradient
-------------------------------------
::

    from onnx_light.onnx_proto._helper import make_node
    from onnx_light.onnx_gradient import gradient_of_nodes

    nodes = [
        make_node("MatMul", ["X", "W"], ["mm"]),
        make_node("Add", ["mm", "b"], ["y"]),
    ]

    grad_fn = gradient_of_nodes(
        nodes=nodes,
        inputs=["X", "W", "b"],
        initializers=[],
        xs=["W", "b"],
        y="y",
        zs=["X"],
    )
    # grad_fn.input  = ["W", "b", "X", "dy"]
    # grad_fn.output = ["grad_W", "grad_b"]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx_light.onnx_proto._helper import FunctionProto, NodeProto, TensorProto

try:
    from ..onnx_py._onnxpygradient import (  # type: ignore[import]
        gradient_of_function,
        gradient_of_nodes,
    )
except ImportError as exc:
    raise ImportError(
        "The onnx_gradient bindings are only available in the extended build "
        "(ONNX_LIGHT_BUILD_KERNELS=ON / full wheel).  "
        "Install the full onnx-light wheel with: pip install onnx-light"
    ) from exc

__all__ = ["gradient_of_function", "gradient_of_nodes"]
