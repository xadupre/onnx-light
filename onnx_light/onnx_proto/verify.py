"""Schema-free structural validation for onnx_proto messages.

Thin Python wrapper around the ``verify`` submodule exposed by the
``_onnxpyprotoop`` C++ extension (see ``onnx_proto/onnx_verify.h``).

Unlike :mod:`onnx_light.onnx_lib.checker`, none of these functions require an
operator-schema registry: they only validate that a protobuf message is
internally consistent on its own terms (required fields set, unique names,
single static assignment (SSA) form, topologically sorted nodes, tensor
payload/dtype consistency, ...). Every function raises ``ValueError`` on the
first violation found.
"""

from __future__ import annotations

from ..onnx_py._onnxpyprotoop import verify as _verify  # type: ignore[attr-defined]

verify_value_info = _verify.verify_value_info
verify_tensor = _verify.verify_tensor
verify_sparse_tensor = _verify.verify_sparse_tensor
verify_attribute = _verify.verify_attribute
verify_node = _verify.verify_node
verify_graph = _verify.verify_graph
verify_function = _verify.verify_function
verify_model = _verify.verify_model

__all__ = [
    "verify_attribute",
    "verify_function",
    "verify_graph",
    "verify_model",
    "verify_node",
    "verify_sparse_tensor",
    "verify_tensor",
    "verify_value_info",
]
