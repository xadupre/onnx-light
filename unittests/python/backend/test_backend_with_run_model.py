# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Backend tests that exercise the C++-implemented ``RunModel`` dispatcher
(exposed through ``onnx_light.onnx_py._onnxpykernels.runtime``) against every
backend test case whose top-level graph contains a single node of an op
registered in ``KernelDispatchTable``.

This is the Python counterpart of
``unittests/cc/onnx_backend_test/test_backend_run_model.cc``: both walk the
same C++-generated backend test registry and validate that ``RunModel``
reproduces the expected outputs without discrepancies.
"""

from __future__ import annotations

import re
import unittest

import numpy as np

import onnx_light.onnx as onnxl
from onnx_light.backend.test.case import make_test_class, collect_test_case
from onnx_light.onnx import numpy_helper as onh
from onnx_light.onnx_py._onnxpykernels import runtime as rt


# Operators currently registered in
# ``onnx_light/onnx_kernels/run_nodes.cc::KernelDispatchTable``. Backend
# test cases whose top-level graph is a single node of one of these ops are
# the only ones ``RunModel`` can execute today.
_IMPLEMENTED_OPS: frozenset[str] = frozenset({"Abs", "Neg", "Add", "Sub", "Mul", "Div"})


def _default_opset_version(model: onnxl.ModelProto) -> int:
    """Returns the version of the default ai.onnx opset for ``model``."""
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return 18


def run_model_backend(model: onnxl.ModelProto, *inputs: np.ndarray) -> list[np.ndarray]:
    """Executes ``model`` through the C++ ``RunModel`` dispatcher.

    Mirrors the signature expected by :func:`make_test_class` (the same as
    :func:`onnxruntime_backend` in ``test_backend_with_onnxruntime.py``):
    ``rt(model, *inputs)`` and returns the model's outputs as numpy arrays
    in graph-output order.
    """
    ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(_default_opset_version(model))))

    # Wire positional ``inputs`` to the model's declared graph inputs by name.
    graph_inputs = [vi.name for vi in model.graph.input]
    if len(inputs) != len(graph_inputs):
        raise ValueError(
            f"Expected {len(graph_inputs)} positional inputs for graph inputs "
            f"{graph_inputs}, got {len(inputs)}."
        )
    for name, arr in zip(graph_inputs, inputs):
        tp = onh.from_array(np.ascontiguousarray(arr), name=name)
        ctx.set(name, rt.tensor_from_proto(tp))

    rt.run_model(model, ctx)

    outputs: list[np.ndarray] = []
    for vi in model.graph.output:
        t = ctx.get(vi.name)
        dtype_map = {
            int(onnxl.TensorProto.FLOAT): np.float32,
            int(onnxl.TensorProto.DOUBLE): np.float64,
            int(onnxl.TensorProto.INT32): np.int32,
            int(onnxl.TensorProto.INT64): np.int64,
            int(onnxl.TensorProto.UINT8): np.uint8,
            int(onnxl.TensorProto.INT8): np.int8,
            int(onnxl.TensorProto.UINT16): np.uint16,
            int(onnxl.TensorProto.INT16): np.int16,
            int(onnxl.TensorProto.UINT32): np.uint32,
            int(onnxl.TensorProto.UINT64): np.uint64,
            int(onnxl.TensorProto.BOOL): np.bool_,
            int(onnxl.TensorProto.FLOAT16): np.float16,
        }
        dtype = dtype_map.get(int(t.data_type))
        if dtype is None:
            raise NotImplementedError(
                f"run_model_backend cannot convert C++ Tensor data_type={t.data_type} to numpy."
            )
        arr = np.frombuffer(t.raw_data(), dtype=dtype)
        outputs.append(arr.reshape(tuple(int(d) for d in t.shape)))
    return outputs


def _single_node_op_type(tc) -> str | None:
    """Returns the ``op_type`` of the top-level node when ``tc.model.graph``
    contains exactly one node, otherwise ``None``.
    """
    if tc.model is None:
        return None
    nodes = list(tc.model.graph.node)
    if len(nodes) != 1:
        return None
    return nodes[0].op_type


def _build_include_regex() -> list[str]:
    """Returns the include-regex list selecting every backend test case whose
    top-level graph is a single node of one of the implemented operators.

    Building the list once at module load avoids re-walking the registry from
    inside :func:`make_test_class`.
    """
    names: list[str] = []
    for name, tc in collect_test_case().items():
        op = _single_node_op_type(tc)
        if op is not None and op in _IMPLEMENTED_OPS:
            names.append(name)
    if not names:
        # Fallback: keep a regex that matches nothing so ``make_test_class``
        # generates an empty test class instead of every case in the registry.
        return [r"^$"]
    # Anchor each name to avoid accidental substring matches.
    return [r"^" + re.escape(n) + r"$" for n in names]


TestRunModelBackend = make_test_class(
    run_model_backend, include_regex=_build_include_regex()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
