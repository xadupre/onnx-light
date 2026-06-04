# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx_light.onnx_optim.shape_inference.

Exercises onnx-light's own C++-backed ``infer_shapes_model`` pipeline,
distinct from ``onnx_light.onnx.shape_inference.infer_shapes`` (which
mirrors upstream ONNX). Both paths share the structured-vs-raw toggle
pattern from :mod:`fuzz_shape_inference` so the harness reaches the
per-node inference dispatch even when random bytes would not parse.
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import onnx_light.onnx as onnx
    from onnx_light.onnx import TensorProto, helper
    from onnx_light.onnx_optim import shape_inference as optim_shape_inference

# Cap Python recursion so a single deeply-nested input cannot kill the
# fuzzer on every iteration, mirroring fuzz_shape_inference.py.
sys.setrecursionlimit(1000)

_UNARY = (
    "Relu",
    "Sigmoid",
    "Tanh",
    "Abs",
    "Neg",
    "Exp",
    "Log",
    "Sqrt",
    "Identity",
    "Floor",
    "Ceil",
)

_SUBGRAPH_OPS = ("If", "Loop", "Scan")


def _const_bool(name, value=True):
    tensor = helper.make_tensor(name, TensorProto.BOOL, [], [value])
    return helper.make_node("Constant", [], [name], value=tensor)


def _build_branch(fdp, depth, max_depth):
    nodes = []
    start = f"s_{depth}"
    start_tensor = helper.make_tensor(start, TensorProto.FLOAT, [1], [0.0])
    nodes.append(helper.make_node("Constant", [], [start], value=start_tensor))

    if depth < max_depth and fdp.ConsumeBool():
        sub_op = _SUBGRAPH_OPS[fdp.ConsumeIntInRange(0, len(_SUBGRAPH_OPS) - 1)]
        out = f"sub_{depth}"
        body = _build_branch(fdp, depth + 1, max_depth)
        if sub_op == "If":
            cond = f"c_{depth}"
            nodes.append(_const_bool(cond))
            else_body = _build_branch(fdp, depth + 1, max_depth)
            nodes.append(
                helper.make_node(
                    "If", [cond], [out], then_branch=body, else_branch=else_body
                )
            )
        elif sub_op == "Loop":
            trip = f"M_{depth}"
            trip_t = helper.make_tensor(trip, TensorProto.INT64, [], [1])
            nodes.append(helper.make_node("Constant", [], [trip], value=trip_t))
            cond = f"c_{depth}"
            nodes.append(_const_bool(cond))
            nodes.append(helper.make_node("Loop", [trip, cond], [out], body=body))
        else:  # Scan
            nodes.append(
                helper.make_node(
                    "Scan", [start], [out], body=body, num_scan_inputs=1
                )
            )
        last = out
    else:
        last = start
        n_ops = fdp.ConsumeIntInRange(0, 4)
        for i in range(n_ops):
            op = _UNARY[fdp.ConsumeIntInRange(0, len(_UNARY) - 1)]
            nxt = f"v_{depth}_{i}"
            nodes.append(helper.make_node(op, [last], [nxt]))
            last = nxt

    return helper.make_graph(
        nodes,
        f"branch_{depth}",
        inputs=[],
        outputs=[helper.make_tensor_value_info(last, TensorProto.FLOAT, None)],
    )


def _build_model(fdp):
    max_depth = fdp.ConsumeIntInRange(0, 80)
    graph = _build_branch(fdp, depth=0, max_depth=max_depth)
    opset = fdp.ConsumeIntInRange(7, 20)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def TestOneInput(data):
    if len(data) < 2:
        return
    toggles = data[-1]
    use_structured = bool(toggles & 0x04)

    try:
        if use_structured:
            fdp = atheris.FuzzedDataProvider(data[:-1])
            model = _build_model(fdp)
        else:
            model = onnx.load(data)
        optim_shape_inference.infer_shapes_model(model)
    except RecursionError:
        return
    except Exception:
        # Malformed fuzz inputs raise expected exceptions (ValueError,
        # InferenceError, parser errors, ...). Real bugs surface as
        # crashes, hangs, or sanitizer reports.
        return


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
