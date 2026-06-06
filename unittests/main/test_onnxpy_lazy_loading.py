# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the lazy-loading behavior of :mod:`onnx_light.onnx_py._onnxpy`."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(script: str) -> str:
    """Execute ``script`` in a fresh interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_importing_shim_does_not_load_extensions() -> None:
    """Importing the shim must not import any of the compiled extensions."""
    out = _run("""
        import sys
        import onnx_light.onnx_py._onnxpy  # noqa: F401

        loaded = sorted(
            n
            for n in sys.modules
            if n.startswith("onnx_light.onnx_py._onnx")
            and n != "onnx_light.onnx_py._onnxpy"
        )
        print(loaded)
        """)
    assert out.strip() == "[]"


def test_proto_only_access_does_not_load_optim_or_backend() -> None:
    """Looking up a proto attribute imports ``_onnxpyprotoop`` only."""
    out = _run("""
        import sys
        from onnx_light.onnx_py import _onnxpy

        _ = _onnxpy.ModelProto

        loaded = sorted(
            n.rsplit(".", 1)[-1]
            for n in sys.modules
            if n.startswith("onnx_light.onnx_py._onnxpy")
            or n.startswith("onnx_light.onnx_py._onnxbackend")
            or n.startswith("onnx_light.onnx_py._onnxkernels")
        )
        # Only the proto extension (and its submodules) should be loaded.
        assert "_onnxpyprotolib" not in loaded, loaded
        assert "_onnxpyoptim" not in loaded, loaded
        assert "_onnxkernels" not in loaded, loaded
        assert "_onnxbackend" not in loaded, loaded
        assert "_onnxpyprotoop" in loaded, loaded
        print("ok")
        """)
    assert out.strip() == "ok"


def test_shape_inference_merges_proto_and_optim_attributes() -> None:
    """``shape_inference`` exposes attributes from both contributing extensions."""
    out = _run("""
        from onnx_light.onnx_py import _onnxpy

        si = _onnxpy.shape_inference
        # ``infer_shapes`` comes from the proto extension; ``infer_shapes_model``
        # comes from the optim extension. Both must be exposed.
        assert hasattr(si, "infer_shapes")
        assert hasattr(si, "infer_shapes_model")
        print("ok")
        """)
    assert out.strip() == "ok"


def test_unknown_attribute_raises_attribute_error() -> None:
    out = _run("""
        from onnx_light.onnx_py import _onnxpy

        try:
            _onnxpy.this_attribute_does_not_exist
        except AttributeError:
            print("ok")
        """)
    assert out.strip() == "ok"
