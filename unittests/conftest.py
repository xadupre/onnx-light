# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration shared by the onnx-light unit tests.

In a *reduced* build (``onnx-light-reduced``, compiled with
``ONNX_LIGHT_BUILD_KERNELS=OFF``) the operator-kernel runtime
(``_onnxpykernels``) and the backend-test registries (``_onnxpybackend``) are
not available. The test modules that exercise those features fail to import and
are skipped automatically so the reduced CI job can run the proto / schema /
optim / manipulation / lib tests without listing every excluded file.

When the kernels are available (the regular build) this file is a no-op.
"""

from __future__ import annotations

from pathlib import Path


def _kernels_available() -> bool:
    """Returns True when the operator-kernel extension is importable."""
    try:
        import onnx_light.onnx_py._onnxpykernels  # noqa: F401
    except ImportError:
        return False
    return True


_KERNELS_AVAILABLE = _kernels_available()

# Import targets that only exist in the full build. A test module referencing
# any of these is skipped in a reduced build.
_REDUCED_ONLY_IMPORTS = (
    "_onnxpykernels",
    "_onnxpybackend",
    "onnx_light._reference",
    "onnx_light.onnx.reference",
    "onnx_light.onnx.backend",
    "onnx_light.onnx_lib.backend",
)


def _needs_kernels(path: Path) -> bool:
    """Returns True when *path* references a full-build-only import."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return any(token in text for token in _REDUCED_ONLY_IMPORTS)


def pytest_ignore_collect(collection_path, config):
    """Skips kernel/backend test modules when running a reduced build."""
    if _KERNELS_AVAILABLE:
        return None
    path = Path(str(collection_path))
    if path.is_file() and path.suffix == ".py" and _needs_kernels(path):
        return True
    return None
