# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""In-tree PEP 517 build backend wrapping :mod:`scikit_build_core.build`.

The backend behaves exactly like ``scikit_build_core.build`` for the regular
build. When the ``ONNX_LIGHT_REDUCED`` environment variable is set to a truthy
value it produces the *reduced* distribution instead:

* the operator-kernel runtime and the backend-test registries are skipped by
  forwarding ``-DONNX_LIGHT_BUILD_KERNELS=OFF`` to CMake, and
* the wheel/sdist is published under a different distribution name
  (``onnx-light-reduced``) so it can coexist with the full ``onnx-light``
  package. The importable package stays ``onnx_light`` in both cases, so the
  reduced wheel is installed and imported the same way.

The distribution name is changed by temporarily rewriting the static
``[project].name`` field in ``pyproject.toml`` for the duration of the wrapped
hook (the original file is always restored). This keeps a single source tree
able to emit both the full and the reduced distributions without duplicating
``pyproject.toml``.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from scikit_build_core import build as _scikit_build

__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]

# Environment variable that switches the build to the reduced distribution.
REDUCED_ENV = "ONNX_LIGHT_REDUCED"

# Distribution names. The reduced build ships a distinct distribution name but
# the same importable ``onnx_light`` package.
FULL_DISTRIBUTION_NAME = "onnx-light"
REDUCED_DISTRIBUTION_NAME = "onnx-light-reduced"

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

_TRUTHY = {"1", "on", "true", "yes", "y"}


def is_reduced_build() -> bool:
    """Returns True when the reduced distribution should be built.

    Returns:
        bool: True when ``ONNX_LIGHT_REDUCED`` is set to a truthy value.
    """
    return os.environ.get(REDUCED_ENV, "").strip().lower() in _TRUTHY


def _augment_config_settings(
    config_settings: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Returns config settings forcing ONNX_LIGHT_BUILD_KERNELS=OFF when reduced.

    Args:
        config_settings: The config settings forwarded by the build frontend.

    Returns:
        The config settings, with the CMake define added for the reduced build.
        An explicit ``cmake.define.ONNX_LIGHT_BUILD_KERNELS`` already provided by
        the caller is preserved.
    """
    if not is_reduced_build():
        return config_settings
    augmented = dict(config_settings or {})
    augmented.setdefault("cmake.define.ONNX_LIGHT_BUILD_KERNELS", "OFF")
    return augmented


@contextlib.contextmanager
def _reduced_distribution_name() -> Iterator[None]:
    """Temporarily rewrites ``[project].name`` to the reduced distribution name.

    The original ``pyproject.toml`` is always restored, even when the wrapped
    hook raises.

    Yields:
        None: A context in which ``pyproject.toml`` carries the reduced name.
    """
    if not is_reduced_build():
        yield
        return

    original = _PYPROJECT.read_text(encoding="utf-8")
    needle = f'name = "{FULL_DISTRIBUTION_NAME}"'
    if needle not in original:
        # Name already changed or formatted differently: do not guess, build as-is.
        yield
        return

    patched = original.replace(needle, f'name = "{REDUCED_DISTRIBUTION_NAME}"', 1)
    _PYPROJECT.write_text(patched, encoding="utf-8")
    try:
        yield
    finally:
        _PYPROJECT.write_text(original, encoding="utf-8")


def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """Builds a wheel, applying the reduced distribution settings when requested."""
    with _reduced_distribution_name():
        return _scikit_build.build_wheel(
            wheel_directory, _augment_config_settings(config_settings), metadata_directory
        )


def build_sdist(sdist_directory: str, config_settings: Optional[Dict[str, Any]] = None) -> str:
    """Builds an sdist, applying the reduced distribution settings when requested."""
    with _reduced_distribution_name():
        return _scikit_build.build_sdist(
            sdist_directory, _augment_config_settings(config_settings)
        )


def build_editable(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """Builds an editable wheel, applying the reduced settings when requested."""
    with _reduced_distribution_name():
        return _scikit_build.build_editable(
            wheel_directory, _augment_config_settings(config_settings), metadata_directory
        )


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: Optional[Dict[str, Any]] = None
) -> str:
    """Prepares wheel metadata, applying the reduced settings when requested."""
    with _reduced_distribution_name():
        return _scikit_build.prepare_metadata_for_build_wheel(
            metadata_directory, _augment_config_settings(config_settings)
        )


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: Optional[Dict[str, Any]] = None
) -> str:
    """Prepares editable metadata, applying the reduced settings when requested."""
    with _reduced_distribution_name():
        return _scikit_build.prepare_metadata_for_build_editable(
            metadata_directory, _augment_config_settings(config_settings)
        )


def get_requires_for_build_wheel(config_settings: Optional[Dict[str, Any]] = None) -> List[str]:
    """Returns the build requirements for a wheel build."""
    return _scikit_build.get_requires_for_build_wheel(_augment_config_settings(config_settings))


def get_requires_for_build_sdist(config_settings: Optional[Dict[str, Any]] = None) -> List[str]:
    """Returns the build requirements for an sdist build."""
    return _scikit_build.get_requires_for_build_sdist(_augment_config_settings(config_settings))


def get_requires_for_build_editable(
    config_settings: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Returns the build requirements for an editable build."""
    return _scikit_build.get_requires_for_build_editable(
        _augment_config_settings(config_settings)
    )
