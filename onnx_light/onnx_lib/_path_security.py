"""Path security utilities for external data handling.

Provides validation against path traversal, symlink escape, and other
filesystem-based attacks relevant to loading untrusted ONNX models.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath, PureWindowsPath

__all__ = ["validate_external_data_path"]


def _is_relative_and_contained(location: str) -> bool:
    """Checks whether *location* is a relative path that does not escape its parent.

    Mirrors the C++ validation in TensorProto::LoadExternalData (onnx.cc):
    the lexically-normalised path must have no root component and its first
    segment must not be ``..``.
    """
    if not location:
        return False
    # Normalise using both POSIX and Windows semantics to cover cross-platform models.
    for PathCls in (PurePosixPath, PureWindowsPath):
        p = PathCls(location)
        if p.is_absolute():
            return False
        parts = list(p.parts)
        # Reject leading ".."
        if parts and parts[0] == "..":
            return False
    # Also use os.path for the current platform
    normed_os = os.path.normpath(location)
    if os.path.isabs(normed_os):
        return False
    if normed_os.startswith(".."):
        return False
    return True


def validate_external_data_path(
    location: str, base_dir: str, *, allow_absolute: bool = False
) -> str:
    """Validates and resolves an external data path safely.

    Ensures that the *location* (typically read from model metadata) does not
    escape *base_dir* via path traversal sequences, absolute paths, or symlink
    indirection.

    Args:
        location: The external data location string from model metadata.
        base_dir: The trusted base directory (e.g. model directory).
        allow_absolute: If ``True``, allow *location* to be an absolute path
            (for explicitly user-supplied locations).  If ``False`` (default),
            reject absolute locations.

    Returns:
        The resolved canonical path to the external data file.

    Raises:
        ValueError: If *location* would escape *base_dir* or is otherwise unsafe.
    """
    if not location:
        raise ValueError("External data location must not be empty.")

    # Step 1: Lexical validation (cheap, no filesystem access)
    if not allow_absolute and not _is_relative_and_contained(location):
        raise ValueError(
            f"External data location {location!r} must be a relative path that "
            f"does not escape the base directory."
        )

    # Step 2: Join with base_dir and resolve to canonical path
    if os.path.isabs(location):
        if not allow_absolute:
            raise ValueError(
                f"External data location {location!r} is an absolute path. "
                f"Set allow_absolute=True to permit this."
            )
        candidate = os.path.realpath(location)
    else:
        candidate = os.path.realpath(os.path.join(base_dir, location))

    # Step 3: Canonical containment check (catches symlink escapes in any
    # parent component, handles normalisation of ".." etc.)
    real_base = os.path.realpath(base_dir)
    # Ensure trailing separator for prefix comparison to avoid matching
    # /base_dir_extra when checking containment in /base_dir.
    if not real_base.endswith(os.sep):
        real_base_prefix = real_base + os.sep
    else:
        real_base_prefix = real_base

    if not (candidate == real_base or candidate.startswith(real_base_prefix)):
        raise ValueError(
            f"External data path resolves to {candidate!r} which is outside "
            f"the trusted base directory {real_base!r}."
        )

    return candidate
