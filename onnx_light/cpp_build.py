"""Paths needed to link extensions against the loaded onnx-light runtime."""

import importlib.util
from pathlib import Path


def _extension_directory() -> Path:
    """Returns the directory holding the compiled extensions and runtime libraries."""
    spec = importlib.util.find_spec("onnx_light.onnx_py._onnxpyprotoop")
    if spec is None or spec.origin is None:
        raise ImportError(
            "The compiled extension onnx_light.onnx_py._onnxpyprotoop is missing. "
            "Build onnx-light before linking another extension against it."
        )
    return Path(spec.origin).resolve().parent


def _find_runtime_library(library_dir: Path, name: str) -> Path | None:
    """Returns the shared library *name* in *library_dir*, None when absent."""
    for pattern in (f"lib{name}.so", f"lib{name}.dylib", f"{name}.dll"):
        candidates = sorted(library_dir.glob(pattern))
        if candidates:
            return candidates[0].resolve()
    return None


def _find_import_library(source_root: Path, name: str) -> Path | None:
    """Returns the Windows import library *name* in the build tree, None when absent."""
    for pattern in (f"lib/**/{name}", f"build/**/{name}"):
        candidates = sorted(source_root.glob(pattern))
        if candidates:
            return candidates[0].resolve()
    return None


def get_cpp_build_info() -> dict[str, str]:
    """Returns the headers and shared libraries used by the Python runtime.

    The libraries are looked up next to the loaded ``_onnxpyprotoop``
    extension, not in the source tree, so the result is correct for editable
    installs as well. Linking an extension to another build or installed copy
    of ``lib_onnx_core`` would create a second process-wide kernel registry.

    Returns:
        A dictionary with ``include_dir``, ``library_dir`` and one
        ``<component>_library`` entry per runtime library built as a shared
        library. Windows and macOS link every runtime library but
        ``lib_onnx_proto`` statically into each Python extension, so
        ``core_library`` is only reported on Linux. On Windows, the matching
        ``<component>_import_library`` is added when the build tree is
        available.
    """
    package_dir = Path(__file__).resolve().parent
    library_dir = _extension_directory()
    info = {"include_dir": str(package_dir), "library_dir": str(library_dir)}
    for component in ("core", "proto"):
        name = f"lib_onnx_{component}"
        library = _find_runtime_library(library_dir, name)
        if library is None:
            continue
        info[f"{component}_library"] = str(library)
        if library.suffix == ".dll":
            import_library = _find_import_library(package_dir.parent, f"{name}.lib")
            if import_library is not None:
                info[f"{component}_import_library"] = str(import_library)
    return info
