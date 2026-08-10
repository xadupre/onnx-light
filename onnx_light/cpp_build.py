"""Paths needed to link extensions against the loaded onnx-light runtime."""

from pathlib import Path


def _find_runtime_library(library_dir: Path, name: str) -> Path:
    candidates = [
        *library_dir.glob(f"lib{name}.so"),
        *library_dir.glob(f"lib{name}.dylib"),
        *library_dir.glob(f"{name}.dll"),
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find the onnx-light C++ runtime library {name!r} under "
            f"{library_dir}. Build onnx-light inplace before linking another "
            "extension against it."
        )
    return candidates[0].resolve()


def get_cpp_build_info() -> dict[str, str]:
    """Returns the headers and shared library used by the Python runtime.

    The returned library is deliberately the copy next to ``_onnxpykernels``.
    Linking an extension to another build or installed copy of
    ``lib_onnx_core`` would create a second process-wide kernel registry.
    """
    package_dir = Path(__file__).resolve().parent
    library_dir = package_dir / "onnx_py"
    info = {
        "include_dir": str(package_dir),
        "core_library": str(_find_runtime_library(library_dir, "lib_onnx_core")),
        "proto_library": str(_find_runtime_library(library_dir, "lib_onnx_proto")),
    }
    if Path(info["core_library"]).suffix == ".dll":
        source_root = package_dir.parent
        for component in ("core", "proto"):
            name = f"lib_onnx_{component}.lib"
            import_libraries = sorted(source_root.glob(f"lib/**/{name}"))
            if not import_libraries:
                import_libraries = sorted(source_root.glob(f"build/**/{name}"))
            if not import_libraries:
                raise FileNotFoundError(
                    f"Could not find {name}, the import library for the onnx-light "
                    "C++ runtime. Build and install onnx-light inplace before linking "
                    "against it."
                )
            info[f"{component}_import_library"] = str(import_libraries[0].resolve())
    return info
