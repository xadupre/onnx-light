"""Python API comparison helpers.

These helpers walk the public Python API of two packages (typically
:mod:`onnx` and :mod:`onnx_light.onnx`) and report the discrepancies.
They cover three levels:

* the set of sub-modules exposed by each package,
* the public functions exposed by each sub-module,
* the parameter names of functions that exist on both sides.

Re-exported third-party symbols (``sys``, ``typing``, ``os``,
``NamedTuple``, …) are filtered out by checking ``__module__`` /
``__name__`` against the top-level package, so the diff focuses on the
real API surface.
"""

from __future__ import annotations

import inspect
import pkgutil
import types
from typing import Iterable, NamedTuple

#: Default list of sub-modules to compare when checking
#: :mod:`onnx` versus :mod:`onnx_light.onnx`.
DEFAULT_SUBMODULES: tuple[str, ...] = (
    "helper",
    "numpy_helper",
    "checker",
    "defs",
    "parser",
    "shape_inference",
    "version_converter",
    "compose",
    "utils",
    "inliner",
)


class SignatureDiff(NamedTuple):
    """Difference between two function signatures."""

    name: str
    onnx_params: tuple[str, ...]
    onnxl_params: tuple[str, ...]


def _belongs_to_package(module_name: str | None, package_name: str) -> bool:
    """Returns ``True`` when ``module_name`` is ``package_name`` or one of its sub-modules."""
    if not module_name:
        return False
    return module_name == package_name or module_name.startswith(package_name + ".")


def list_submodules(package: types.ModuleType) -> list[str]:
    """Returns the sorted list of public sub-modules of ``package``.

    Sub-modules are discovered through :func:`pkgutil.iter_modules` when
    available (regular packages) and fall back to attribute inspection
    when ``__path__`` is missing (which is the case for
    :mod:`onnx_light.onnx` because every sub-module is re-exported as a
    plain attribute).  Re-exported third-party modules (``sys``,
    ``typing``, ``os``, …) are filtered out so that only the modules
    that genuinely belong to ``package`` are listed.
    """
    package_name = package.__name__
    top_level = package_name.split(".", 1)[0]
    names: set[str] = set()
    path = getattr(package, "__path__", None)
    if path is not None:
        for info in pkgutil.iter_modules(path):
            if not info.name.startswith("_"):
                names.add(info.name)
    for attr_name in dir(package):
        if attr_name.startswith("_"):
            continue
        value = getattr(package, attr_name, None)
        if not isinstance(value, types.ModuleType):
            continue
        if not _belongs_to_package(getattr(value, "__name__", None), top_level):
            continue
        names.add(attr_name)
    return sorted(names)


def list_public_functions(module: types.ModuleType) -> list[str]:
    """Returns the sorted list of public functions defined in ``module``.

    Functions re-exported from other packages (for example
    :class:`typing.NamedTuple`) are filtered out by checking
    ``__module__`` against the top-level package.
    """
    module_name = module.__name__
    top_level = module_name.split(".", 1)[0]
    result: list[str] = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        value = getattr(module, attr_name, None)
        if not (inspect.isfunction(value) or inspect.isbuiltin(value)):
            continue
        owner = getattr(value, "__module__", None)
        if owner is not None and not _belongs_to_package(owner, top_level):
            continue
        result.append(attr_name)
    return sorted(result)


def _signature_parameters(func: object) -> tuple[str, ...] | None:
    """Returns the parameter names of *func* or ``None`` if not introspectable."""
    try:
        signature = inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return tuple(signature.parameters)


def compare_signatures(
    module_a: types.ModuleType,
    module_b: types.ModuleType,
    common_names: Iterable[str],
) -> list[SignatureDiff]:
    """Returns the signature mismatches for functions common to both modules."""
    diffs: list[SignatureDiff] = []
    for name in sorted(common_names):
        func_a = getattr(module_a, name, None)
        func_b = getattr(module_b, name, None)
        params_a = _signature_parameters(func_a)
        params_b = _signature_parameters(func_b)
        if params_a is None or params_b is None:
            continue
        if params_a != params_b:
            diffs.append(SignatureDiff(name, params_a, params_b))
    return diffs


def _compare_modules(
    module_a: types.ModuleType | None,
    module_b: types.ModuleType | None,
) -> dict[str, list]:
    """Compares the public function surface of two modules."""
    if module_a is None or module_b is None:
        return {
            "missing_in_onnxl": [] if module_b is not None else ["<entire module>"],
            "extra_in_onnxl": [] if module_a is not None else ["<entire module>"],
            "common": [],
            "signature_diffs": [],
        }
    funcs_a = set(list_public_functions(module_a))
    funcs_b = set(list_public_functions(module_b))
    common = funcs_a & funcs_b
    return {
        "missing_in_onnxl": sorted(funcs_a - funcs_b),
        "extra_in_onnxl": sorted(funcs_b - funcs_a),
        "common": sorted(common),
        "signature_diffs": compare_signatures(module_a, module_b, common),
    }


def compare_submodule(
    submodule_name: str,
    package_a: types.ModuleType,
    package_b: types.ModuleType,
) -> dict[str, list]:
    """Compares the public function surface of one sub-module in two packages.

    Returns a dictionary with keys ``missing_in_onnxl``,
    ``extra_in_onnxl``, ``common`` and ``signature_diffs`` (the
    ``onnxl`` suffix is used by convention because the primary use
    case is the comparison of :mod:`onnx` versus
    :mod:`onnx_light.onnx`, but the helper itself is symmetric).
    """
    return _compare_modules(
        getattr(package_a, submodule_name, None),
        getattr(package_b, submodule_name, None),
    )


def compare_top_level_functions(
    package_a: types.ModuleType,
    package_b: types.ModuleType,
) -> dict[str, list]:
    """Compares the top-level public functions of two packages.

    Useful to surface entry points such as :func:`onnx.load` /
    :func:`onnx.save` that are exposed directly on the package and not
    in a sub-module.  Same return contract as :func:`compare_submodule`.
    """
    return _compare_modules(package_a, package_b)


def compare_packages(
    package_a: types.ModuleType,
    package_b: types.ModuleType,
    submodules: Iterable[str] = DEFAULT_SUBMODULES,
) -> dict[str, dict]:
    """Compares two packages at every level.

    Returns a dictionary with three keys:

    * ``submodules`` — ``{"missing_in_onnxl": [...], "extra_in_onnxl": [...]}``
      listing the sub-modules exposed on only one side;
    * ``top_level`` — the report returned by
      :func:`compare_top_level_functions`;
    * ``per_submodule`` — ``{submodule_name: report}`` where ``report``
      is the dictionary returned by :func:`compare_submodule`.
    """
    submods_a = set(list_submodules(package_a))
    submods_b = set(list_submodules(package_b))
    per_submodule = {
        name: compare_submodule(name, package_a, package_b) for name in submodules
    }
    return {
        "submodules": {
            "missing_in_onnxl": sorted(submods_a - submods_b),
            "extra_in_onnxl": sorted(submods_b - submods_a),
            "common": sorted(submods_a & submods_b),
        },
        "top_level": compare_top_level_functions(package_a, package_b),
        "per_submodule": per_submodule,
    }
