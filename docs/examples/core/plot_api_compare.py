"""
.. _l-example-plot-api-compare:

Compares the Python API of ``onnx`` and ``onnx_light.onnx``
============================================================

This example walks the public Python API exposed by the upstream
:mod:`onnx` package and by :mod:`onnx_light.onnx` and reports the
discrepancies between the two.

For every common sub-module (``helper``, ``numpy_helper``,
``checker``, ``defs``, ``parser``, ``shape_inference``,
``version_converter``, ``compose``, ``utils``) the script lists:

* the sub-modules that are exposed by one package but not the other,
* the public functions that are exposed by one package but not the
  other,
* the public functions that exist in both packages but whose
  signatures differ (positional/keyword parameter names).

The goal is to make it easy to spot which parts of the upstream API
are not yet implemented in :mod:`onnx_light` and to document the
intentional differences in parameter naming.

The same helper functions are used by the unit test
``unittests/main/test_plot_api_compare.py``.
"""

from __future__ import annotations

import inspect
import pkgutil
import types
from typing import Iterable, NamedTuple

import onnx
import onnx_light.onnx as onnxl

#####################################
# Configuration
# +++++++++++++
#
# The list of sub-modules to compare.  Only sub-modules that exist on
# the ``onnx_light.onnx`` side are considered, but we additionally
# report the sub-modules exposed by ``onnx`` that are missing from
# ``onnx_light.onnx``.
SUBMODULES = (
    "helper",
    "numpy_helper",
    "checker",
    "defs",
    "parser",
    "shape_inference",
    "version_converter",
    "compose",
    "utils",
)


#####################################
# Helpers
# +++++++


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
    ``typing``, ``os``, ...) are filtered out so that only the modules
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
    ``typing.NamedTuple``) are filtered out by checking
    ``__module__``.
    """
    module_name = module.__name__
    # ``onnx_light.onnx.helper`` is re-exported from ``onnx_light.onnx_lib.helper``
    # so accept any module belonging to the same top-level package.
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
    """Returns the list of signature mismatches for functions common to both modules."""
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


def compare_submodule(submodule_name: str) -> dict[str, list]:
    """Compares the public function surface of one sub-module.

    Returns a dictionary with keys ``missing_in_onnxl``,
    ``extra_in_onnxl``, ``common`` and ``signature_diffs``.
    """
    module_onnx = getattr(onnx, submodule_name, None)
    module_onnxl = getattr(onnxl, submodule_name, None)
    if module_onnx is None or module_onnxl is None:
        return {
            "missing_in_onnxl": [] if module_onnxl is not None else ["<entire module>"],
            "extra_in_onnxl": [] if module_onnx is not None else ["<entire module>"],
            "common": [],
            "signature_diffs": [],
        }
    funcs_onnx = set(list_public_functions(module_onnx))
    funcs_onnxl = set(list_public_functions(module_onnxl))
    common = funcs_onnx & funcs_onnxl
    return {
        "missing_in_onnxl": sorted(funcs_onnx - funcs_onnxl),
        "extra_in_onnxl": sorted(funcs_onnxl - funcs_onnx),
        "common": sorted(common),
        "signature_diffs": compare_signatures(module_onnx, module_onnxl, common),
    }


#####################################
# Sub-module overview
# +++++++++++++++++++
#
# We first compare the *set* of sub-modules exposed by each package.

submods_onnx = set(list_submodules(onnx))
submods_onnxl = set(list_submodules(onnxl))

print(f"onnx       : {len(submods_onnx)} sub-modules")
print(f"onnx_light : {len(submods_onnxl)} sub-modules")
print()
print("Sub-modules only in onnx:")
for name in sorted(submods_onnx - submods_onnxl):
    print(f"  - {name}")
print()
print("Sub-modules only in onnx_light.onnx:")
for name in sorted(submods_onnxl - submods_onnx):
    print(f"  - {name}")


#####################################
# Function-level comparison
# +++++++++++++++++++++++++
#
# For every sub-module from :data:`SUBMODULES` we display the
# discrepancies.

for submod_name in SUBMODULES:
    report = compare_submodule(submod_name)
    print()
    print(f"=== {submod_name} ===")
    print(f"  common functions          : {len(report['common'])}")
    print(f"  missing in onnx_light.onnx: {len(report['missing_in_onnxl'])}")
    print(f"  extra in onnx_light.onnx  : {len(report['extra_in_onnxl'])}")
    print(f"  signature mismatches      : {len(report['signature_diffs'])}")
    if report["missing_in_onnxl"]:
        print("  - missing in onnx_light.onnx:")
        for name in report["missing_in_onnxl"]:
            print(f"      * {name}")
    if report["extra_in_onnxl"]:
        print("  - extra in onnx_light.onnx:")
        for name in report["extra_in_onnxl"]:
            print(f"      * {name}")
    if report["signature_diffs"]:
        print("  - signature mismatches:")
        for diff in report["signature_diffs"]:
            print(f"      * {diff.name}")
            print(f"          onnx       : {diff.onnx_params}")
            print(f"          onnx_light : {diff.onnxl_params}")


#####################################
# Aggregate summary
# +++++++++++++++++
#
# Finally, a small summary table summarises the situation across all
# inspected sub-modules.

total_missing = 0
total_extra = 0
total_diffs = 0
total_common = 0
for submod_name in SUBMODULES:
    report = compare_submodule(submod_name)
    total_missing += len(report["missing_in_onnxl"])
    total_extra += len(report["extra_in_onnxl"])
    total_diffs += len(report["signature_diffs"])
    total_common += len(report["common"])

print()
print("Summary")
print("-------")
print(f"  total common functions    : {total_common}")
print(f"  total missing in onnx_light: {total_missing}")
print(f"  total extra in onnx_light  : {total_extra}")
print(f"  total signature mismatches : {total_diffs}")
