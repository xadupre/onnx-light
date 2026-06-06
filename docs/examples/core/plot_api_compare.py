"""
.. _l-example-plot-api-compare:

Compares the Python API of ``onnx`` and ``onnx_light.onnx``
============================================================

This example walks the public Python API exposed by the upstream
:mod:`onnx` package and by :mod:`onnx_light.onnx` and reports the
discrepancies between the two.

It relies on :mod:`onnx_light.compatibility`, a small sub-package
dedicated to checking whether a Python package is compatible with
:mod:`onnx_light`.  The same helpers are used by the unit test
``unittests/main/test_plot_api_compare.py``.

For every common sub-module (``helper``, ``numpy_helper``,
``checker``, ``defs``, ``parser``, ``shape_inference``,
``version_converter``, ``compose``, ``utils``, ``inliner``) the script
lists:

* the sub-modules that are exposed by one package but not the other,
* the public functions that are exposed by one package but not the
  other,
* the public functions that exist in both packages but whose
  signatures differ (positional/keyword parameter names).

It also compares the *top-level* public functions of both packages
(:func:`onnx.load` / :func:`onnx.save` vs :func:`onnx_light.onnx.load` /
:func:`onnx_light.onnx.save`).

Finally, it lists the ONNX node-level backend tests (exposed by
:mod:`onnx.backend.test`) that do not yet have a counterpart in
:mod:`onnx_light.backend.test`.
"""

from __future__ import annotations

import onnx
import onnx.backend.base
import onnx.inliner  # noqa: F401  -- ensure the inliner sub-module is bound on ``onnx``
from onnx.backend.test import BackendTest

import onnx_light.onnx as onnxl
from onnx_light.backend.test.case.base import collect_test_case
from onnx_light.compatibility import DEFAULT_SUBMODULES, compare_packages

#####################################
# Run the comparison
# ++++++++++++++++++

report = compare_packages(onnx, onnxl, submodules=DEFAULT_SUBMODULES)

#####################################
# Sub-module overview
# +++++++++++++++++++

submods = report["submodules"]
print(f"common sub-modules                : {len(submods['common'])}")
print(f"sub-modules only in onnx          : {len(submods['missing_in_onnxl'])}")
print(f"sub-modules only in onnx_light    : {len(submods['extra_in_onnxl'])}")
print()
print("Sub-modules only in onnx:")
for name in submods["missing_in_onnxl"]:
    print(f"  - {name}")
print()
print("Sub-modules only in onnx_light.onnx:")
for name in submods["extra_in_onnxl"]:
    print(f"  - {name}")


#####################################
# Top-level functions
# +++++++++++++++++++
#
# :func:`onnx.load` / :func:`onnx.save` and their ``onnx_light`` counterparts
# live directly on the package (no sub-module), so they are reported
# separately here.

top = report["top_level"]
print()
print("=== top-level ===")
print(f"  common functions          : {len(top['common'])}")
print(f"  missing in onnx_light.onnx: {len(top['missing_in_onnxl'])}")
print(f"  extra in onnx_light.onnx  : {len(top['extra_in_onnxl'])}")
print(f"  signature mismatches      : {len(top['signature_diffs'])}")
if top["missing_in_onnxl"]:
    print("  - missing in onnx_light.onnx:")
    for name in top["missing_in_onnxl"]:
        print(f"      * {name}")
if top["extra_in_onnxl"]:
    print("  - extra in onnx_light.onnx:")
    for name in top["extra_in_onnxl"]:
        print(f"      * {name}")
if top["signature_diffs"]:
    print("  - signature mismatches:")
    for diff in top["signature_diffs"]:
        print(f"      * {diff.name}")
        print(f"          onnx       : {diff.onnx_params}")
        print(f"          onnx_light : {diff.onnxl_params}")


#####################################
# Function-level comparison
# +++++++++++++++++++++++++

for submod_name in DEFAULT_SUBMODULES:
    sub = report["per_submodule"][submod_name]
    print()
    print(f"=== {submod_name} ===")
    print(f"  common functions          : {len(sub['common'])}")
    print(f"  missing in onnx_light.onnx: {len(sub['missing_in_onnxl'])}")
    print(f"  extra in onnx_light.onnx  : {len(sub['extra_in_onnxl'])}")
    print(f"  signature mismatches      : {len(sub['signature_diffs'])}")
    if sub["missing_in_onnxl"]:
        print("  - missing in onnx_light.onnx:")
        for name in sub["missing_in_onnxl"]:
            print(f"      * {name}")
    if sub["extra_in_onnxl"]:
        print("  - extra in onnx_light.onnx:")
        for name in sub["extra_in_onnxl"]:
            print(f"      * {name}")
    if sub["signature_diffs"]:
        print("  - signature mismatches:")
        for diff in sub["signature_diffs"]:
            print(f"      * {diff.name}")
            print(f"          onnx       : {diff.onnx_params}")
            print(f"          onnx_light : {diff.onnxl_params}")


#####################################
# Aggregate summary
# +++++++++++++++++

total_missing = len(top["missing_in_onnxl"])
total_extra = len(top["extra_in_onnxl"])
total_diffs = len(top["signature_diffs"])
total_common = len(top["common"])
for submod_name in DEFAULT_SUBMODULES:
    sub = report["per_submodule"][submod_name]
    total_missing += len(sub["missing_in_onnxl"])
    total_extra += len(sub["extra_in_onnxl"])
    total_diffs += len(sub["signature_diffs"])
    total_common += len(sub["common"])

print()
print("Summary")
print("-------")
print(f"  total common functions     : {total_common}")
print(f"  total missing in onnx_light: {total_missing}")
print(f"  total extra in onnx_light  : {total_extra}")
print(f"  total signature mismatches : {total_diffs}")


#####################################
# Backend test differences
# ++++++++++++++++++++++++
#
# ``onnx.backend.test`` exposes one node-level test case per operator
# scenario.  ``onnx_light`` reimplements the same backend test
# infrastructure under :mod:`onnx_light.backend.test`.  This section
# lists the ONNX node-level backend tests that do **not** yet have a
# counterpart in :mod:`onnx_light` (i.e. there is no ``onnx_light``
# backend test whose name contains the stripped ONNX test name).


class _DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def supports_device(cls, device):
        return True


def _onnx_node_cpu_test_names() -> list[str]:
    """Returns all ONNX node-level backend test method names ending in ``_cpu``."""
    bt = BackendTest(_DummyBackend, "dummy")
    cls = bt.test_cases["OnnxBackendNodeModelTest"]
    return sorted(
        m
        for m in dir(cls)
        if m.startswith("test_") and m.endswith("_cpu") and callable(getattr(cls, m))
    )


_light_names = set(collect_test_case().keys())
_onnx_tests = _onnx_node_cpu_test_names()

_missing_in_onnxl: list[str] = []
for _method_name in _onnx_tests:
    _stripped = _method_name[len("test_") : -len("_cpu")]
    if not any(_stripped in _light_name for _light_name in _light_names):
        _missing_in_onnxl.append(_stripped)

print()
print("=== backend tests ===")
print(f"  ONNX node-level backend tests       : {len(_onnx_tests)}")
print(f"  onnx_light backend test cases       : {len(_light_names)}")
print(f"  ONNX tests missing in onnx_light    : {len(_missing_in_onnxl)}")
if _missing_in_onnxl:
    print("  - ONNX backend tests without an onnx_light counterpart:")
    for _name in _missing_in_onnxl:
        print(f"      * {_name}")
