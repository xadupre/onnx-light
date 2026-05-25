"""Compatibility helpers between :mod:`onnx_light` and other Python packages.

This sub-package is intended to host utilities that check whether a
Python package is compatible with :mod:`onnx_light` — same set of
sub-modules, same set of public functions and aligned signatures.

The first comparator targets :mod:`onnx` (the upstream ONNX package)
and is used both programmatically (see
``unittests/main/test_plot_api_compare.py``) and as a gallery example
(see ``docs/examples/core/plot_api_compare.py``).
"""

from __future__ import annotations

from .api_compare import (  # noqa: F401
    DEFAULT_SUBMODULES,
    SignatureDiff,
    compare_packages,
    compare_signatures,
    compare_submodule,
    compare_top_level_functions,
    list_public_functions,
    list_submodules,
)
from .schema_diff import (  # noqa: F401
    AttributeDiff,
    ConstraintDiff,
    DocDiff,
    ParameterDiff,
    SchemaDiff,
    compare_schemas,
)
