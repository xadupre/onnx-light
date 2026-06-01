"""Python-defined backend test cases registered through :class:`Base`.

The :func:`onnx_light.backend.test.case.base.collect_test_case` helper
iterates over all subclasses of
:class:`onnx_light.backend.test.case.base.Base` and runs their
``export_*`` methods to populate the global test registry. Importing the
modules in this sub-package is therefore enough to make the test cases
they define discoverable.
"""

from __future__ import annotations

from . import attention as _attention  # noqa: F401
