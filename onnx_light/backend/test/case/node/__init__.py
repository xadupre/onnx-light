# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Python-defined backend node test cases.

Each submodule of this package defines one or more :class:`Base` subclasses
whose ``export*`` static methods register backend test cases through
:func:`onnx_light.backend.test.case.base.expect`. They mirror the upstream
``onnx.backend.test.case.node`` modules so that the same canonical node test
cases are available in ``onnx_light``.
"""
from . import add  # noqa: F401
