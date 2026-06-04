# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import onnx_light.onnx as onnx
    from onnx_light.onnx import checker


def TestOneInput(data):
    try:
        model = onnx.load(data)
        checker.check_model(model)
    except Exception:
        return


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
