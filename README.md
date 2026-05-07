# onnx-light

[![core](https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml)
[![build](https://github.com/xadupre/onnx-light/actions/workflows/build.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/build.yml)
[![mypy](https://github.com/xadupre/onnx-light/actions/workflows/mypy.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/mypy.yml)
[![Documentation](https://github.com/xadupre/onnx-light/actions/workflows/docs.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/docs.yml)
[![Style](https://github.com/xadupre/onnx-light/actions/workflows/style.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/style.yml)
[![pyrefly](https://github.com/xadupre/onnx-light/actions/workflows/pyrefly.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/pyrefly.yml)
[![Spelling](https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml)
[![codecov](https://codecov.io/gh/xadupre/onnx-light/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/onnx-light)
[![GitHub repo size](https://img.shields.io/github/repo-size/xadupre/onnx-light)](https://github.com/xadupre/onnx-light)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

onnx without protobuf

## Getting started

Install the package in editable mode:

```bash
pip install -e .[dev]
```

Run a quick check:

```bash
python -c "import onnx_light; print(onnx_light.__version__)"
```

Load a model with parallelized tensor parsing:

```python
import onnx_light.onnx as onnxl

model = onnxl.load("model.onnx", parallel=True, num_threads=4)
print(model.ir_version)
```
