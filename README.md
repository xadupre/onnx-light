# onnx-light

[![core](https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml)
[![build](https://github.com/xadupre/onnx-light/actions/workflows/build.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/build.yml)
[![mypy](https://github.com/xadupre/onnx-light/actions/workflows/mypy.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/mypy.yml)
[![Documentation](https://github.com/xadupre/onnx-light/actions/workflows/docs.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/docs.yml)
[![Style](https://github.com/xadupre/onnx-light/actions/workflows/style.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/style.yml)
[![clang-format](https://github.com/xadupre/onnx-light/actions/workflows/clang_format.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/clang_format.yml)
[![pyrefly](https://github.com/xadupre/onnx-light/actions/workflows/pyrefly.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/pyrefly.yml)
[![Spelling](https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml/badge.svg)](https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml)
[![codecov](https://codecov.io/gh/xadupre/onnx-light/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/onnx-light)
[![GitHub repo size](https://img.shields.io/github/repo-size/xadupre/onnx-light)](https://github.com/xadupre/onnx-light)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

onnx without protobuf

[Documentation](https://sdpython.github.io/doc/onnx-light/dev/index.html)

## Key advantages over onnx

- **Files larger than 2 GB** – The standard `onnx` package relies on protobuf,
  which enforces a 2 GB message-size limit and cannot load or save models that
  exceed that threshold. `onnx-light` bypasses protobuf entirely and supports
  arbitrarily large ONNX files.
- **Parallel loading** – Tensor weights can be read in parallel using multiple
  threads, which significantly reduces wall-clock load time for large models:

  ```python
  import onnx_light.onnx

  model = onnx_light.onnx.load("model.onnx", parallel=True, num_threads=4)
  ```

## Getting started

Install the package in editable mode:

```bash
pip install -e .[dev]
```

Run a quick check:

```bash
python -c "import onnx_light; print(onnx_light.__version__)"
```

Load a model with parallel tensor parsing:

```python
import onnx_light.onnx

model = onnx_light.onnx.load("model.onnx", parallel=True, num_threads=4)
print(model.ir_version)
```

## Standalone CMake executable for serialize/parse profiling

You can build a standalone benchmark executable (no Python extension required):

```bash
cmake -S . -B build-prof -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DONNX_LIGHT_BUILD_BENCHMARKS=ON -DONNX_LIGHT_BUILD_PYTHON=OFF
cmake --build build-prof --target bench_parse_serialize
```

Run it:

```bash
./build-prof/bench_parse_serialize -n 20 -t 1
```

On Windows (Visual Studio profiler), configure/build with a Visual Studio generator:

```bat
cmake -S . -B build-prof-vs -G "Visual Studio 17 2022" -A x64 ^
  -DONNX_LIGHT_BUILD_BENCHMARKS=ON -DONNX_LIGHT_BUILD_PYTHON=OFF
cmake --build build-prof-vs --config RelWithDebInfo --target bench_parse_serialize
```

Then profile `build-prof-vs\RelWithDebInfo\bench_parse_serialize.exe` from
**Performance Profiler** (`Alt+F2`) in Visual Studio.
