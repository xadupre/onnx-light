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

[Documentation](https://sdpython.github.io/doc/onnx-light/dev/index.html)

## onnx without protobuf

- **Files larger than 2 GB** – The standard `onnx` package relies on protobuf,
  which enforces a 2 GB message-size limit and cannot load or save models that
  exceed that threshold. `onnx-light` bypasses protobuf entirely and supports
  arbitrarily large ONNX files.
- **Parallel loading** – Tensor weights can be read in parallel using multiple
  threads, which significantly reduces wall-clock load time for large models
- **Zero-copy parsing** – creates the ModelProto without any tensor copy,
  all initializers point to the data inside ModelProto

## Getting started

Install the package in editable mode:

```bash
pip install -e .[dev]
```

or

```bash
python setup.py build_ext --inplace
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

## Using onnx_light as a C++ library

### Installing the C++ library

Build and install the static library and headers to a local prefix
(Python extension not required):

```bash
cmake -S . -B build-install
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_LIGHT_BUILD_PYTHON=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build  build-install
cmake --install build-install
```

This installs:
- `liblib_onnx_cpp.a` (the static library) into `<prefix>/lib`
- All public C++ headers into `<prefix>/include/onnx_light`
- CMake package config files into `<prefix>/lib/cmake/onnx_light`

### Using `find_package(onnx_light)` in your project

Once installed, any CMake project can locate and link the library with:

```cmake
find_package(onnx_light REQUIRED)
target_link_libraries(my_target PRIVATE onnx_light::onnx_light)
```

Pass `-DCMAKE_PREFIX_PATH=<prefix>` when configuring your project if the
library was installed to a non-standard prefix.

### Standalone example: `examples/load_onnx_light_time`

The `examples/load_onnx_light_time` directory contains a self-contained CMake
project that demonstrates loading an ONNX file and reporting load timing
statistics (plus model metadata) with the onnx_light C++ API.

Build it after installing the library:

```bash
cmake -S examples/load_onnx_light_time -B build-load-onnx-light-time \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/local
cmake --build build-load-onnx-light-time
```

Run it:

```bash
./build-load-onnx-light-time/load_onnx_light_time path/to/model.onnx 10 4
```

Example output:

```
Loaded: path/to/model.onnx
  Average load (ms): 5.321
  Min load (ms)    : 5.002
  Max load (ms)    : 5.889
  IR version       : 9
  Producer name    : my_framework
  Graph name       : my_graph
  Nodes            : 42
  Inputs           : 2
  Outputs          : 1
  Initializers     : 10
```
