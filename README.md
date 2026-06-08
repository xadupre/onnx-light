# onnx-light

<p align="center">
  <a href="https://github.com/xadupre/onnx-light">
    <img src="docs/_static/logo.svg" alt="onnx-light logo" width="280">
  </a>
</p>

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

See also the [ONNX roadmap](https://github.com/onnx/onnx/blob/main/ROADMAP.md)
for the upstream project's direction and priorities.

## onnx without protobuf

- **ONNX Files larger than 2 GB** (protobuf is limited to 2Gb)
- **Parallel loading and saving**: 3 times faster with 4 threads compared
  to the single-threaded path
- **Zero-copy parsing** – creates the ModelProto without any tensor copy
- **Aligned external tensor offsets** – external tensor data can be written
  with explicit offset alignment
- **No serialize/parse round-trip for C++ tools** – the Python `ModelProto`
  *is* the C++ `ModelProto`

## Modular C++ libraries

The C++ code is split into several small libraries so a downstream project
can link only what it needs:

- `onnx_light::lib_onnx_proto` – protobuf-compatible message types,
  parser / serializer, external data, optional AES-256 encrypted save / load.
- `onnx_light::lib_onnx_bil` – current ONNX C++ API
- `onnx_light::lib_onnx_op` – lightweight `LightOpSchema` registrations for
  ONNX operator domains, with no shape inference.
- `onnx_light::onnx_light` – full ONNX-compatible schemas (with history),
  checker, inliner, shape inference and version converter.
- `onnx_light::lib_onnx_optim` – shape-inference dispatch table, expression
  engine and graph optimization helpers.
- `onnx_light::onnx_kernels` – C++ kernels used to generate the beckend
- `onnx_light::onnx_backend_test` – C++ backend test infrastructure and
  reference operator kernels.

## Kernels

- Each operator has a corresponding runtime implementation in C++,
  it is used to generated the C++ output of the backend tests.

## Backend Tests

- Fully written in C++, it can be used in any language.
- Output are always generated with a C++ kernel.

## Software Bill of Materials (SBOM)

A [CycloneDX 1.7](https://cyclonedx.org/) Software Bill of Materials is shipped
at the root of the repository as [`sbom.cdx.json`](sbom.cdx.json) and is also
included in the source distribution. It lists the third-party components
bundled into the built artifacts (currently only `nanobind`, used to expose the
C++ extension to Python). The file is validated against the CycloneDX 1.7
schema by the [`SBOM`](.github/workflows/sbom.yml) GitHub Actions workflow.

## Getting started

Install the package in editable mode:

```bash
pip install -e .[dev] -v
```

or

```bash
python setup.py build_ext --inplace
```

`setup.py build_ext` configures CMake with `-DCMAKE_BUILD_TYPE=Release` by
default (unless `CMAKE_ARGS` already sets `CMAKE_BUILD_TYPE`).
``--cpp-tests`` can be used to build with the tests.

To speed up compilation with multiple threads, pass `--parallel` (or `-j`) with
the number of jobs:

```bash
python setup.py build_ext --inplace --parallel 8
```

By default, `python setup.py build_ext` now auto-enables parallel builds
(`--parallel <cpu_count>`) unless `CMAKE_BUILD_PARALLEL_LEVEL` is already set.

Alternatively, when installing with pip, you can control parallel builds using
the ``CMAKE_BUILD_PARALLEL_LEVEL`` environment variable:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e .[dev]
```

Run a quick check:

```bash
python -c "import onnx_light; print(onnx_light.__version__)"
```

Build and run the C++ unit tests from the editable build:

With `pip install`:

```bash
pip install -C build-dir=build -C cmake.build-type=Debug -C cmake.define.ONNX_LIGHT_BUILD_TESTS=ON -e .[dev] -v
ctest --test-dir build --output-on-failure
```

With `setup.py`:

```bash
python setup.py build_ext --inplace --build-temp build --cpp-tests
ctest --test-dir build --output-on-failure
```

On multi-config generators such as Visual Studio, add the matching
configuration to `ctest`: use `-C Debug` when the build was configured with
`cmake.build-type=Debug`, and `-C Release` after `python setup.py build_ext
--cpp-tests`.

Load a model with parallel tensor parsing:

```python
import onnx_light.onnx

model = onnx_light.onnx.load("model.onnx", num_threads=4)
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
- `liblib_onnx_proto.a`, `liblib_onnx_op.a`, and `liblib_onnx_lib.a` (the static libraries) into `<prefix>/lib`
- All public C++ headers into `<prefix>/include/onnx_light`
- CMake package config files into `<prefix>/lib/cmake/onnx_light`

### Using `find_package(onnx_light)` in your project

Once installed, any CMake project can locate and link the library with:

```cmake
find_package(onnx_light REQUIRED)
target_link_libraries(my_target PRIVATE onnx_light::onnx_light)
```

If the code only needs protobuf-compatible message parsing/serialization and does
not need operator schemas, checker, or shape inference, it can link against the
lighter ``onnx_light::lib_onnx_proto`` target instead:

```cmake
find_package(onnx_light REQUIRED)
target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_proto)
```

If the code needs lightweight math operator schemas without shape inference, it
can link against ``onnx_light::lib_onnx_op`` and query
``onnx_op::math::GetAllOnnxOpMathSchemasWithHistory()``.

Pass `-DCMAKE_PREFIX_PATH=<prefix>` when configuring your project if the
library was installed to a non-standard prefix.
