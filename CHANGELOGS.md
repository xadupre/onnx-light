# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.7] – Unreleased

### New Features

- Added a `RuntimeContext` allocator framework (`SimpleRawBufferAllocator`, allocator-backed kernel outputs) and custom-op `shape_tag` registration hooks.
- Added a symbolic gradient framework (reverse-mode `Conv`/normalization gradients with backend-test verification) and a peak-memory dispatch registry.
- Extended the Python/serialization API (`SaveToFileDescriptor`, `SerializeToOstream`) and added `cp313`/`cp314` wheels.

### Improvements

- Reorganized the C++ tree around a new intermediate `lib_onnx_core` and an `onnx_extensions` layout (`onnx_backend_test`/`onnx_kernels`/`onnx_gradient` moves, `onnx_optim`→`onnx_shapes` rename, `ExecutionPlan` split).
- Migrated kernels to the dedicated `Shape` type and routed their output allocation through the `RuntimeContext` allocator, preserving ownership and reusing shared empty-tensor fallbacks.
- Reduced string copies and allocations across parsing/graph paths (`RepeatedProtoField`, more `constexpr` helpers, removed `memcpy`/type-punning hotspots) and extended shape-tag metadata coverage.

### Fixes

- Corrected in-place reuse and shape-tag propagation across many operators and preserved `doc_string` in the IR converter.
- Propagated upstream ONNX fixes (shape inference, locale-independent text parser/printer, `TopK` `sorted=0`) and hardened version conversion for malformed `Scan`/default-domain imports.
- Fixed expression-simplification, SVG rendering edge cases, and spelling across the codebase.

### Security

- Fixed a `size_t` underflow in `Scan` shape inference (GHSA-qrhj-v62m-vmpf) and hardened zero-copy ORT parsing and tar-extraction path validation.
- Prevented command injection/insecure I/O (`graphviz` args, temp files) and GitHub Actions code injection in workflows.
- Addressed external-data advisories, raised the ONNX minimum to 1.21.0, and added defense-in-depth checks (GHSA-3jf9-582g-jjmq, GHSA-xrch-8vh7-h656, GHSA-p893-rvq9-2xf9, GHSA-hqmj-h5c6-369m, GHSA-8qff-7g33-75mx).

### Testing

- Extended C++/Python backend and shape-inference coverage (gradient verification, pooling, local functions, metadata/value-tag enforcement, `OpSchema` attribute parity).
- Delayed backend-test `ModelProto` construction via `TestCase::emplace_model()` and forwarded `TestMode` through the Python facade.

### Documentation & CI

- Added the changelog and string-types documentation, bumped the release version to `0.1.7` across canonical metadata.
- Sped up CI (`sccache` for `clang-tidy`, trimmed Windows jobs, parallel preflights, C++-build skips in style/typing jobs).

## [0.1.4] – 2026-07-07

### Fixes

- Fix Win32 narrowing in backend-test DLPack shape conversion ([#3223](https://github.com/xadupre/onnx-light/pull/3223))
- Suppress GCC 13 false-positive `-Wfree-nonheap-object` in `ComputeScaleIndex` ([#3227](https://github.com/xadupre/onnx-light/pull/3227))

### Security

- Validate `raw_data` alignment in `ParseData` to prevent out-of-bounds copy (propagate ONNX #8032) ([#3225](https://github.com/xadupre/onnx-light/pull/3225))

## [0.1.3] – 2026-07-06

### New Features

- Added serialization safety and diagnostics: `SerializeOptions.max_serialized_size_bytes` cap, node indexes in `pretty_onnx`, verbose `fillshape` output/progress.
- Extended `fillshape`/CLI tooling: `--token` binding, `--release-info`, `run` subcommand, `--verbose [LEVEL]`, `python -m onnx_light fillshape`.
- Added shape/dimension utilities: `dim_ranges_from_expressions`, `make_random_input`, C++/nanobind shape/axes/weight metadata tagging with tag-aware Mermaid/SVG rendering.
- Added `ai.rt` `DelayedInitializer` schema/kernel/shape-inference, a pre-serialization weight rewrite callback, and ChaCha20-Poly1305 encrypted model I/O.
- Added parsing/graph-rendering features: `ParseFromIstream`, parse callback support, `include_inplace`/`include_release` rendering options, in-place reuse metadata recording, and a graph-input overwrite guard.

### Improvements

- Updated onnxruntime integration and mirrored an upstream nanobind cross-compile CMake fix (onnx/onnx#8157).
- Improved shape inference: Python `//` floor-division semantics, symbolic `Reshape`/`Slice` expressions, empty-axis broadcasting, half-precision `RMSNormalization`.
- Propagated and reused shape-tag metadata through `Reshape` and `fillshape` writing paths.
- Reorganized in-place reuse and value-tag helpers into `onnx_shapes.annotations`.
- Miscellaneous: improved `to_svg` layout, `RepeatedField` iterable support, `ByteSpan`/`TensorProto` deleter implementation, tiny external-tensor inlining moved into `ParseOptions`.

## [0.1.1] – 2026-06-22

### New Features

- Added a pure-Python textproto parser/serializer and `format="textproto"` support in `load`/`save`.
- Extended shape-expression support: exact-division operator (`/:`), `compare_expressions`.
- Extended proto API surface: `WhichOneof`, keyword-argument constructors, `RepeatedProtoField.add()` kwargs, `String.decode()`.
- Added sub-byte dtype support (`int2`/`uint2`/`int4`/`uint4`/`float4`) in `make_tensor` and exposed `OpSchema.Attribute.default_value`.
- Added `PrintOptions` indentation/`inline_threshold`, GitHub Releases wheel publishing, and offset-aware `Attention` `is_causal` masking for external KV cache.

### Improvements

- Brought `infer_shapes` to parity with onnx (`check_type`, `strict_mode`, `data_prop`) and validated `ConvTranspose` group divisibility.
- Registered repeated-field containers as `collections.abc.Sequence` with list/`String` comparison support.
- Guessed in-place input reuse from `onnx_shapes` shape inference and reduced Windows/macOS wheel size.
- Bumped mirrored ONNX to 1.23.0, enabled the ONNX backend optional-sequence loop case, and extended `Where` backend coverage across dtypes.
- Replaced ad-hoc exception throws with `EXT_ENFORCE_INVALID`/`EXT_THROW_INVALID`, fixed serialization/IOBinding gaps (`TensorProto.segment`, sub-byte `Cast`, low-precision ORT backend tests), and passed non-null pointer params by reference (onnx#8105).

### Testing

- Added unit tests for proto/numpy helpers, print helpers, and field-serialization helpers.
- Added unit tests for sparse/sequence/optional helpers and `RunLoopWithSequenceState`.
- Added unit tests for `get_total_memory_gb`, `get_cpu_topology`, `_schema_to_rst`, and `hide_stdout`.
- Added C++ serialization tests and shape-inference examples (`MatMul` with an initializer weight, sequential `TopK`).
- Added manual CI jobs validating onnx-light against ir-py, mbext, and yobx.

## [0.1.0] – 2026-06-18

Initial public release.
