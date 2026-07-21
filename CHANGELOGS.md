# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.7] – Unreleased

### New Features

- Add `SimpleRawBufferAllocator` with virtual interface and wire into `RuntimeContext` ([#3241](https://github.com/xadupre/onnx-light/pull/3241))
- Add `RuntimeContext*` to all kernel `operator()` signatures for allocator-backed output buffers ([#3261](https://github.com/xadupre/onnx-light/pull/3261))
- Added a symbolic gradient framework: reverse-mode gradients for `Conv` ([#3554](https://github.com/xadupre/onnx-light/pull/3554)) and normalization operators ([#3556](https://github.com/xadupre/onnx-light/pull/3556)), backend-test gradient verification across all registered ops ([#3566](https://github.com/xadupre/onnx-light/pull/3566)), and a gradient gallery/API/design documentation page ([#3523](https://github.com/xadupre/onnx-light/pull/3523)).
- Added a peak-memory dispatch registry to `onnx_core/shapes` ([#3604](https://github.com/xadupre/onnx-light/pull/3604)).
- Added `SaveToFileDescriptor` and `SerializeToOstream` to the Python API ([#3590](https://github.com/xadupre/onnx-light/pull/3590)).
- Added custom-op `shape_tag` registration hooks to `ComputeContext` ([#3560](https://github.com/xadupre/onnx-light/pull/3560)).
- Added `cp313`/`cp314` standard wheel targets to the release and weekly builds ([#3579](https://github.com/xadupre/onnx-light/pull/3579)).

### Improvements

- Introduced an intermediate `lib_onnx_core` library between `lib_onnx_proto` and the higher-level libraries ([#3575](https://github.com/xadupre/onnx-light/pull/3575)).
- Reorganized the C++ tree under `onnx_extensions`: moved `onnx_backend_test` and `onnx_kernels` ([#3612](https://github.com/xadupre/onnx-light/pull/3612)), and the `onnx_gradient` library ([#3602](https://github.com/xadupre/onnx-light/pull/3602)) and Python facade ([#3606](https://github.com/xadupre/onnx-light/pull/3606)); renamed the `onnx_optim` module to `onnx_shapes` and `lib_onnx_optim` to `lib_onnx_shape` ([#3596](https://github.com/xadupre/onnx-light/pull/3596)); and split `ExecutionPlan` into `execution_plan.h/.cc` ([#3608](https://github.com/xadupre/onnx-light/pull/3608)).
- Improved allocators and preserved allocator ownership across kernels, including `Loop` INT64/BOOL scalars ([#3564](https://github.com/xadupre/onnx-light/pull/3564)), `SliceTensorAlongAxis` ([#3562](https://github.com/xadupre/onnx-light/pull/3562)), and a shared empty-tensor fallback in kernel dispatch ([#3543](https://github.com/xadupre/onnx-light/pull/3543), [#3567](https://github.com/xadupre/onnx-light/pull/3567)).
- Reduced unnecessary `std::string` copies in the `OnnxParser::Parse(GraphProto&)` and graph-input collection paths ([#3552](https://github.com/xadupre/onnx-light/pull/3552), [#3547](https://github.com/xadupre/onnx-light/pull/3547), [#3529](https://github.com/xadupre/onnx-light/pull/3529)), migrated remaining node-list paths to `utils::RepeatedProtoField` ([#3525](https://github.com/xadupre/onnx-light/pull/3525)), and added more `constexpr` helpers ([#3530](https://github.com/xadupre/onnx-light/pull/3530)).
- Migrated most kernels (cast, elementwise, math, tensor, sequence, reduction, pooling, `TopK`, `Einsum`, `MatMul`, `Where`, normalization, etc.) from `std::vector<int64_t>` to the dedicated `Shape` type for shape/state handling.
- Routed kernel output allocation through the `RuntimeContext` allocator across many kernels (`Bernoulli`, `CumSum`/`CumProd`, `Min`/`Max`/`Mean`/`Sum`, `NonMaxSuppression`, `DelayedInitializer`, `QuantizeLinear`, `TopK`, image decoder, `MatMulInteger`), replacing ad-hoc or inline buffers.
- Extended shape-tag metadata coverage to initializers and cached symbolic byte-size simplification for inplace-reuse analysis.
- Build/tooling improvements: MSVC hardening aligned with ONNX Runtime, `NodeProto` repeated-string `str.join` compatibility, `constexpr` enum-name helpers.
- Eliminated `memcpy`/type-punning hotspots (`cast_float8`, `MelWeightMatrix`, `EyeLike`) and added shared helpers (`Shape::product()`, `ElementSize`, `BroadcastShape`) to remove duplicated kernel code.

### Fixes

- Corrected in-place reuse annotations for `Transpose`/`Reshape`/`Unsqueeze`/`EyeLike` (`kGreater` vs `kEqual`, symbolic dimensions).
- Fixed shape-tag propagation across `Concat`/`Reshape`/`Cast`/`Sub` and seeded `weight` tags for graph inputs/outputs/initializers.
- Propagated upstream ONNX shape-inference fixes (`Scan` `num_scan_inputs` underflow, `Range` symbolic dims).
- Fixed several kernels (`If`, `Bernoulli`, `EyeLike`, `ConstantOfShape`, `CastLike`, `AffineGrid`, `BatchNormalization`, `MelWeightMatrix`, multi-output kernels) to use the `RuntimeContext` allocator and avoid unnecessary allocations/`memcpy`.
- Fixed expression-simplification integer-coefficient combination and SVG barycenter crossing-minimisation edge handling.
- Preserved `doc_string` on `Value` in the IR protobuf converter ([#3618](https://github.com/xadupre/onnx-light/pull/3618)).
- Hardened version conversion for malformed `Scan` nodes and `ai.onnx` default-domain imports ([#3544](https://github.com/xadupre/onnx-light/pull/3544)).
- Propagated upstream locale-independent ONNX text parser/printer fixes ([#3546](https://github.com/xadupre/onnx-light/pull/3546)) and `TopK` `sorted=0` acceptance ([#3545](https://github.com/xadupre/onnx-light/pull/3545)).
- Fixed spelling issues across the codebase ([#3610](https://github.com/xadupre/onnx-light/pull/3610)).

### Security

- Fixed a `size_t` underflow in `Scan` shape inference (GHSA-qrhj-v62m-vmpf).
- Hardened the zero-copy ORT parsing path and tar-extraction path validation.
- Validated `graphviz` format arguments and replaced insecure temp-file creation to prevent command injection/insecure I/O.
- Prevented GitHub Actions code/expression injection in the `clang_tidy` and release-wheel workflows.
- Addressed external-data security advisories (GHSA-3jf9-582g-jjmq, GHSA-xrch-8vh7-h656), raised the ONNX minimum to 1.21.0 (GHSA-p893-rvq9-2xf9), and investigated/closed GHSA-hqmj-h5c6-369m and GHSA-8qff-7g33-75mx with added defense-in-depth checks.

### Testing

- Added regression tests for upstream ONNX shape-inference fixes (`Loop` early-exit, `ScatterND` reductions, `AvgPool`/`Conv` edge cases).
- Enforced full metadata-tagging coverage in shape-tag backend tests.
- Added C++ backend coverage for pooling, local-function shape propagation, and the `qwen3_4_layers_like` case.
- Added inplace-metadata expectations for `tiny_llm` `Unsqueeze` and more backend-test options.
- Delayed C++ backend-test `ModelProto` construction via `TestCase::emplace_model()` across registered cases.
- Enforced expected `onnx_light.value_tag` on inputs/outputs/initializers in `shape_tag` backend tests ([#3621](https://github.com/xadupre/onnx-light/pull/3621)).
- Forwarded `TestMode` through the `collect_test_case` Python facade ([#3620](https://github.com/xadupre/onnx-light/pull/3620)).
- Verified that `LightOpSchema` and the `onnx_light` `defs` `OpSchema` share the same attributes ([#3528](https://github.com/xadupre/onnx-light/pull/3528)).

### Documentation

- Added the changelog to the documentation ([#3569](https://github.com/xadupre/onnx-light/pull/3569)).
- Updated the string-types documentation for proto fields ([#3541](https://github.com/xadupre/onnx-light/pull/3541)).

### Build & CI

- Added an intermediate build target and reorganized libraries (see Improvements), and bumped the release version to `0.1.7` across canonical metadata ([#3577](https://github.com/xadupre/onnx-light/pull/3577)).
- Sped up CI: cached the `clang-tidy` build with `sccache` ([#3617](https://github.com/xadupre/onnx-light/pull/3617)), trimmed Windows CI ([#3527](https://github.com/xadupre/onnx-light/pull/3527), [#3542](https://github.com/xadupre/onnx-light/pull/3542)), ran the reduced and no-onnx preflights in parallel ([#3593](https://github.com/xadupre/onnx-light/pull/3593)), and skipped the C++ build in the style/typing jobs ([#3581](https://github.com/xadupre/onnx-light/pull/3581), [#3583](https://github.com/xadupre/onnx-light/pull/3583)).

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
