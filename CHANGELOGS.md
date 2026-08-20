# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.20] – Unreleased

### Improvements

- Treated C++ compiler warnings as errors on GCC/Clang too (`-Werror`), controlled by the
  new `ONNX_LIGHT_WERROR` CMake option which also gates the existing MSVC `/WX`.

- Made the requested degree of parallelism effective: a runtime session now leases a shared
  CPU executor matching its execution policy and every parallel region its kernels launch
  runs on exactly those participants instead of a process-wide thread pool.
- Exposed typed CPU execution requests, immutable resolutions, executor sharing identities,
  and optional dispatch counters through the Python runtime and `ReferenceEvaluator`.
- Made calibration use an explicitly selected CPU executor and reject execution descriptors
  that disagree with the active participants.
- Tagged node runtime events with the exact shared CPU executor instance and its effective
  participants, and fixed future prepared execution to the same executor ownership contract.

### Documentation & CI

- Standardized the python imports of the helper modules: `import onnx.helper as oh` and
  `import onnx.numpy_helper as onh` (and their `onnx_light.onnx` counterparts) replace
  the `from onnx import helper, numpy_helper` form in the code base and the documentation.
- Bumped the release version to `0.1.20`.

## [0.1.19] – 2026-08-18

### Improvements

- Kept `ReferenceEvaluator` NumPy inputs zero-copy across repeated runs, including
  non-contiguous inputs, while preserving their lifetime and allocator accounting.
- Improved element-wise steady-state performance with a persistent low-latency thread pool,
  physical-core-aware defaults, and calibration that matches the runtime thread count.
- Optimized the `Abs` kernel for floating-point types with sign-bit masking and a lower,
  tunable parallel-execution threshold.

### Documentation & CI

- Updated the `Abs` benchmark to compare steady-state execution fairly across supported
  element types and omit incomparable `bfloat16` speed-ups.

## [0.1.18] – 2026-08-18

### Improvements

- Kept backend benchmark models compatible with ONNX IR version 13.
- Generated `BFLOAT16` backend benchmark cases by default.
- Declared `ml_dtypes>=0.5.0` as a package dependency.

### Fixes

- Fixed release builds against older NumPy versions that do not expose `PyDataType_KIND`.

### Documentation & CI

- Bumped the release version to `0.1.18`.
- Added a design page for buffer-reuse arenas and marked the I/O arena plan complete.
- Reorganized the design documentation and runtime examples.

## [0.1.17] – 2026-08-17

### New Features

- Added a pattern-based `GraphBuilder` optimizer: a `GraphGraph` read-only index, a `PatternOptimization` engine that matches and rewrites subgraphs (recursively, including nested subgraphs), a pattern registry and dispatch table, a dedicated patterns library, and canonicalization patterns for redundant `Cast`, binary `Cast`, and `Cast`-op-`Cast`.
- Added a kernel auto-tuning framework: a `KernelTuning` registry and persistent tuning cache with atomic updates, shared unary/binary calibration, portable GEMM tuning, and per-session integration through the runtime session and kernel context, tuning logical and math kernels.
- Added constant folding to `GraphBuilder`, backed by the read-only `GraphGraph` index and build-time constant-information analysis.
- Added a `RunModel` C++ helper for whole-model execution with `Tensor` I/O and exposed it in Python.
- Exposed a public tensor comparison helper with `atol`/`rtol`.
- Added persistent execution and I/O buffer-reuse arenas with independent retention controls,
  output leases that safely outlive a runtime context, and Python runtime integration.

### Improvements

- Routed every built-in kernel output by output slot directly to its final arena and every
  workspace to the execution arena, eliminating the final execution-to-I/O copy on the normal
  runtime path, including for mixed-output nodes.
- Improved runtime performance and expanded the Python NumPy kernel bindings.
- Added a CPU descriptor exposing processor topology for processor-aware kernel thresholds.
- Made the opaque type unconditional (propagated from onnx/onnx#8269).
- Corrected optional input/output presence in `InferenceContextImpl` (propagated from onnx/onnx#8271).
- Made `ScopedResource` use a `Traits` type to avoid a pointer non-type template parameter (propagated from onnx/onnx#8279).

### Fixes

- Fixed a local build failure in `setup.py`.
- Fixed a `cpp-tests` install failure by disabling GMock in the fetched googletest.

### Documentation & CI

- Bumped the release version to `0.1.17`.
- Added next-steps design pages for profiling and async execution, parallel model initialization with prepacking, prepared execution, a buffer-reuse arena allocator, processor-aware kernel thresholds, and wheel packaging.
- Documented the kernel tuning workflow and graph pattern integration.
- Added a benchmark for cold kernel tuning resolution and characterized allocator-backed output lifetime.
- Updated the formatting and typing tools.
- Switched C++ backend tests to `RandnTensor` instead of `Tensor::FromX` + `Randn`.
- Improved schema-comparison coverage.
- Removed the numpy `run_model` benchmark from the `plot_abs_benchmark` example and asserted onnx-light is slower than ONNX Runtime at the smallest size.

## [0.1.16] – 2026-08-10

### Improvements

- Stopped zero-initialising result buffers on allocation, zeroing explicitly only where kernels rely on it.
- Reduced the binary size of `lib_onnx_proto` further across several passes.
- Exported `ParseLimitExceeded` in the proto public API.
- Fixed `get_cpp_build_info` for editable installs and static builds.

### Documentation & CI

- Bumped the release version to `0.1.16`.
- Added a next-steps design page for a C++ pattern-based `GraphBuilder` optimizer.
- Added last-updated dates to the gallery examples and split `plot_onnx_time` into three separate figures.
- Added backend run-model tests for previously untested logical kernels.
- Improved test coverage for `run_nodes.cc`, `simple_tensor.cc`, `node_helpers`, and `onnx_light_helpers.cc`.
- Added a test verifying the serialized `raw_data` size matches the buffer size for `QuantizedTensorProto`.
- Enabled build caching in CI.

## [0.1.15] – 2026-08-08

### New Features

- Added support for registering custom kernels globally or per session.
- Gave every kernel a unique device- and library-qualified name, exposed through the session and Python.

### Improvements

- Made dispatch-table overrides order-independent.
- Spawned the load/save thread-pool workers lazily to fix slow small-model initialization.
- Reduced the stripped `lib_onnx_proto` size by hiding internal symbols and explicitly exporting
  its public cross-library ABI.

### Documentation & CI

- Bumped the release version to `0.1.15`.
- Annotated `onnx.h` with the canonical ONNX proto definitions.
- Added a gallery example benchmarking model initialization (ONNX Runtime vs onnx-light).
- Added a `register_custom_kernel` example in both C++ and Python.
- Added quantization and next-steps documentation.

## [0.1.14] – 2026-08-07

### New Features

- Added a `NodeProto` callback hook to `ParseOptions` and `SerializeOptions` so callers can inspect or rewrite each node while a model is being parsed or serialized.

### Improvements

- Propagated reverse and bidirectional `RNN`/`GRU`/`LSTM` support (propagated from onnx/onnx#7935).
- Improved ONNX Runtime compatibility: added ordering operators and an implicit conversion to `const std::string&` to `OptionalString`, made `add_string_data` accept an `OptionalString`, switched `AttributeProto.strings` to `FIELD_REPEATED_STR`, and stopped creating an empty shape in `TypeProto::FromString` for tensor types without a shape.
- Modernized the `ParseData` `raw_data` handling (propagated from onnx/onnx#8109).
- Made `FileLoadMode.AUTO` use a buffered stream instead of memory mapping.
- Honored `file_load_mode="MMAP"` when loading models with external data.
- Loaded deeply nested control-flow models by raising the default proto recursion limit.
- Accepted the packed wire format for repeated integer proto fields.
- Added an opt-in bypass for the external `TensorProto` location error in the checker.
- Reduced the binary size of `onnx_proto` via section garbage collection / dead-code elimination.

### Fixes

- Fixed single-file mmap `no_copy` zero-copy parsing.
- Added proto2-style presence tracking for `FIELD_BYTES`.
- Fixed the `Attention` mode-3 fully-masked `NaN` output.
- Stopped reading an optional message sub-field from auto-creating it.
- Fixed a `ParseFromFile` `RuntimeError` when `file_load_mode` is set together with an `external_data_file`.
- Returned the real parse result from `TraditionalMLData::ParseFromArray`.
- Refined the cyclic model-local function error message (propagated from onnx/onnx#8233).
- Fixed compilation when OpenSSL is built without ChaCha20-Poly1305.
- Fixed the `ScatterND` `updates` shape documentation (propagated from onnx/onnx#8212).

### Security

- Added overflow-checked `int64` arithmetic in shape inference (propagated from onnx/onnx#8031).
- Bounded external-data padding on serialization (propagated from onnx/onnx#8260).

### Testing

- Made the upstream node backend-test consumers work when `onnx` builds node test
  data on the fly (`TestCase.model` / `TestCase.data_sets` in memory with
  `model_dir` set to `None`) instead of materialising it on disk under
  `onnx/backend/test/data/node` (propagated from onnx/onnx#7959). Both the
  `schema_comparison` backend-test counter and the `ImageDecoder` case generator
  now prefer the in-memory model and fall back to the on-disk layout.
- Added a `TreeEnsembleLeafLike` backend test case (`UINT8` `nodes_modes` via `int32_data`).
- Fixed the skipped external-data tests and a latent `Normalizer` benchmark count bug.
- Fixed unit tests failing due to the latest `onnx-weekly`.
- Investigated and documented the known-missing ONNX backend tests.

### Documentation & CI

- Added a benchmark tool measuring single-model load times across `onnx`, `onnxruntime`, and `onnx_light`.
- Extended `plot_onnx_time.py`: produced three graphs (all / Python / C++), used a single argument parser, added a `ReferenceEvaluator` load benchmark, and fixed the `--model` option for custom models and external weights.
- Added a descriptive-statistics-over-ND-initializers example and persisted the ONNX graphs built by gallery examples to disk.
- Added a `--run-cpp-tests` option to `setup.py` to build and run the C++ tests, and ran the C++ tests in the Windows x86 CI job.
- Enabled the `modernize-concat-nested-namespaces` check (propagated from onnx/onnx#8150) and the clang-tidy math-parentheses and consistent-parameter-name checks (propagated from onnx/onnx#8259).
- Bumped the release version to `0.1.14`.

## [0.1.12] – 2026-08-05

### New Features

- Added `__setitem__` (index assignment) to the `RepeatedProtoField` Python bindings, so `model.graph.node[i] = node` works; negative indices and bounds checks match `__getitem__`, and the assigned element is stored as a copy.

### Improvements

- Added an opt-in `allow_external_output_allocators` option to `RuntimeSession` (C++ `RuntimeSessionOptions` and the Python constructors, with a matching read-only accessor/property) that relaxes the per-node output-allocator verification so kernels may legitimately return outputs allocated outside the session's common allocator; the strict check remains the default.
- Added a `str_<name>()` accessor to the `FIELD_STR` proto macro that returns the stored string as a `const std::string&` (or a shared empty string when unset), avoiding the `OptionalString` wrapper.
- Moved `SerializeOptions`, `ParseOptions`, and their supporting types (`TensorBufferOptions`, `FileLoadMode`, `SerializeFormat`) out of `onnx_proto/stream_class.h` into a dedicated `onnx_proto/serialize_options.h` header.

### Testing

- Asserted that untagged backend test cases carry no in-place-reuse, `shape_tag`, or release metadata, checking every case strictly by tag.

### Documentation & CI

- Split the `plot_onnx_time` benchmark chart into dedicated Python-API and C++-API graphs alongside the existing combined chart.
- Raised the Windows CI build timeout to 60 minutes.
- Bumped the release version to `0.1.12`.

## [0.1.11] – 2026-08-03

### New Features

- Added recursive `RemoveUnusedNodes`, `RemoveIdentityNodes`, and `RemoveDuplicateInitializers` passes to `GraphBuilder`.
- Added recursive duplicate-node removal (common subexpression elimination) to `GraphBuilder`.
- Added local-function inlining to `GraphBuilder`.

### Improvements

- Added an opt-in `steal` parameter to the `tensor_to_numpy` runtime binding: when set and the tensor owns its bytes inline (not allocator-backed and not a borrowed view), the buffer's ownership is transferred to NumPy through a DLPack-style capsule instead of borrowing it, so the source tensor can be released while the array lives on. `ReferenceEvaluator` uses it when converting terminal graph outputs.
- Made `RuntimeSession::Run` detach borrowed graph outputs from the model: when a declared graph output is a borrowed view into the model (e.g. a `Constant` reading its value's `raw_data` or an initializer passed straight through), the session materializes it into an owned tensor before returning (new `Tensor::ToOwned()` / `Tensor::is_borrowed()`). The output then owns its bytes and stays valid after the model is released. Exposed `Tensor.has_borrowed_data()` to Python.
- Made `ReferenceEvaluator` import standard-dtype runtime `Tensor` outputs into NumPy through the DLPack exchange protocol (`Tensor.__dlpack__` / `numpy.from_dlpack`), keeping the zero-copy conversion while relying on the standard protocol; bfloat16/float8 and sub-byte/STRING tensors keep their existing fallbacks.
- Made the C++ `RunNode` / `RuntimeSession` dispatch device-aware: a `RuntimeContext` pinned to a non-CPU device now resolves the device-qualified kernel and fails with a diagnostic naming the device when none is registered, instead of silently dispatching to the CPU kernel.
- Routed `Tensor::elem_num()` and `Tensor::size_from_dim()` through the new `safe_dim_product` helper so tensor dimension overflow and negative dimensions raise `tensor_error` (propagated from onnx/onnx#8220).
- Turned the `TensorProto` `dims` into `int64_t`.
- Made `OptionalString::value()` return a `const std::string&`.
- Improved compatibility with ONNX Runtime.
- Preserved unshaped `Scan` inputs in the `Scan` 8 → 9 version-converter adapter.
- Validated `int32_data` payload sizes in `VerifyTensor` (propagated from onnx/onnx#8211).
- Preserved signed zero for zero-point-less `FLOAT4E2M1` `QuantizeLinear`.
- Removed the remaining C++ compiler warnings under `-Wall` / `ONNX_HARDENING`.

### Testing

- Fixed the `-Wmissing-field-initializers` warning in the backend-test `IoData` initializers.

### Documentation & CI

- Bumped the release version to `0.1.11`.
- Fixed the macOS C++ release build by raising the deployment target to 13.3.
- Dropped the Python 3.10 wheels and fixed the Windows release wheel repair.
- Enabled ASan container-overflow detection in CI (propagated from onnx/onnx#8213).
- Hardened the ASan CI options with `alloc_dealloc_mismatch=1` and `abort_on_error=1` (propagated from onnx/onnx#7471).
- Disabled precompiled headers when an sccache/ccache launcher is active.
- Fixed the C++ documentation namespace (`onnx::` → `onnx_light::`).
- Refreshed the custom-kernel how-to for the current `ReferenceEvaluator` runtime APIs.
- Documented the core/extension registration design for shape, peak-memory, kernels, backend tests, and `LightOpSchema`.

## [0.1.10] – 2026-07-30

### New Features

- Added `ReferenceEvaluator.unregister_custom_kernel` to restore a built-in kernel after a custom one has been registered.

### Improvements

- Extended `ParallelFor` to the unary elementwise kernels and parallelized the `Abs` kernel execution.
- Reused `RuntimeSession` instances and made the `MatMul` kernel more cache-friendly after profiling `RuntimeSession` on `tiny_llm`.
- Fixed the O(N²) `GraphBuilder` construction from a `ModelProto`.
- Simplified `run_nodes` and removed the unused `CallModelLocalFunction` helper.
- Improved the Python bindings and the `collect_test_case` backend-test collection helpers.
- Removed the TIFF/WebP/JPEG2000 `ImageDecoder` support and the `ONNX_LIGHT_BUILD_IMAGE_CODECS` flag.

### Testing

- Added the missing `Softmax` backend test cases to match the ONNX node tests.
- Fixed the runtime coverage report to honor tolerances and flag missing reference outputs.

### Documentation & CI

- Documented how to run and verify `clang-format` in the Copilot guidelines.
- Fixed the release wheel build by dropping the unsupported `cp313t` cibuildwheel selector.
- Bumped the release version to `0.1.10`.

## [0.1.9] – 2026-07-29

### New Features

- Introduced a `RuntimeSession` that separates one-time kernel initialization from execution, and extended `ComputeContext` to orchestrate graph analyses and build the `ExecutionPlan`.
- Added an incremental `GraphBuilder` in `onnx_core/builder` with Python bindings, a `ModelProto`-to-`GraphBuilder` import path (with local-function/subgraph round-trip support), and an opt-in `check_shapes` flag on `RuntimeSession` to validate concrete against symbolic shapes.

### Improvements

- Routed kernel intermediate and output buffers through the `RuntimeContext` allocator instead of temporary `std::vector`s (`Attention`, `FlexAttention`, `Gather`, `GatherND`, `Range`, `Resize`, `Momentum`, `RegexFullMatch`, `RNN`, `LinearClassifier`, `SVMClassifier`, and a direct-to-output `Gemm`).
- Migrated many kernels to the fixed-capacity `Shape` type for rank-sized working arrays instead of `std::vector<int64_t>` (conv/pool, `Reduce*`, `Squeeze`/`Unsqueeze`, `Where`, `Pad`, `Slice`, `Compress`, `Expand`, `CenterCropPad`, `DequantizeLinear`, `QLinearConv`, `QLinearMatMul`, and the `RowMajorStrides`/`ResolveAxes` helpers).
- Cached the `Einsum` contraction plan, built `TreeEnsemble*` structures in the kernel constructors, and reused per-row/per-sample scratch in `TfIdfVectorizer` and `TreeEnsembleClassifier`.
- Migrated repeated proto-message fields from `std::vector<T>` to `RepeatedProtoField<T>` across `NodeProto`, the `GraphBuilder` inputs/outputs/initializers/attributes, and the shape-inference and schema APIs (including `TypeProto`).
- Moved `DimSum` and `IsZeroDim` into the `expressions` module and renamed the `_onnxpyoptim` extension module to `_onnxpycore`.

### Testing

- Added C++ and Python tests that run every backend model case through `RuntimeSession`, plus `GraphBuilder` round-trip coverage over all backend test cases.

### Documentation & CI

- Refreshed the design-page links and shape-inference docs to match the current library layout, showed all top-level navigation links, renamed "Operators" to "Ops" on the documentation main page, and bumped the release version to `0.1.9`.
- Enabled `sccache` on the Ubuntu CI jobs, bumped the GitHub Actions versions, and enabled ASan container-overflow detection.

## [0.1.8] – 2026-07-22

### New Features

- Added a `RuntimeParameters` class to control graph execution parallelism.
- Introduced a first-class `ExecuteAction` describing every memory-management and execution step of an `ExecutionPlan` (allocate/delete buffer, lock/unlock, transfer, execute node, create/delete shape), each tied to the owning allocator and exposed through the Python bindings with a concise `summary()` helper.
- Added peak-memory annotations to the shape-inference pipeline and a `WritePeakMemoryToMetadata` step to persist the estimate.
- Added a `Resize` opset 18 → 17 version-converter adapter.

### Improvements

- Refactored `ExecutionPlan` around the `ExecuteAction` list: every constructor now takes a `RawBufferAllocator*` and builds its actions through a single virtual extension point, and the per-node release schedule is derived entirely from that list (dropping the redundant `annotated_`/topology-fallback state) with `RunNodes` release routed through the action replay.
- Reorganized the C++ tree: moved execution primitives and annotations into `onnx_core/compute`, split `ComputeContext` into `compute_context.h`/`compute_context.cc`, moved `ExecuteActionKindName` out of the header, and switched switch-based enum→string helpers to `inline constexpr const char *`.
- Improved SVG readability by rendering edge labels smaller, staggered to reduce collisions, and without a highlight halo.

### Fixes

- Validated `int32_data` size for non-packed types in the checker (propagate onnx/onnx#8211).
- Fixed a `run_add_node_test` compile break by aligning the backend-test namespace and `TestCase` API usage.

### Testing

- Extended the big Qwen3 shape-inference case with expected intermediate shapes and embedded/verified golden in-place-reuse metadata.
- Added a `cc_release` test case covering a graph initializer in `not_used_after` metadata and per-operator benchmark coverage to the C++ backend test suite.

### Documentation & CI

- Synced the Python API docs with the current public modules, renamed the example galleries (`core` → `proto`, `optimization` → `core`), refreshed documentation, and bumped the release version to `0.1.8`.

## [0.1.7] – 2026-07-21

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
