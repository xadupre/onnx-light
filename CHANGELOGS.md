# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.5] – Unreleased

### New Features

- Add `SimpleRawBufferAllocator` with virtual interface and wire into `RuntimeContext` ([#3241](https://github.com/xadupre/onnx-light/pull/3241))
- Add `RuntimeContext*` to all kernel `operator()` signatures for allocator-backed output buffers ([#3261](https://github.com/xadupre/onnx-light/pull/3261))
- Add benchmark script to export dummy Qwen3-0.6B (4 layers) with yobx transformers and plot `ComputeContext` memory ([#3344](https://github.com/xadupre/onnx-light/pull/3344))
- Add `memory_without_initializers` column to Qwen3 memory profile example ([#3446](https://github.com/xadupre/onnx-light/pull/3446))
- Plot all four requested Qwen memory configurations ([#3451](https://github.com/xadupre/onnx-light/pull/3451))

### Improvements

- Extend shape-tag metadata coverage to initializers ([#3233](https://github.com/xadupre/onnx-light/pull/3233))
- Replace `vector` fields in `Tensor` with dedicated `Shape` and `RawBuffer` types ([#3238](https://github.com/xadupre/onnx-light/pull/3238))
- Build modifications for onnx-light ([#3239](https://github.com/xadupre/onnx-light/pull/3239))
- Track allocator-owned `RawBuffer` in `Tensor` and route `RuntimeContext` tensor storage through allocator ([#3259](https://github.com/xadupre/onnx-light/pull/3259))
- Add `onnx_light.not_used_after` metadata for last-use graph inputs/initializers ([#3298](https://github.com/xadupre/onnx-light/pull/3298))
- Use `Shape` in `elementwise_helpers` for broadcast shape metadata ([#3323](https://github.com/xadupre/onnx-light/pull/3323))
- Eliminate `memcpy` in `Range` kernel preallocated-output path ([#3317](https://github.com/xadupre/onnx-light/pull/3317))
- Eliminate `memcpy` in `Random` kernels ([#3318](https://github.com/xadupre/onnx-light/pull/3318))
- Eliminate `memcpy` from `Multinomial` kernel ([#3319](https://github.com/xadupre/onnx-light/pull/3319))
- Propagate onnx/onnx#8168: update shape inference tests to use parser ([#3292](https://github.com/xadupre/onnx-light/pull/3292))
- Use `RuntimeContext` allocator for `Einsum` outputs ([#3338](https://github.com/xadupre/onnx-light/pull/3338))
- Use `Shape` in `cast_helper` tensor builder APIs ([#3332](https://github.com/xadupre/onnx-light/pull/3332))
- Make enum name helpers `constexpr` where compile-time evaluation is possible ([#3342](https://github.com/xadupre/onnx-light/pull/3342))
- Use `Shape` instead of `std::vector<int64_t>` in `Loop`/`Scan` runtime shape paths ([#3334](https://github.com/xadupre/onnx-light/pull/3334))
- Make `Constant` return borrowed tensor storage instead of copying ([#3331](https://github.com/xadupre/onnx-light/pull/3331))
- Make `NodeProto` repeated string fields compatible with `str.join` ([#3349](https://github.com/xadupre/onnx-light/pull/3349))
- Align MSVC hardening with ONNX Runtime Spectre runtime-library handling ([#3361](https://github.com/xadupre/onnx-light/pull/3361))
- Use `RuntimeContext` allocator in `float16_promote` ([#3365](https://github.com/xadupre/onnx-light/pull/3365))
- Return native `Shape` from `ConstantOfShape` shape input reader ([#3364](https://github.com/xadupre/onnx-light/pull/3364))
- Use `RuntimeContext` allocators for `Loop`/`Scan` stacked outputs ([#3363](https://github.com/xadupre/onnx-light/pull/3363))
- Cache symbolic byte-size simplification in inplace reuse analysis ([#3359](https://github.com/xadupre/onnx-light/pull/3359))
- Add `Shape::product()` to eliminate duplicated local helpers in kernel files ([#3379](https://github.com/xadupre/onnx-light/pull/3379))
- Replace `memcpy` type-punning in `cast_float8` with `std::bit_cast` ([#3384](https://github.com/xadupre/onnx-light/pull/3384))
- Use allocator-backed storage in `cast_helper` tensor builders ([#3386](https://github.com/xadupre/onnx-light/pull/3386))
- Use `Shape` in `kernel_cumsum` `SplitShape` helper signature ([#3391](https://github.com/xadupre/onnx-light/pull/3391))
- Refactor `Bernoulli` to use a shared `ElementSize` helper ([#3377](https://github.com/xadupre/onnx-light/pull/3377))
- Make `NonMaxSuppression` allocator-aware for selected-indices output ([#3407](https://github.com/xadupre/onnx-light/pull/3407))
- Extract common `BroadcastShape` helper using `Shape` ([#3420](https://github.com/xadupre/onnx-light/pull/3420))
- Construct `MatMulInteger` zero-point buffers with an explicit allocator ([#3419](https://github.com/xadupre/onnx-light/pull/3419))
- Replace `std::vector<int64_t>` with `Shape` in `AveragePool`, `MaxPool`, and `MaxUnpool` kernels ([#3421](https://github.com/xadupre/onnx-light/pull/3421))
- Make `DelayedInitializer` allocator-aware in the runtime output path ([#3422](https://github.com/xadupre/onnx-light/pull/3422))
- Use `Shape` for reduction-kernel shape paths instead of `std::vector<int64_t>` ([#3423](https://github.com/xadupre/onnx-light/pull/3423))
- Migrate internal `TopK` shape helpers to `Shape` ([#3424](https://github.com/xadupre/onnx-light/pull/3424))
- Use `Shape` instead of `std::vector<int64_t>` in `LayerNormalization` kernel ([#3432](https://github.com/xadupre/onnx-light/pull/3432))
- Use `Shape` instead of `std::vector<int64_t>` for `Einsum` kernel shape fields ([#3434](https://github.com/xadupre/onnx-light/pull/3434))
- Route `RuntimeContext` allocator through `Min`/`Max`/`Mean`/`Sum` output allocation ([#3425](https://github.com/xadupre/onnx-light/pull/3425))
- Use the runtime allocator in `Where` output allocation ([#3433](https://github.com/xadupre/onnx-light/pull/3433))
- Use `Shape` across tensor kernels ([#3439](https://github.com/xadupre/onnx-light/pull/3439))
- Use `Shape` across sequence kernels ([#3440](https://github.com/xadupre/onnx-light/pull/3440))
- Use `Shape` across math kernels ([#3441](https://github.com/xadupre/onnx-light/pull/3441))
- Use `RuntimeContext` allocator for `CumSum`/`CumProd` outputs ([#3442](https://github.com/xadupre/onnx-light/pull/3442))
- Use `Shape` temporaries in `Where` broadcast normalization ([#3445](https://github.com/xadupre/onnx-light/pull/3445))
- Use allocator-backed temporary zero-point buffers in `QuantizeLinear` ([#3447](https://github.com/xadupre/onnx-light/pull/3447))
- Replace more kernel-local `std::vector<int64_t>` shape paths with `Shape` ([#3448](https://github.com/xadupre/onnx-light/pull/3448))
- Use `RawBufferAllocator` for `TopK` output tensor allocation ([#3449](https://github.com/xadupre/onnx-light/pull/3449))
- Use `RawBufferAllocator` for fixed-size temporaries in `kernel_image_decoder.cc` ([#3450](https://github.com/xadupre/onnx-light/pull/3450))
- Use `PackedByteSize` consistently in `Range` kernel buffer sizing ([#3456](https://github.com/xadupre/onnx-light/pull/3456))
- Unify `MatMul` and `MatMulInteger` runtime output-shape computation ([#3454](https://github.com/xadupre/onnx-light/pull/3454))
- Normalize `MatMul` helper signatures to use `Shape` in `QLinearMatMul` ([#3455](https://github.com/xadupre/onnx-light/pull/3455))

### Fixes

- Fix `Transpose` inplace reuse kind is always `kGreater`, never `kEqual` ([#3235](https://github.com/xadupre/onnx-light/pull/3235))
- Fix inplace reuse missing `kGreater` for `Transpose`/`Reshape` with symbolic dimensions ([#3253](https://github.com/xadupre/onnx-light/pull/3253))
- Add shape tag annotation for model outputs ([#3255](https://github.com/xadupre/onnx-light/pull/3255))
- Fix backward tag propagation for `mask_float`/`mask_4d` through `Sub` in `tiny_llm` ([#3257](https://github.com/xadupre/onnx-light/pull/3257))
- Fix shape tag propagation for `Concat`, `Reshape`, and `Cast` ([#3263](https://github.com/xadupre/onnx-light/pull/3263))
- Propagate onnx#8114: validate `Scan` `num_scan_inputs` to prevent shape-inference underflow ([#3293](https://github.com/xadupre/onnx-light/pull/3293))
- Correct `Unsqueeze` inplace reuse annotation when shape proof is unavailable ([#3296](https://github.com/xadupre/onnx-light/pull/3296))
- Fix `If::operator()` — avoid extra allocation and `memcpy`, propagate full `RuntimeContext` to branches ([#3302](https://github.com/xadupre/onnx-light/pull/3302))
- Fix `Bernoulli` kernel: use allocator from `RuntimeContext` for output tensor ([#3306](https://github.com/xadupre/onnx-light/pull/3306))
- Fix `EyeLike`: inplace writes directly to output and non-inplace delegates to inplace ([#3330](https://github.com/xadupre/onnx-light/pull/3330))
- Fix verbose mode in `compute_inplace_reuse_graph` and `compute_release_after_shape_tagged` ([#3353](https://github.com/xadupre/onnx-light/pull/3353))
- Avoid temporary probability buffering in the `Bernoulli` kernel ([#3355](https://github.com/xadupre/onnx-light/pull/3355))
- Remove the remaining `memcpy` path from the `EyeLike` kernel ([#3354](https://github.com/xadupre/onnx-light/pull/3354))
- Fix `Range` shape inference to propagate existing symbolic dims instead of emitting `Range_dim0` ([#3371](https://github.com/xadupre/onnx-light/pull/3371))
- Fix expression simplification to combine like terms with integer coefficients in multi-factor products ([#3373](https://github.com/xadupre/onnx-light/pull/3373))
- Fix `ConstantOfShape` kernel to use the allocator from `RuntimeContext` ([#3435](https://github.com/xadupre/onnx-light/pull/3435))
- Fix SVG barycenter crossing minimisation to use adjacent-layer edges only ([#3427](https://github.com/xadupre/onnx-light/pull/3427))
- Fix `CastLike` ignoring its `RuntimeContext` allocator ([#3437](https://github.com/xadupre/onnx-light/pull/3437))
- Fix `AffineGrid` to use an allocator-backed `Tensor` in `NormalisedCoords` ([#3436](https://github.com/xadupre/onnx-light/pull/3436))
- Fix `MakeOutputTensor` allocator propagation across all multi-output kernels ([#3438](https://github.com/xadupre/onnx-light/pull/3438))
- Fix shape-tag seeding to always tag graph inputs/outputs/initializers as `weight` ([#3431](https://github.com/xadupre/onnx-light/pull/3431))
- Fix `BatchNormalization::TrainingForward` allocator propagation to the `y` output ([#3444](https://github.com/xadupre/onnx-light/pull/3444))
- Fix `MelWeightMatrix` to eliminate an intermediate heap allocation ([#3452](https://github.com/xadupre/onnx-light/pull/3452))
- Fix `elementwise_helpers.h` to use `Shape` instead of `std::vector<int64_t>` for `idx` ([#3457](https://github.com/xadupre/onnx-light/pull/3457))

### Security

- Fix `size_t` underflow in `Scan` shape inference (GHSA-qrhj-v62m-vmpf) ([#3294](https://github.com/xadupre/onnx-light/pull/3294))
- Harden zero-copy parse path for ORT format guardrails ([#3268](https://github.com/xadupre/onnx-light/pull/3268))
- Harden advisory-related tar extraction path validation ([#3270](https://github.com/xadupre/onnx-light/pull/3270))
- Validate `graphviz` format argument to prevent command injection ([#3286](https://github.com/xadupre/onnx-light/pull/3286))
- Replace insecure temporary file creation in path security test ([#3287](https://github.com/xadupre/onnx-light/pull/3287))
- Prevent Actions code injection in `clang_tidy.yml` ([#3288](https://github.com/xadupre/onnx-light/pull/3288))
- Prevent Actions expression injection in release wheel workflows ([#3289](https://github.com/xadupre/onnx-light/pull/3289))
- Address GHSA-3jf9-582g-jjmq — bounds and file-size validation for external data ([#3320](https://github.com/xadupre/onnx-light/pull/3320))
- Harden external data loading against hardlink and symlink attacks (GHSA-xrch-8vh7-h656) ([#3322](https://github.com/xadupre/onnx-light/pull/3322))
- Raise upstream ONNX minimum/default to 1.21.0 for GHSA-p893-rvq9-2xf9 ([#3324](https://github.com/xadupre/onnx-light/pull/3324))
- Investigated GHSA-hqmj-h5c6-369m (`onnx.hub.load` silent-bypass): onnx-light has never implemented a hub module and is not affected ([#3274](https://github.com/xadupre/onnx-light/issues/3274))
- Investigated GHSA-8qff-7g33-75mx (TOCTOU arbitrary file read/write in `save_external_data`): onnx-light does not expose a Python `save_external_data` function; saving routes through C++ with Python-level pre-validation (`validate_external_data_path`) that rejects symlinks and hardlinks at the target path. Added defense-in-depth symlink checks at the C++ write layer (`FileWriteStream`, `validate_weights_file_is_next_to_model`, `validate_external_location_is_next_to_model`) to close the residual TOCTOU window, and regression tests to cover this path ([#3271](https://github.com/xadupre/onnx-light/issues/3271))

### Testing

- Add Loop early-exit shape inference regression test (onnx/onnx#8146) ([#3290](https://github.com/xadupre/onnx-light/pull/3290))
- Add `ScatterND` element-level index test cases for `max`/`min` reductions (onnx#8099) ([#3291](https://github.com/xadupre/onnx-light/pull/3291))
- Enforce full metadata tagging coverage in shape-tag backend tests ([#3300](https://github.com/xadupre/onnx-light/pull/3300))
- Add C++ backend coverage for dilated `VALID` `AveragePool` output shape ([#3343](https://github.com/xadupre/onnx-light/pull/3343))
- Add C++ backend test for `ValueAsShape` propagation through a local-function `Range` boundary ([#3351](https://github.com/xadupre/onnx-light/pull/3351))
- Add opset-18 `AvgPool` backend regressions for ORT PR 29629 ([#3362](https://github.com/xadupre/onnx-light/pull/3362))
- Add backend test for `Conv` `SAME_UPPER` `auto_pad` with stride > 1 (onnxruntime#26734) ([#3367](https://github.com/xadupre/onnx-light/pull/3367))
- Add `qwen3_4_layers_like` C++ shape-inference backend test case ([#3369](https://github.com/xadupre/onnx-light/pull/3369))
- Add missing tiny-llm `Unsqueeze` inplace metadata expectations ([#3429](https://github.com/xadupre/onnx-light/pull/3429))
- Add more options to the backend tests ([#3453](https://github.com/xadupre/onnx-light/pull/3453))
- Delay C++ backend-test `ModelProto` construction with `TestCase::emplace_model()` across registered cases ([#3458](https://github.com/xadupre/onnx-light/pull/3458))

## [0.1.4] – 2026-07-07

### Fixes

- Fix Win32 narrowing in backend-test DLPack shape conversion ([#3223](https://github.com/xadupre/onnx-light/pull/3223))
- Suppress GCC 13 false-positive `-Wfree-nonheap-object` in `ComputeScaleIndex` ([#3227](https://github.com/xadupre/onnx-light/pull/3227))

### Security

- Validate `raw_data` alignment in `ParseData` to prevent out-of-bounds copy (propagate ONNX #8032) ([#3225](https://github.com/xadupre/onnx-light/pull/3225))

## [0.1.3] – 2026-07-06

### New Features

- Add `SerializeOptions.max_serialized_size_bytes` to hard-cap serialization output size ([#3207](https://github.com/xadupre/onnx-light/pull/3207))
- Show node indexes in `pretty_onnx` graph output ([#3031](https://github.com/xadupre/onnx-light/pull/3031))
- Add `--token` support to `fillshape` for binding symbolic dimension tokens to ranges ([#3028](https://github.com/xadupre/onnx-light/pull/3028))
- Add `include_release` option to `to_svg` and `to_mermaid` ([#3020](https://github.com/xadupre/onnx-light/pull/3020))
- Add `fillshape --release-info` to emit release metadata independently ([#3019](https://github.com/xadupre/onnx-light/pull/3019))
- Add verbose progress output for `fillshape` execution ([#3012](https://github.com/xadupre/onnx-light/pull/3012))
- Expose random input generation as `onnx_light.onnx.tools.make_random_input` ([#3009](https://github.com/xadupre/onnx-light/pull/3009))
- Add `Logger` to `onnx_helpers` ([#2999](https://github.com/xadupre/onnx-light/pull/2999))
- Add `dim_ranges_from_expressions` to infer dimension ranges from equality constraints ([#2988](https://github.com/xadupre/onnx-light/pull/2988))
- Add C++/nanobind shape, axes, and weight metadata tagging and tag-aware Mermaid/SVG rendering ([#2959](https://github.com/xadupre/onnx-light/pull/2959))
- Add runtime progress output for verbose `ReferenceEvaluator` execution ([#2961](https://github.com/xadupre/onnx-light/pull/2961))
- Add `run` subcommand to generate random inputs and execute a model from the CLI ([#2956](https://github.com/xadupre/onnx-light/pull/2956))
- Add `--verbose [LEVEL]` support to `fillshape` CLI ([#2952](https://github.com/xadupre/onnx-light/pull/2952))
- Add ChaCha20-Poly1305 support for encrypted model I/O (`ONNXCRY2`) ([#2948](https://github.com/xadupre/onnx-light/pull/2948))
- Add light-only `ai.rt` `DelayedInitializer` schema, runtime kernel, and shape inference ([#2940](https://github.com/xadupre/onnx-light/pull/2940))
- Add pre-serialization weight rewrite callback for model save paths ([#2944](https://github.com/xadupre/onnx-light/pull/2944))
- Add `python -m onnx_light fillshape` command ([#2938](https://github.com/xadupre/onnx-light/pull/2938))
- Implement `ParseFromIstream` for all proto classes ([#2942](https://github.com/xadupre/onnx-light/pull/2942))
- Add support for a callback function when parsing a model ([#2928](https://github.com/xadupre/onnx-light/pull/2928))
- Add `include_inplace` option to `to_mermaid`/`to_svg` renderers ([#2930](https://github.com/xadupre/onnx-light/pull/2930))
- Add function to record in-place reuse opportunities into node metadata ([#2918](https://github.com/xadupre/onnx-light/pull/2918))
- Guard graph inputs from in-place overwrite with an opt-in flag ([#2916](https://github.com/xadupre/onnx-light/pull/2916))
- Add ChaCha20 `raw_data` callback example for weight serialization/parsing ([#2957](https://github.com/xadupre/onnx-light/pull/2957))

### Improvements

- Update onnxruntime integration and related compatibility changes ([#3205](https://github.com/xadupre/onnx-light/pull/3205))
- Mirror upstream onnx#8157: fix nanobind Python target resolution in cross-compile CMake path ([#3217](https://github.com/xadupre/onnx-light/pull/3217))
- Add C++ backend test for `kRelease` event in shape inference ([#3105](https://github.com/xadupre/onnx-light/pull/3105))
- Align floor-division simplification with Python `//` semantics ([#3026](https://github.com/xadupre/onnx-light/pull/3026))
- Propagate Reshape shape-tag metadata backward through producer chains ([#3024](https://github.com/xadupre/onnx-light/pull/3024))
- Reuse the core shape-inference API for `fillshape` shape-tag metadata writing ([#3022](https://github.com/xadupre/onnx-light/pull/3022))
- Seed `weight` tags for rank-2 `FLOAT` graph inputs in shape-tag inference ([#3021](https://github.com/xadupre/onnx-light/pull/3021))
- Fix empty-axis broadcasting and add half-precision `RMSNormalization` support ([#3018](https://github.com/xadupre/onnx-light/pull/3018))
- Extend `ComputeContext` with shape-tag release info ([#3017](https://github.com/xadupre/onnx-light/pull/3017))
- Allow Transpose in-place reuse detection when storage size is equal ([#3015](https://github.com/xadupre/onnx-light/pull/3015))
- Move tiny external-tensor inlining from `fillshape` into `ParseOptions` ([#3011](https://github.com/xadupre/onnx-light/pull/3011))
- Move `inplace_reuse` and value-tag helpers into `onnx_optim.annotations` ([#3007](https://github.com/xadupre/onnx-light/pull/3007))
- Show release, in-place, and shape-tag annotations in `fillshape --show` output ([#3005](https://github.com/xadupre/onnx-light/pull/3005))
- Fix symbolic shape inference for `Reshape` shapes built from `Unsqueeze(Gather(Shape(...)))` ([#3003](https://github.com/xadupre/onnx-light/pull/3003))
- Simplify nested floor divisions such as `x//5//2` into `x//10` ([#3001](https://github.com/xadupre/onnx-light/pull/3001))
- Slice shape inference now emits symbolic expressions instead of fresh names ([#2954](https://github.com/xadupre/onnx-light/pull/2954))
- Simplify divisible additive offsets for floor and exact division in expressions ([#2950](https://github.com/xadupre/onnx-light/pull/2950))
- Improve `to_svg` layout: shorten edges and reduce crossings ([#2932](https://github.com/xadupre/onnx-light/pull/2932))
- Accept arbitrary iterables in `RepeatedField.extend`/`__init__` ([#2934](https://github.com/xadupre/onnx-light/pull/2934))
- Implement deleter in `ByteSpan` and `TensorProto` ([#2926](https://github.com/xadupre/onnx-light/pull/2926))
- Refactor `ComputeInPlaceReuse` into a `ComputeContext` class ([#2924](https://github.com/xadupre/onnx-light/pull/2924))

## [0.1.1] – 2026-06-22

### New Features

- Add a pure-Python textproto parser/serializer; support `format="textproto"` in `load`/`save` ([#2903](https://github.com/xadupre/onnx-light/pull/2903))
- Add `/:`  exact division operator to shape expressions and update Reshape shape inference ([#2913](https://github.com/xadupre/onnx-light/pull/2913))
- Add `compare_expressions` to compare two symbolic expressions ([#2836](https://github.com/xadupre/onnx-light/pull/2836))
- Add `WhichOneof` to `TypeProto` and `OptionalProto` bindings ([#2816](https://github.com/xadupre/onnx-light/pull/2816))
- Add protobuf-style keyword-argument constructor to proto classes ([#2851](https://github.com/xadupre/onnx-light/pull/2851))
- Support protobuf-style keyword arguments in `RepeatedProtoField.add()` ([#2814](https://github.com/xadupre/onnx-light/pull/2814))
- Add `String.decode()` to mimic `bytes.decode` for `STRING`/`STRINGS` attribute values ([#2859](https://github.com/xadupre/onnx-light/pull/2859))
- Support `int2`/`uint2`/`int4`/`uint4`/`float4` in `make_tensor` ([#2857](https://github.com/xadupre/onnx-light/pull/2857))
- Expose `OpSchema.Attribute.default_value` to match ONNX API ([#2840](https://github.com/xadupre/onnx-light/pull/2840))
- Add indentation and `inline_threshold` options to `PrintOptions` ([#2883](https://github.com/xadupre/onnx-light/pull/2883))
- Publish wheel files directly to GitHub Releases on tag runs ([#2829](https://github.com/xadupre/onnx-light/pull/2829))
- Make Attention `is_causal` masking offset-aware (bottom-right) for external KV cache ([#2906](https://github.com/xadupre/onnx-light/pull/2906))

### Improvements

- Bring `infer_shapes` to parity with onnx (`check_type`, `strict_mode`, `data_prop`) ([#2853](https://github.com/xadupre/onnx-light/pull/2853))
- Register repeated field containers as `collections.abc.Sequence` ([#2847](https://github.com/xadupre/onnx-light/pull/2847))
- Support comparing scalar `RepeatedField` to a Python list ([#2855](https://github.com/xadupre/onnx-light/pull/2855))
- Return `False` for `String` comparisons against non-string objects ([#2861](https://github.com/xadupre/onnx-light/pull/2861))
- Guess in-place input reuse from `onnx_optim` shape inference ([#2826](https://github.com/xadupre/onnx-light/pull/2826))
- Reduce Windows and macOS wheel size by excluding the C++ install tree ([#2820](https://github.com/xadupre/onnx-light/pull/2820))
- Bump mirrored ONNX version to 1.23.0 ([#2908](https://github.com/xadupre/onnx-light/pull/2908))
- Enable ONNX backend optional-sequence loop case in `ReferenceEvaluator` ([#2904](https://github.com/xadupre/onnx-light/pull/2904))
- Add `Where` backend cases for all supported dtypes and fix `BOOL` x/y path ([#2888](https://github.com/xadupre/onnx-light/pull/2888))
- Validate `ConvTranspose` group divisibility in shape inference ([#2910](https://github.com/xadupre/onnx-light/pull/2910))
- Fix shape inference on graph values without a type field ([#2890](https://github.com/xadupre/onnx-light/pull/2890))
- Fix `SequenceInsert` to accept single-element `[1]`-shaped position ([#2900](https://github.com/xadupre/onnx-light/pull/2900))
- Run ONNX `sequence_map` tests through the reference evaluator ([#2897](https://github.com/xadupre/onnx-light/pull/2897))
- Replace `throw std::invalid_argument` with `EXT_ENFORCE_INVALID` / `EXT_THROW_INVALID` ([#2827](https://github.com/xadupre/onnx-light/pull/2827))
- Use `EXT_ENFORCE_INVALID` for guard-throw blocks in kernels ([#2881](https://github.com/xadupre/onnx-light/pull/2881))
- Fix dropped `TensorProto.segment` field in C++ serialization ([#2869](https://github.com/xadupre/onnx-light/pull/2869))
- Fix `InferenceSessionAllTypes.run` return type ([#2863](https://github.com/xadupre/onnx-light/pull/2863))
- Fix onnx-light proto API gaps to pass ir-py test suite ([#2834](https://github.com/xadupre/onnx-light/pull/2834))
- Implement sub-byte IOBinding for `Cast FLOAT↔INT2` ([#2865](https://github.com/xadupre/onnx-light/pull/2865))
- Support low-precision dtypes in ORT backend tests using IOBinding ([#2796](https://github.com/xadupre/onnx-light/pull/2796))
- Propagate Attention attribute docstring sync (`is_causal`, `qk_matmul_output_mode`) from onnx/onnx#8068 ([#2256](https://github.com/xadupre/onnx-light/pull/2256))
- Pass non-null pointer params by reference in internal C++ helpers (propagate onnx#8105) ([#2912](https://github.com/xadupre/onnx-light/pull/2912))
- Add a build tag to the reduced wheel so its filename differs from the full wheel ([#2838](https://github.com/xadupre/onnx-light/pull/2838))

### Testing

- Add unit tests for proto/numpy helper functions ([#2873](https://github.com/xadupre/onnx-light/pull/2873))
- Add unit tests for `get_total_memory_gb`, `get_cpu_topology`, `_schema_to_rst`, and `hide_stdout` ([#2867](https://github.com/xadupre/onnx-light/pull/2867))
- Add unit tests for `RunLoopWithSequenceState` ([#2879](https://github.com/xadupre/onnx-light/pull/2879))
- Add unit tests for proto field serialization helpers ([#2877](https://github.com/xadupre/onnx-light/pull/2877))
- Add unit tests for `write_as_string` print helpers ([#2875](https://github.com/xadupre/onnx-light/pull/2875))
- Add unit tests for sparse/sequence/optional helper functions ([#2871](https://github.com/xadupre/onnx-light/pull/2871))
- Add C++ serialization tests ([#2869](https://github.com/xadupre/onnx-light/pull/2869))
- Add shape inference test for MatMul with an initializer weight ([#2863](https://github.com/xadupre/onnx-light/pull/2863))
- Add shape inference examples with two sequential TopK nodes ([#2831](https://github.com/xadupre/onnx-light/pull/2831))
- Add manual CI job validating onnx-light against ir-py and mbext ([#2824](https://github.com/xadupre/onnx-light/pull/2824))
- Add manually-triggered CI workflow to run yobx tests against onnx-light ([#2845](https://github.com/xadupre/onnx-light/pull/2845))

## [0.1.0] – 2026-06-18

Initial public release.
