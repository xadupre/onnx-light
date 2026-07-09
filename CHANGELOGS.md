# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.5] – Unreleased

### Security

- Investigated GHSA-hqmj-h5c6-369m (`onnx.hub.load` silent-bypass): onnx-light has never implemented a hub module and is not affected ([#3274](https://github.com/xadupre/onnx-light/issues/3274))

## [0.1.4] – 2026-07-07

### Fixes

- Fix Win32 narrowing in backend-test DLPack shape conversion ([#3223](https://github.com/xadupre/onnx-light/pull/3223))

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
