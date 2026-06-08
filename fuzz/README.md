# onnx-light OSS-Fuzz Harnesses (C++)

This directory contains [libFuzzer](https://llvm.org/docs/LibFuzzer.html)-based
C++ fuzz targets for `onnx-light`, ported from the upstream ONNX
harnesses introduced in [onnx/onnx#8052](https://github.com/onnx/onnx/pull/8052)
and rewritten directly against the onnx-light C++ API.

## Harnesses

| File                              | Entry point fuzzed                                                       | Input path                          |
|-----------------------------------|--------------------------------------------------------------------------|-------------------------------------|
| `fuzz_checker.cc`                 | `ModelProto::ParseFromString` + `checker::check_model`                   | Raw bytes → protobuf parser         |
| `fuzz_model_loader.cc`            | `ModelProto::ParseFromString` + graph walk + `checker::check_model`      | Raw bytes → protobuf parser         |
| `fuzz_parser.cc`                  | `OnnxParser::Parse<ModelProto>`                                          | UTF-8 text (ONNX text format)       |
| `fuzz_shape_inference.cc`         | `shape_inference::InferShapes`                                           | Raw bytes → protobuf parser         |
| `fuzz_optim_shape_inference.cc`   | `onnx_optim::shapes::InferShapesModel`                                   | Raw bytes → protobuf parser         |
| `fuzz_version_converter.cc`       | `version_conversion::ConvertVersion`                                     | Raw bytes → protobuf parser         |
| `make_seed_corpus.cc`             | *(seed generator, not a fuzzer)*                                         | Writes seed files for OSS-Fuzz      |

## Building

The harnesses are compiled when `ONNX_LIGHT_BUILD_FUZZERS=ON` is
passed to CMake. Clang is required because libFuzzer
(`-fsanitize=fuzzer,...`) ships with Clang.

```bash
CC=clang CXX=clang++ cmake -S . -B build-fuzz \
    -DONNX_LIGHT_BUILD_FUZZERS=ON \
    -DONNX_LIGHT_BUILD_PYTHON=OFF
cmake --build build-fuzz -j
```

The default sanitizer set is `address` (so the link line is
`-fsanitize=fuzzer,address`). Pass `-DONNX_LIGHT_FUZZER_SANITIZERS=...`
to override it; the `fuzzer` sanitizer is always added automatically.

## Running locally

Each harness is a standard libFuzzer executable:

```bash
./build-fuzz/fuzz_checker -runs=1000
./build-fuzz/fuzz_parser -runs=1000
./build-fuzz/fuzz_shape_inference -runs=1000
./build-fuzz/fuzz_optim_shape_inference -runs=1000
./build-fuzz/fuzz_version_converter -runs=1000
```

To generate seed corpora that OSS-Fuzz uses as starting inputs:

```bash
./build-fuzz/make_seed_corpus \
    /tmp/fuzz_seeds/version_converter \
    /tmp/fuzz_seeds/parser \
    /tmp/fuzz_seeds/shape_inference
```

The harnesses accept a seed-corpus directory as a positional
argument:

```bash
./build-fuzz/fuzz_shape_inference /tmp/fuzz_seeds/shape_inference -runs=1000
```

## How OSS-Fuzz uses these files

The companion OSS-Fuzz infrastructure clones this repository and
builds each `fuzz/fuzz_*.cc` target. The OSS-Fuzz `build.sh` should
configure the project with
`-DONNX_LIGHT_BUILD_FUZZERS=ON -DONNX_LIGHT_BUILD_PYTHON=OFF`, build
each `fuzz_*` target, run `make_seed_corpus`, and zip each output
directory into the matching `$OUT/fuzz_<target>_seed_corpus.zip`.

## Continuous fuzzing in CI

The `.github/workflows/fuzz.yml` workflow builds the harnesses with
Clang + libFuzzer and runs a short smoke campaign (`-runs=2000` per
harness) on a weekly schedule (Mondays at 06:00 UTC), on manual
`workflow_dispatch`, and on pull requests that touch `fuzz/**`,
`.github/workflows/fuzz.yml`, or `CMakeLists.txt`. It is meant to
catch regressions in the harnesses themselves and obvious shallow
bugs; long-running coverage-guided campaigns are still expected to be
driven by OSS-Fuzz.

## Design notes

### Why `catch (...) { return 0; }`?

Fuzz targets must never crash on expected errors — only on
*unexpected* ones (memory corruption, hangs, sanitizer reports). All
protobuf parse failures, `ValidationError`, `InferenceError`,
`ConvertError`, etc. are expected when the fuzzer feeds random bytes.
Catching them lets libFuzzer keep searching for inputs that cause
real bugs.

### Why `LLVMFuzzerTestOneInput`?

`LLVMFuzzerTestOneInput` is the [standard libFuzzer entry point](https://llvm.org/docs/LibFuzzer.html#fuzz-target).
Each harness defines it with `extern "C"` so libFuzzer's runtime can
call it directly without name mangling.

## Adding a new harness

1. Create `fuzz/fuzz_<name>.cc` following the pattern of an existing
   harness (single `extern "C" int LLVMFuzzerTestOneInput(const
   uint8_t *data, size_t size)` entry point that catches every
   exception).
2. CMake picks the new file up automatically because the fuzzer
   target list is globbed from `fuzz/fuzz_*.cc`.
3. If the fuzzer benefits from seed inputs, add them to
   `fuzz/make_seed_corpus.cc` and wire up the matching output
   directory in the OSS-Fuzz `build.sh`.
4. Open a PR here; once merged, update the OSS-Fuzz `build.sh` if a
   new seed directory was added.
