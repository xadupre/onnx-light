# onnx-light OSS-Fuzz Harnesses

This directory contains [atheris](https://github.com/google/atheris)-based Python fuzz
targets for `onnx-light`, ported from the upstream ONNX harnesses introduced in
[onnx/onnx#8052](https://github.com/onnx/onnx/pull/8052) and adapted to
`onnx-light`'s public API surface.

## Harnesses

| File | Entry point fuzzed | Input path |
|---|---|---|
| `fuzz_checker.py` | `onnx_light.onnx.load` + `checker.check_model` | Raw bytes → protobuf parser |
| `fuzz_model_loader.py` | `onnx_light.onnx.load` + `checker.check_model` | Raw bytes → protobuf parser |
| `fuzz_parser.py` | `onnx_light.onnx.parser.parse_model` | UTF-8 text (ONNX text format) |
| `fuzz_shape_inference.py` | `onnx_light.onnx.shape_inference.infer_shapes` | Raw bytes **and** structured model (toggle byte) |
| `fuzz_optim_shape_inference.py` | `onnx_light.onnx_optim.shape_inference.infer_shapes_model` | Raw bytes **and** structured model (toggle byte) |
| `fuzz_version_converter.py` | `onnx_light.onnx.version_converter.convert_version` | Raw bytes → protobuf parser |
| `make_seed_corpus.py` | *(seed generator, not a fuzzer)* | Produces seed zips for OSS-Fuzz |

## Differences from upstream onnx harnesses

`onnx-light` exposes a slightly smaller surface than `onnx`:

* `onnx_light.checker.check_model` takes a `ModelProto` only (no raw bytes
  overload, no `full_check` flag). The checker harness therefore loads first
  and then checks.
* `onnx_light.shape_inference.infer_shapes(model)` does not accept
  `strict_mode` / `check_type` kwargs. The shape-inference harness still
  consumes a toggle byte to switch between the raw-bytes path and the
  structured-model path; the unused toggle bits are reserved for future use.
* `onnx_light.onnx.load(bytes)` replaces `onnx.load_model_from_string`.

## How OSS-Fuzz uses these files

The companion OSS-Fuzz infrastructure clones this repository and runs each
`fuzz_*.py` file via `compile_python_fuzzer`. The `build.sh` in that repo
should reference these files from `$SRC/onnx-light/onnx_light/fuzz/`.

## Running a harness locally

Atheris requires a libFuzzer-instrumented Python build; the easiest way is via
the OSS-Fuzz Docker image. For quick local smoke-tests:

```bash
pip install atheris
python -m onnx_light.fuzz.fuzz_checker -runs=1000
python -m onnx_light.fuzz.fuzz_parser -runs=1000
python -m onnx_light.fuzz.fuzz_shape_inference -runs=1000
python -m onnx_light.fuzz.fuzz_optim_shape_inference -runs=1000
python -m onnx_light.fuzz.fuzz_version_converter -runs=1000
```

To generate the seed corpora that OSS-Fuzz uses as starting inputs:

```bash
python -m onnx_light.fuzz.make_seed_corpus /tmp/vc_seeds.zip /tmp/parser_seeds.zip
```

## Continuous fuzzing in CI

The `.github/workflows/fuzz.yml` workflow runs a short smoke campaign
(`-runs=2000` per harness) on a weekly schedule (Mondays at 06:00 UTC),
on manual `workflow_dispatch`, and on pull requests that touch
`onnx_light/fuzz/**`. It is meant to catch regressions in the harnesses
themselves and obvious shallow bugs; long-running coverage-guided
campaigns are still expected to be driven by OSS-Fuzz.

## Design notes

### Why `except Exception: return`?

Fuzz targets must never crash on expected errors — only on *unexpected* ones
(memory corruption, hangs, sanitizer reports). All protobuf parse failures,
`ValidationError`, `InferenceError`, `ConvertError`, etc. are expected when
the fuzzer feeds random bytes. Swallowing them lets libFuzzer keep searching
for inputs that cause real bugs.

### Why `TestOneInput`?

`TestOneInput` is the [required entry-point name](https://github.com/google/atheris#usage)
for atheris harnesses.

### `fuzz_shape_inference.py` toggle byte

The shape inference harness exercises two code paths per iteration, selected
by the last byte of the input:

| Bit | Meaning |
|---|---|
| `0x04` | Use structured model builder (If/Loop/Scan subgraphs) instead of raw bytes |

The remaining bits are reserved for future toggles. This lets a single
harness cover both the protobuf-parser path and the recursive subgraph
visitor without needing separate fuzzers.

### Recursion limit in `fuzz_shape_inference.py`

`sys.setrecursionlimit(1000)` keeps a single deeply-nested input from
crashing the fuzzer process on every iteration, so libFuzzer can continue
finding unrelated bugs.

## Adding a new harness

1. Create `onnx_light/fuzz/fuzz_<name>.py` following the pattern of an
   existing harness.
2. If the fuzzer benefits from seed inputs, add them to `make_seed_corpus.py`
   and wire up the output zip in the OSS-Fuzz `build.sh`.
3. Open a PR here; once merged, update the OSS-Fuzz `build.sh` if a new seed
   zip was added.
