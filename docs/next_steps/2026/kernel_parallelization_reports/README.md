# Kernel parallelization baseline reports

Machine-readable baselines produced by `python -m onnx_light kernel-baseline`
(see `docs/next_steps/2026/2026-08_kernel_parallelization.rst`). Each file
combines the Step D kernel-coverage inventory with Step E benchmark corpus
results for one machine; regenerate with:

```
python -m onnx_light kernel-baseline --output <arch>_baseline.json
```

The command is read-only: it never invokes `onnxruntime` and never writes to
the kernel tuning cache.

| File | Architecture | Notes |
| --- | --- | --- |
| `x86_64_baseline.json` | x86-64 (AMD EPYC 7763) | Generated in the CI sandbox used to develop this tool. |

An ARM64 report has not been published yet: this repository's automation
does not currently have access to ARM64 hardware. Add `arm64_baseline.json`
here, generated with the same command, once such access is available.
