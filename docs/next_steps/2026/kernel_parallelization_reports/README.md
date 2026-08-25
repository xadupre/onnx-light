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
| `x86_64_calibration.json` | x86-64 (AMD EPYC 7763) | Step G calibration report; see below. |

An ARM64 report has not been published yet: this repository's automation
does not currently have access to ARM64 hardware. Add `arm64_baseline.json`
and `arm64_calibration.json` here, generated with the same commands, once
such access is available.

## Calibration reports

`x86_64_calibration.json` is produced by calibrating and persisting every
calibratable kernel tuning key missing from a tuning cache with:

`onnx_light.kernel_tuning.apply_kernel_tuning_updates(path=...)`.

It then reloads that cache path in a separate process
(`onnx_light.kernel_tuning.load_kernel_tuning_cache` /
`kernel_tuning_parameters`) and records the reload outcome under
`reload_verification`: `loaded_keys` must equal `calibratable_keys`,
`incompatible_keys` and `invalid_keys` must be `0`, and
`published_profiles_resolved` must equal `loaded_keys`, proving the
persisted profile matches the exact processor and execution descriptor a
default-policy session resolves without recalibrating or touching the
tuning registry beyond the one explicit load. `calibrated_profiles` and
`diagnostics` record each selected value and the reasoning message reported
by the kernel's calibration callback. None of these calibrated values were
promoted to the portable schema defaults in this pass (see Step G in
`2026-08_kernel_parallelization.rst`): only a single x86-64 machine was
available, so promoting them without a matching ARM64 measurement would
risk an undeclared regression on that architecture. The persisted cache
profile itself already lets this machine use the calibrated values ahead of
any default change, without affecting other processors.
