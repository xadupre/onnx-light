# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Collects native kernel evidence and verifies persisted profiles in separate processes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from onnx_light.onnx_py._onnxpykernels import runtime  # type: ignore[import]
from onnx_light.tools.kernel_baseline import run_kernel_baseline_report

KEY_FIELDS = ("library", "kernel", "implementation", "element_type", "device", "tuning_abi")


def profile_key(profile):
    """Returns the complete registry key, including the tuning ABI."""
    return tuple(profile[field] for field in KEY_FIELDS)


def write_report(path, report):
    """Writes a machine-readable report without dropping native diagnostics."""
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(condition, message):
    """Raises when measurement evidence fails validation."""
    if not condition:
        raise ValueError(message)


def baseline(threads, repeat):
    """Measures the fixed corpus under serial and explicit session policies."""
    return run_kernel_baseline_report(
        cpu_policies=(("serial", 1), ("session_thread", threads)), repeat=repeat, warmup=2, seed=0
    )


def collect(output, threads, repeat, duration_ms, memory_bytes):
    """Collects a clean baseline and calibrates every available CPU key."""
    output.mkdir(parents=True, exist_ok=False)
    cache = str(output / "kernel_tuning.cache")
    policy = runtime.CpuExecutionPolicy()
    policy.num_threads = threads
    resolved = runtime.resolve_cpu_execution_policy(policy)
    check(resolved.effective_threads == threads, "Requested thread count was not admitted")
    inventory = runtime.kernel_tuning_parameters(path=cache, num_threads=threads)["kernels"]
    check(
        all(item["active_source"] == "portable_default" for item in inventory),
        "Baseline requires an empty tuning cache and a fresh process",
    )
    metadata = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "execution": {
            "requested_threads": threads,
            "effective_threads": resolved.effective_threads,
            "spin_policy": str(policy.spin_policy),
            "spin_budget": policy.spin_budget,
            "resolved_spin": {
                "policy": str(resolved.spin.policy),
                "iterations": resolved.spin.iterations,
                "duration_ns": resolved.spin.duration_ns,
            },
            "affinity_policy": str(policy.affinity_policy),
            "allow_nested_parallelism": resolved.allow_nested_parallelism,
            "uses_smt": resolved.uses_smt,
            "uses_efficiency_cores": resolved.uses_efficiency_cores,
            "diagnostics": resolved.diagnostics,
        },
        "measurement": {
            "repeat": repeat,
            "warmup": 2,
            "seed": 0,
            "maximum_duration_ms": duration_ms,
            "maximum_memory_bytes": memory_bytes,
        },
        "registry": inventory,
    }
    write_report(output / "metadata.json", metadata)
    write_report(output / "baseline.json", baseline(threads, repeat))
    selected = [item for item in inventory if item["calibratable"]]
    check(selected, "No calibratable CPU keys registered")
    profiles = []
    for index, item in enumerate(selected):
        report = runtime.calibrate_kernel_tuning(
            item["kernel"],
            element_types=[item["element_type"]],
            library=item["library"],
            implementation=item["implementation"],
            device=item["device"],
            cpu_execution=policy,
            maximum_duration_ms=duration_ms,
            maximum_memory_bytes=memory_bytes,
            profiling_capacity=128,
            profiling_hardware_counters=False,
            save=True,
            path=cache,
        )
        write_report(output / f"candidate-{index:03d}.json", report)
        check(
            [profile_key(p) for p in report["calibrated"]] == [profile_key(item)],
            f"Incomplete or ambiguous calibration: {profile_key(item)}",
        )
        check(report["cache_update"]["status"] == "updated", "Profile persistence failed")
        for diagnostic in report["candidate_diagnostics"]:
            for event in diagnostic["events"]:
                check(
                    event["observed_threads"] <= event["admitted_threads"] <= threads,
                    f"Thread budget exceeded: {event}",
                )
        profiles.extend(report["calibrated"])
    write_report(
        output / "calibration.json",
        {
            "source_revision": metadata["source_revision"],
            "calibratable_keys": len(selected),
            "calibrated_profiles": profiles,
        },
    )


def verify(output, incompatible=False):
    """Checks schema/ABI and descriptor selection without recalibrating."""
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    threads = metadata["execution"]["effective_threads"]
    if incompatible:
        threads = 1 if threads != 1 else 2
    cache = str(output / "kernel_tuning.cache")
    loaded = runtime.load_kernel_tuning_cache(path=cache, num_threads=threads)
    write_report(output / ("incompatible-reload.json" if incompatible else "reload.json"), loaded)
    check(loaded["status"] == "loaded", "Cache could not be read")
    check(not loaded["invalid"] and not loaded["stale"], "Invalid schema or tuning ABI")
    expected = {profile_key(p): p for p in report["calibrated_profiles"]}
    check(
        len(expected) == len(report["calibrated_profiles"]) == report["calibratable_keys"]
        and bool(expected),
        "Empty, duplicate or incomplete published profiles",
    )
    active = runtime.kernel_tuning_parameters(path=cache, num_threads=threads)["kernels"]
    if incompatible:
        check(not loaded["loaded"], "Incompatible execution descriptor was loaded")
        check(
            {profile_key(p) for p in loaded["incompatible"]} == set(expected),
            "Not all mismatched profiles were rejected",
        )
        check(
            all(p["active_source"] == "portable_default" for p in active),
            "Incompatible profiles became active",
        )
        return
    check(not loaded["incompatible"], "Native processor/execution descriptor mismatch")
    check({profile_key(p) for p in loaded["loaded"]} == set(expected), "Incomplete reload")
    selected = [item for item in active if profile_key(item) in expected]
    check(len(selected) == len(expected), "Registered profiles missing from inspection")
    for item in selected:
        key = profile_key(item)
        check(set(item["parameter_names"]) == set(expected[key]["values"]), "Schema mismatch")
        check(item["active_source"] == "published_profile", f"Profile not selected: {key}")
        check(item["active_values"] == expected[key]["values"], f"Values differ: {key}")
    report["reload_verification"] = {
        "load_status": loaded["status"],
        "loaded_keys": len(loaded["loaded"]),
        "published_profiles_resolved": len(selected),
        "incompatible_keys": len(loaded["incompatible"]),
        "invalid_keys": len(loaded["invalid"]),
    }
    write_report(output / "calibration.json", report)
    tuned = baseline(threads, metadata["measurement"]["repeat"])
    write_report(output / "tuned-baseline.json", tuned)
    original = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    comparisons = []
    fields = ("op_type", "element_type", "shape_label", "size", "cpu_policy")
    for before, after in zip(original["benchmarks"], tuned["benchmarks"], strict=True):
        check(all(before[f] == after[f] for f in fields), "Benchmark corpus changed")
        comparisons.append(
            {
                **{field: before[field] for field in fields},
                "baseline_seconds": before["median_kernel_execution_seconds"],
                "tuned_seconds": after["median_kernel_execution_seconds"],
                "same_machine_speedup": (
                    before["median_kernel_execution_seconds"]
                    / after["median_kernel_execution_seconds"]
                ),
            }
        )
    write_report(output / "comparison.json", comparisons)


def main():
    """Runs one measurement phase; verification must use a fresh process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "verify", "verify-incompatible"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--maximum-duration-ms", type=int, default=10000)
    parser.add_argument("--maximum-memory-bytes", type=int, default=268435456)
    args = parser.parse_args()
    check(args.threads > 0 and args.repeat > 0, "Threads and repetitions must be positive")
    check(
        args.maximum_duration_ms > 0 and args.maximum_memory_bytes > 0,
        "Calibration budgets must be positive",
    )
    if args.phase == "collect":
        collect(
            args.output.resolve(),
            args.threads,
            args.repeat,
            args.maximum_duration_ms,
            args.maximum_memory_bytes,
        )
    else:
        verify(args.output.resolve(), incompatible=args.phase == "verify-incompatible")


if __name__ == "__main__":
    main()
