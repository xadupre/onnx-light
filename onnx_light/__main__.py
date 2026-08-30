"""Entry point for ``python -m onnx_light``.

Supported subcommands
---------------------
fillshape
    Fills a model's ``graph.value_info`` and ``graph.output`` with the shapes
    inferred by ``onnx_shapes`` shape inference.

    Usage::

        python -m onnx_light fillshape model.onnx [options]

    Options:

    ``--output OUTPUT`` / ``-o OUTPUT``
        Write the result to *OUTPUT* instead of overwriting the input file.
        When the model stores weights in a separate file (external data), the
        output is placed in the **same directory as the input model** so that
        the relative weight-file paths encoded in the proto remain valid.  The
        weight file itself is never written again.
    ``--keep``
        Seed the inference context from any shapes already present in
        ``graph.value_info`` / ``graph.output`` (i.e. call
        ``infer_shapes_model`` with ``prefill_with_value_info_output=True``).
        Existing non-conflicting shapes are preferred over newly inferred
        ones.
    ``--inplace-info``
        After shape inference, also compute in-place buffer-reuse
        opportunities and write them into each node's ``metadata_props``
        under the key ``onnx_light.inplace_reuse``.
    ``--release-info``
        After shape inference, compute last-use release hints and write
        them into each node's ``metadata_props`` under the key
        ``onnx_light.release_after``.
    ``--shape-tag``
        After shape inference, infer semantic ``shape``/``axes``/``weight``/``ambiguous``
        tags for every value and node in the graph and record them in
        ``metadata_props`` (per-value key ``onnx_light.value_tag`` and
        per-node key ``onnx_light.node_tag``).
    ``--token NAME=LOW:HIGH``
        Bind a symbolic dimension token to an inclusive integer range before
        running shape inference.  May be specified multiple times.
        For example, ``--token seq=1:128`` treats ``seq`` as having range
        ``[1, 128]`` (the lower bound is used for shape propagation).
        Symbolic dims not covered by ``--token`` remain symbolic.
    ``--show``
        Print the inferred shapes to stdout; do **not** save the model.
    ``--verbose [LEVEL]``
        Print shape-inference progress information. With no level, defaults
        to ``1`` (summary). With ``2``, also prints per-event details.

show
    Prints a human-readable, Mermaid or SVG rendering of an ONNX model.

    Usage::

        python -m onnx_light show model.onnx [options]

    Options:

    ``--format FORMAT`` / ``-f FORMAT``
        Output format: ``pretty`` (default), ``mermaid``, ``svg``, ``dot``,
        ``onnx-compact``, ``builder`` or ``cpp``.  The last three emit code that
        rebuilds the model (see :mod:`onnx_light.tools.translate`).
    ``--output OUTPUT`` / ``-o OUTPUT``
        Write the rendered text to *OUTPUT* instead of printing to stdout.
    ``--shape-inference``
        Run onnx-light shape inference on the model before rendering.
    ``--include-shapes / --no-shapes``
        Include (or suppress) shape annotations in the rendered output.
        Enabled by default; only has effect for the ``mermaid`` and ``svg``
        formats.
    ``--include-attributes``
        Include node attributes in the rendered output.
    ``--include-inplace``
        Show in-place buffer-reuse annotations (``onnx_light.inplace_reuse``
        metadata).
    ``--include-release``
        Show post-execution release hints (``onnx_light.release_after``
        metadata).  Only used by the ``pretty`` format.
    ``--include-node-tags``
        Show semantic ``shape``/``axes``/``weight``/``ambiguous`` node-tag annotations
        (``onnx_light.node_tag`` metadata).  Only used by the ``pretty``
        format.
    ``--no-initializers``
        Exclude initializer nodes from the rendered graph.  Only used by the
        ``mermaid`` and ``svg`` formats.
    ``--direction DIRECTION``
        Flowchart direction for the ``mermaid`` and ``svg`` formats.
        One of ``TB`` (top-to-bottom, default), ``LR`` (left-to-right),
        ``TD`` or ``BT`` (``mermaid`` only).
    ``--layout LAYOUT``
        Box positioning strategy for the ``svg`` format.  One of
        ``layered`` (default) or ``umap`` (requires ``umap-learn``).

run
    Generates random inputs and runs a model through the onnx-light runtime.

    Usage::

        python -m onnx_light run model.onnx [options]

    Options:

    ``--dim NAME=VALUE``
        Override a symbolic (dynamic) dimension by name.  May be specified
        multiple times.  For example, ``--dim batch=4 --dim seq=16`` sets the
        symbolic dims named ``batch`` and ``seq`` to 4 and 16 respectively.
        Dimensions that are neither concrete nor covered by ``--dim`` fall
        back to 1.
    ``--seed SEED``
        Integer seed for the deterministic pseudo-random input generator.
        Defaults to 0.
    ``--verbose [LEVEL]`` / ``-v [LEVEL]``
        Print run progress information. With no level, defaults to ``1``
        (summary: loading, input shapes, output shapes). With ``2``, also
        prints per-node execution details.
    ``--dump PATH``
        Write all inputs and outputs to *PATH* as an ONNX model whose
        ``graph.initializer`` list contains one ``TensorProto`` per input
        and per output tensor (in that order).  Only ``numpy.ndarray``
        values are stored; non-array outputs (e.g. sequences) are skipped.

backend
    Measures backend test cases selected by a regular expression.

    Usage::

        python -m onnx_light backend --regex "^test_cc_abs" --mode test
        python -m onnx_light backend --regex "_benchmark$" --mode benchmark --json

kernel
    Lists registered native kernels, shows their tunable parameters, or
    calibrates and persists one selected kernel with ``--tune``.

    Usage::

        onnx-light kernel --list
        onnx-light kernel --kernel Gemm --kernel Softmax [--json]

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from typing import TYPE_CHECKING, Any, Callable, TypedDict, cast

if TYPE_CHECKING:
    from .onnx_proto._helper import TypeProto as _TypeProto

# Action string used in RuntimeEvent records for node-dispatch events.
_EVENT_ACTION_RUN_NODE = "run_node"

# Maximum byte size of an external-data tensor that is loaded inline for shape
# inference.  Tensors at or above this threshold remain as external-data
# references and are not read from disk.  Tensors below the threshold are
# loaded because shape inference may need their values (e.g. the ``shape``
# input of a Reshape node).
_FILLSHAPE_TINY_TENSOR_THRESHOLD = 128


class _KernelTunable(TypedDict):
    library: str
    kernel: str
    implementation: str
    element_type: int
    device: int
    device_name: str
    tuning_abi: int
    calibratable: bool
    defaults: dict[str, bool | int | float | str]
    active_values: dict[str, bool | int | float | str]
    parameter_names: list[str]


class _KernelReportItem(TypedDict):
    identifier: str
    library: str
    device: int
    device_name: str
    dtype: str
    implementation: str
    tunables: list[_KernelTunable]


def _resolve_integer_tuning_parameter(
    command: str, specification: str, tunable: _KernelTunable
) -> tuple[str, list[int]]:
    """Resolves an explicit integer tuning comparison specification."""
    parameter_name, separator, value_text = specification.partition("=")
    value_tokens = value_text.split(",") if separator else []
    prefix = f"onnx-light {command}:"
    if not parameter_name or len(value_tokens) < 2 or any(not token for token in value_tokens):
        raise SystemExit(f"{prefix} --parameter expects NAME=default,VALUE[,VALUE...]")
    if value_tokens[0].lower() != "default":
        raise SystemExit(
            f"{prefix} the first --parameter value must be default to define the baseline"
        )
    if parameter_name not in tunable["parameter_names"]:
        raise SystemExit(
            f"{prefix} unknown tunable parameter {parameter_name!r}; "
            f"expected one of {', '.join(tunable['parameter_names'])}"
        )
    current_value = tunable["active_values"][parameter_name]
    if isinstance(current_value, bool) or not isinstance(current_value, int):
        raise SystemExit(
            f"{prefix} explicit comparisons currently support integer tunable parameters"
        )

    parameter_values: list[int] = []
    for token in value_tokens:
        if token.lower() == "default":
            value = current_value
        else:
            try:
                value = int(token)
            except ValueError:
                raise SystemExit(
                    f"{prefix} invalid value {token!r} for tunable parameter "
                    f"{parameter_name!r}; expected a positive integer"
                ) from None
            if value <= 0:
                raise SystemExit(
                    f"{prefix} invalid value {value} for tunable parameter "
                    f"{parameter_name!r}; expected a positive integer"
                )
        if value not in parameter_values:
            parameter_values.append(value)

    if len(parameter_values) < 2:
        raise SystemExit(
            f"{prefix} tunable parameter {parameter_name!r} needs at least two distinct "
            f"values after resolving default={current_value}; received {', '.join(value_tokens)}"
        )
    if len(parameter_values) > 64:
        raise SystemExit(f"{prefix} at most 64 comparison values are supported")
    return parameter_name, parameter_values


def _parse_kernel_element_type(value: str) -> int:
    """Parses an ONNX element-type integer or enum name."""
    try:
        return int(value)
    except ValueError:
        from .onnx import TensorProto

        name = value.upper()
        if not hasattr(TensorProto, name):
            raise argparse.ArgumentTypeError(
                f"unknown ONNX element type {value!r}; use an integer or name such as FLOAT"
            ) from None
        return int(getattr(TensorProto, name))


def _parse_positive_int(value: str) -> int:
    """Parses a strictly positive integer."""
    if value.isdigit() and int(value) > 0:
        return int(value)
    raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")


def _parse_non_negative_int(value: str) -> int:
    """Parses a non-negative integer."""
    if value.isdigit():
        return int(value)
    raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")


def _parse_nonnegative_int(value: str) -> int:
    """Parses a non-negative integer."""
    if value.isdigit():
        return int(value)
    raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")


def _parse_positive_float(value: str) -> float:
    """Parses a strictly positive floating-point value."""
    import math

    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def _parse_kernel_device(value: str) -> int:
    """Parses a CPU, GPU index, or numeric device identifier."""
    normalized = value.upper()
    if normalized == "CPU":
        return -1
    if normalized == "UNDEFINED":
        return -2
    if normalized.startswith("GPU") and normalized[3:].isdigit():
        index = int(normalized[3:])
        if index <= 8191:
            return index
    if re.fullmatch(r"[+-]?\d+", value):
        device = int(value)
        if -2 <= device <= 8191:
            return device
    raise argparse.ArgumentTypeError(
        f"unknown device {value!r}; use CPU, Undefined, GPU0..GPU8191, or -2..8191"
    )


def _kernel_device_name(device: int) -> str:
    """Returns the canonical name of a parsed kernel device."""
    if device == -2:
        return "Undefined"
    if device == -1:
        return "CPU"
    return f"GPU{device}"


def _optional_kernel_device_name(device: int | None) -> str:
    """Returns the selected device name or ``all``."""
    return "all" if device is None else _kernel_device_name(device)


def _registered_kernel_device(identifier: str) -> int:
    """Returns the device encoded in a native kernel identifier."""
    parts = identifier.split(":")
    return -1 if len(parts) == 2 else int(parts[-1])


def _cmd_kernel(args: argparse.Namespace) -> None:
    """Lists, inspects, or tunes registered kernels."""
    from . import kernel_tuning
    from .onnx import TensorProto
    from .onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

    if args.list and (args.tune or args.parameter):
        raise SystemExit("onnx-light kernel: --tune and --parameter require --kernel, not --list")
    if args.parameter and not args.tune:
        raise SystemExit("onnx-light kernel: --parameter requires --tune")

    identifiers = [
        identifier
        for identifier in runtime.registered_kernels()
        if args.device is None or _registered_kernel_device(identifier) == args.device
    ]
    if args.list:
        if args.json:
            print(json.dumps({"kernels": identifiers}, indent=2))
        else:
            print("\n".join(identifiers))
        return

    selected: set[str] = set()
    unknown = []
    for selector in args.kernel:
        matches = [
            identifier
            for identifier in identifiers
            if identifier == selector
            or (":" not in selector and identifier.split(":", maxsplit=2)[1] == selector)
        ]
        if matches:
            selected.update(matches)
        else:
            unknown.append(selector)
    if unknown:
        raise SystemExit(
            f"onnx-light kernel: unknown kernel(s) for device "
            f"{_optional_kernel_device_name(args.device)}: {', '.join(unknown)}"
        )

    selection = {
        "library": args.library or "all",
        "device": _optional_kernel_device_name(args.device),
        "dtype": "all" if args.dtype is None else TensorProto.DataType(args.dtype).name,
        "implementation": args.impl or "all",
    }

    def build_report() -> list[_KernelReportItem]:
        tuning_report = kernel_tuning.kernel_tuning_parameters(
            library=args.library,
            device=args.device,
            element_type=args.dtype,
            implementation=args.impl,
            path=args.cache,
        )
        tuning_by_kernel: dict[tuple[str, int], list[_KernelTunable]] = {}
        for item in cast(list[_KernelTunable], tuning_report["kernels"]):
            tuning_by_kernel.setdefault((item["kernel"], item["device"]), []).append(item)

        result: list[_KernelReportItem] = []
        for identifier in sorted(selected):
            op_type = identifier.split(":", maxsplit=2)[1]
            device = _registered_kernel_device(identifier)
            tunables = sorted(
                tuning_by_kernel.get((op_type, device), []),
                key=lambda item: (item["library"], item["implementation"], item["element_type"]),
            )
            result.append(
                {
                    "identifier": identifier,
                    "library": args.library or "all",
                    "device": device,
                    "device_name": _kernel_device_name(device),
                    "dtype": (
                        "all" if args.dtype is None else TensorProto.DataType(args.dtype).name
                    ),
                    "implementation": args.impl or "all",
                    "tunables": tunables,
                }
            )
        return result

    def print_report(items: list[_KernelReportItem]) -> None:
        for kernel in items:
            print(
                f"{kernel['identifier']} library={kernel['library']} "
                f"device={kernel['device_name']} dtype={kernel['dtype']} "
                f"impl={kernel['implementation']}"
            )
            if not kernel["tunables"]:
                print("  no tunable parameters")
                continue
            for tunable in kernel["tunables"]:
                element_type = TensorProto.DataType(tunable["element_type"]).name
                print(
                    f"  {element_type} library={tunable['library']} "
                    f"device={tunable['device_name']} "
                    f"implementation={tunable['implementation']} abi={tunable['tuning_abi']}"
                )
                for name in tunable["parameter_names"]:
                    print(
                        f"    {name}: default={tunable['defaults'][name]} "
                        f"active={tunable['active_values'][name]}"
                    )

    report = build_report()
    if args.tune:
        import sys

        tuning_options = {
            "maximum_duration_ms": args.maximum_duration_ms,
            "maximum_memory_mb": args.maximum_memory_mb,
            "parameter": args.parameter,
        }
        if len(report) != 1:
            raise SystemExit(
                f"onnx-light kernel: --tune requires exactly one selected kernel; "
                f"got {len(report)}"
            )
        tunables = report[0]["tunables"]
        if not tunables:
            raise SystemExit("onnx-light kernel: selected kernel has no tunable parameters")
        unsupported = [item for item in tunables if not item["calibratable"]]
        if unsupported:
            raise SystemExit(
                "onnx-light kernel: selected kernel has parameters without a calibration callback"
            )
        parameter_name = None
        parameter_values: list[int] = []
        if args.parameter:
            if len(tunables) != 1:
                raise SystemExit(
                    "onnx-light kernel: --parameter requires exactly one tuning schema; "
                    "select --dtype and --impl"
                )
            tunable = tunables[0]
            parameter_name, parameter_values = _resolve_integer_tuning_parameter(
                "kernel", args.parameter, tunable
            )

        def progress(message: str) -> None:
            if args.verbose:
                print(f"[kernel tune] {message}", file=sys.stderr, flush=True)

        duration = args.maximum_duration_ms or "callback-default"
        memory = args.maximum_memory_mb or "callback-default"
        budget_summary = f"maximum_duration_ms={duration} maximum_memory_mb={memory}"
        progress(f"budgets: {budget_summary}")
        progress("captured parameters before tuning")
        calibrations = []
        for index, tunable in enumerate(tunables, 1):
            dtype = TensorProto.DataType(tunable["element_type"]).name
            progress(
                f"calibrating {index}/{len(tunables)}: "
                f"{tunable['library']}/{tunable['kernel']}/{tunable['implementation']} "
                f"dtype={dtype} device={tunable['device_name']}"
            )
            calibrations.append(
                kernel_tuning.calibrate_kernel_tuning(
                    tunable["kernel"],
                    element_types=[tunable["element_type"]],
                    library=tunable["library"],
                    implementation=tunable["implementation"],
                    maximum_duration_ms=args.maximum_duration_ms,
                    maximum_memory_bytes=args.maximum_memory_mb << 20,
                    save=True,
                    path=args.cache,
                    device=tunable["device"],
                    parameter_name=parameter_name,
                    parameter_values=parameter_values,
                )
            )
        after = build_report()
        progress("captured parameters after tuning")
        tune_report = {
            "selection": selection,
            "tuning_options": tuning_options,
            "before": report,
            "calibrations": calibrations,
            "after": after,
        }
        if args.json:
            print(json.dumps(tune_report, indent=2, sort_keys=True))
        else:
            print(f"tuning budgets: {budget_summary}")
            print("before:")
            print_report(report)
            print("after:")
            print_report(after)
            for calibration in calibrations:
                for comparison in calibration["comparisons"]:
                    print(
                        f"side by side: {comparison['parameter_name']} "
                        f"baseline={comparison['baseline_value']} "
                        f"selected={comparison['selected_value']}"
                    )
                    for value in comparison["values"]:
                        marker = (
                            " baseline" if value["value"] == comparison["baseline_value"] else ""
                        )
                        selected_marker = (
                            " selected" if value["value"] == comparison["selected_value"] else ""
                        )
                        print(
                            f"  value={value['value']} "
                            f"time_ms={value['duration_seconds'] * 1000:.3f} "
                            f"speedup={value['speedup']:.3f}x{marker}{selected_marker}"
                        )
            cache_updates = [
                calibration["cache_update"]
                for calibration in calibrations
                if calibration["cache_update"] is not None
            ]
            for cache_update in cache_updates:
                print(
                    f"machine tuning parameters stored in: {cache_update['path']} "
                    f"(status={cache_update['status']})"
                )
        return

    if args.json:
        print(json.dumps({"selection": selection, "kernels": report}, indent=2, sort_keys=True))
    else:
        print_report(report)


def _measure_backend_test_cases_with_timeout(
    cases: list[Any],
    *,
    mode: str,
    include_big: bool,
    generate_benchmark_expected_outputs: bool,
    repeat: int,
    warmup: int,
    max_repeat_time: float,
    timeout_seconds: float,
    tuning: dict[str, Any] | None = None,
    capture_models: bool = False,
) -> list[dict[str, Any]]:
    """Measures backend cases in a worker that is replaced after a timeout."""
    import multiprocessing

    from .onnx_py._onnxpybackend import backend_test  # type: ignore
    from ._backend_cli import (
        backend_test_worker_ready,
        initialize_backend_test_worker,
        measure_backend_test_case_by_name,
    )

    context = multiprocessing.get_context("spawn")

    def start_worker() -> Any:
        pool = context.Pool(
            processes=1, initializer=initialize_backend_test_worker, initargs=(tuning,)
        )
        try:
            pool.apply_async(backend_test_worker_ready).get(timeout=30)
        except multiprocessing.TimeoutError:
            pool.terminate()
            pool.join()
            raise RuntimeError("backend worker did not initialize within 30 seconds") from None
        return pool

    reports = []
    worker = None
    try:
        for case in cases:
            if worker is None:
                worker = start_worker()
            result = worker.apply_async(
                measure_backend_test_case_by_name,
                (
                    case.name,
                    mode,
                    include_big,
                    generate_benchmark_expected_outputs,
                    repeat,
                    warmup,
                    max_repeat_time,
                    capture_models,
                ),
            )
            try:
                reports.append(result.get(timeout=timeout_seconds))
            except multiprocessing.TimeoutError:
                worker.terminate()
                worker.join()
                worker = None
                reports.append(
                    {
                        "name": case.name,
                        "kind": backend_test.test_case_kind_name(case.kind),
                        "tag": backend_test.test_case_tag_name(case.tag),
                        "status": "timeout",
                        "timed_out": True,
                        "error": f"exceeded {timeout_seconds:g} seconds",
                        "data_sets": None,
                        "input_shapes": None,
                        "model_bytes": None,
                        "materialization_seconds": None,
                        "setup_seconds": None,
                        "warmup_seconds": None,
                        "iteration_seconds": [],
                        "run_seconds": None,
                        "mean_seconds": None,
                        "min_seconds": None,
                        "max_seconds": None,
                    }
                )
    finally:
        if worker is not None:
            worker.close()
            worker.join()
    return reports


def _run_backend_test_timing(
    *,
    name_regex: str = "",
    mode: str = "test",
    include_big: bool = False,
    generate_benchmark_expected_outputs: bool = False,
    repeat: int = 10 * (os.cpu_count() or 1),
    warmup: int = 2 * (os.cpu_count() or 1),
    max_repeat_time: float = 1.0,
    timeout_seconds: float = 2.0,
    tuning_comparison: dict[str, Any] | None = None,
    save_models: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Measures backend test cases selected by name and generation mode."""
    import math
    import time
    from pathlib import Path

    if mode not in {"test", "benchmark"}:
        raise ValueError(f"Unknown backend test mode {mode!r}; expected 'test' or 'benchmark'.")
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}.")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")
    if not math.isfinite(max_repeat_time) or max_repeat_time <= 0:
        raise ValueError(f"max_repeat_time must be positive and finite, got {max_repeat_time}.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive and finite, got {timeout_seconds}.")

    output_directory = None
    if save_models is not None:
        output_directory = Path(save_models)
        if output_directory.exists() and not output_directory.is_dir():
            raise SystemExit(
                f"onnx-light backend: --save-models expects a directory, got {save_models!r}"
            )
        output_directory.mkdir(parents=True, exist_ok=True)

    from onnx_light.onnx import backend

    test_mode = backend.TestMode.TEST if mode == "test" else backend.TestMode.BENCHMARK
    total_start = time.perf_counter()
    collection_start = time.perf_counter()
    cases = backend.collect_test_cases_by_name(
        name_regex,
        include_big=include_big,
        mode=test_mode,
        generate_benchmark_expected_outputs=generate_benchmark_expected_outputs,
    )
    collection_seconds = time.perf_counter() - collection_start
    if tuning_comparison is not None and not cases:
        raise SystemExit(
            f"onnx-light backend: regular expression {name_regex!r} selected no backend cases"
        )

    comparison = None
    if tuning_comparison is None:
        case_reports = _measure_backend_test_cases_with_timeout(
            cases,
            mode=mode,
            include_big=include_big,
            generate_benchmark_expected_outputs=generate_benchmark_expected_outputs,
            repeat=repeat,
            warmup=warmup,
            max_repeat_time=max_repeat_time,
            timeout_seconds=timeout_seconds,
            capture_models=save_models is not None,
        )
        for case in case_reports:
            case.update(
                {
                    "parameter_name": None,
                    "parameter_value": None,
                    "baseline_value": None,
                    "speedup": None,
                }
            )
    else:
        import tempfile

        from . import kernel_tuning

        tunable = cast(_KernelTunable, tuning_comparison["tunable"])
        criterion = cast(str, tuning_comparison.get("criterion", "sum"))
        if "parameter_sets" in tuning_comparison:
            parameter_sets = cast(list[dict[str, int]], tuning_comparison["parameter_sets"])
        else:
            parameter_name = cast(str, tuning_comparison["parameter_name"])
            parameter_sets = [
                {parameter_name: value}
                for value in cast(list[int], tuning_comparison["parameter_values"])
            ]
        parameter_names = list(parameter_sets[0])
        reports_by_value = []
        with tempfile.TemporaryDirectory(prefix="onnx-light-backend-tuning-") as temporary:
            for set_index, parameter_set in enumerate(parameter_sets):
                values = dict(tunable["active_values"])
                values.update(parameter_set)
                cache_path = str(Path(temporary) / f"values-{set_index}.cache")
                try:
                    kernel_tuning.set_kernel_tuning_parameters(
                        tunable["kernel"],
                        tunable["element_type"],
                        values,
                        library=tunable["library"],
                        implementation=tunable["implementation"],
                        tuning_abi=tunable["tuning_abi"],
                        path=cache_path,
                        load=False,
                    )
                except ValueError as exc:
                    raise SystemExit(
                        f"onnx-light backend: invalid tuning parameter set {parameter_set}: {exc}"
                    ) from exc
                reports_by_value.append(
                    _measure_backend_test_cases_with_timeout(
                        cases,
                        mode=mode,
                        include_big=include_big,
                        generate_benchmark_expected_outputs=generate_benchmark_expected_outputs,
                        repeat=repeat,
                        warmup=warmup,
                        max_repeat_time=max_repeat_time,
                        timeout_seconds=timeout_seconds,
                        capture_models=save_models is not None,
                        tuning={
                            "kernel": tunable["kernel"],
                            "element_type": tunable["element_type"],
                            "library": tunable["library"],
                            "implementation": tunable["implementation"],
                            "tuning_abi": tunable["tuning_abi"],
                            "values": values,
                            "path": cache_path,
                        },
                    )
                )
                if progress is not None:
                    progress(set_index + 1, len(parameter_sets))

        baseline_parameters = parameter_sets[0]
        baseline_cases = {case["name"]: case for case in reports_by_value[0]}
        case_reports = []
        comparison_values = []
        latency_report = kernel_tuning.analyze_kernel_tuning_latencies(
            [[case["run_seconds"] for case in reports] for reports in reports_by_value], criterion
        )
        selected_index = latency_report["selected_index"]
        baseline_metrics = latency_report["values"][0]
        for set_index, (parameter_set, value_reports) in enumerate(
            zip(parameter_sets, reports_by_value)
        ):
            run_seconds = sum(
                case["run_seconds"] for case in value_reports if case["run_seconds"] is not None
            )
            timed_out = sum(case["timed_out"] for case in value_reports)
            metrics = latency_report["values"][set_index]
            comparison_values.append(
                {
                    "parameters": parameter_set,
                    "value": (
                        next(iter(parameter_set.values())) if len(parameter_set) == 1 else None
                    ),
                    "run_seconds": run_seconds,
                    "speedup": (
                        None
                        if metrics is None or baseline_metrics is None
                        else baseline_metrics["sum"] / metrics["sum"]
                    ),
                    "timed_out": timed_out,
                    "selected": selected_index is not None and set_index == selected_index,
                    **(
                        metrics
                        if metrics is not None
                        else {
                            "average": None,
                            "sum": None,
                            "median": None,
                            "average_speedup": None,
                            "median_speedup": None,
                            "max_speedup": None,
                            "max_latency": None,
                        }
                    ),
                }
            )
            for case in value_reports:
                baseline_case = baseline_cases[case["name"]]
                case_speedup = (
                    baseline_case["run_seconds"] / case["run_seconds"]
                    if baseline_case["run_seconds"] is not None
                    and case["run_seconds"] is not None
                    and case["run_seconds"] > 0
                    else None
                )
                case.update(
                    {
                        "parameter_name": ",".join(parameter_names),
                        "parameter_value": ",".join(
                            str(parameter_set[name]) for name in parameter_names
                        ),
                        "baseline_value": ",".join(
                            str(baseline_parameters[name]) for name in parameter_names
                        ),
                        "speedup": case_speedup,
                    }
                )
                case_reports.append(case)
        comparison = {
            "kernel": tunable["kernel"],
            "element_type": tunable["element_type"],
            "implementation": tunable["implementation"],
            "criterion": criterion,
            "parameter_names": parameter_names,
            "parameter_name": ",".join(parameter_names),
            "baseline_parameters": baseline_parameters,
            "baseline_value": ",".join(
                str(baseline_parameters[name]) for name in parameter_names
            ),
            "selected_parameters": (
                None if selected_index is None else parameter_sets[selected_index]
            ),
            "values": comparison_values,
        }

    saved_model_paths = {}
    for case in case_reports:
        model_bytes = case.pop("model_bytes", None)
        if model_bytes is None:
            continue
        if output_directory is None:
            raise RuntimeError("model_bytes captured without an output directory")
        if Path(case["name"]).name != case["name"]:
            raise SystemExit(
                f"onnx-light backend: test name {case['name']!r} is not a safe model filename"
            )
        model_path = output_directory / f"{case['name']}.onnx"
        model_path.write_bytes(model_bytes)
        saved_model_paths[case["name"]] = str(model_path)
    for case in case_reports:
        case["model_path"] = saved_model_paths.get(case["name"])

    return {
        "name_regex": name_regex,
        "mode": mode,
        "include_big": include_big,
        "generate_benchmark_expected_outputs": generate_benchmark_expected_outputs,
        "repeat": repeat,
        "warmup": warmup,
        "max_repeat_time": max_repeat_time,
        "timeout_seconds": timeout_seconds,
        "selected": len(cases),
        "timed_out": sum(case["timed_out"] for case in case_reports),
        "tuning_comparison": comparison,
        "save_models": save_models,
        "collection_seconds": collection_seconds,
        "cases": case_reports,
        "total_seconds": time.perf_counter() - total_start,
    }


def _cmd_backend_test(args: argparse.Namespace) -> None:
    """Measures selected backend test cases."""

    tuning_comparison = None
    if args.parameter:
        if not args.regex:
            raise SystemExit("onnx-light backend: --parameter requires --regex")
        if not args.kernel or args.dtype is None or not args.impl:
            raise SystemExit(
                "onnx-light backend: --parameter requires --kernel, --dtype, and --impl "
                "to select exactly one tuning schema"
            )
        if args.criterion is None:
            raise SystemExit("onnx-light backend: --parameter requires --criterion")
        from . import kernel_tuning
        from itertools import product

        tuning_report = kernel_tuning.kernel_tuning_parameters(
            kernel=args.kernel,
            library=args.library,
            element_type=args.dtype,
            implementation=args.impl,
        )
        tunables = cast(list[_KernelTunable], tuning_report["kernels"])
        if len(tunables) != 1:
            raise SystemExit(
                f"onnx-light backend: tuning selectors matched {len(tunables)} schemas; "
                "expected exactly one"
            )
        parameter_options = [
            _resolve_integer_tuning_parameter("backend", specification, tunables[0])
            for specification in args.parameter
        ]
        parameter_names = [name for name, _ in parameter_options]
        if len(set(parameter_names)) != len(parameter_names):
            raise SystemExit("onnx-light backend: every --parameter name must be unique")
        parameter_set_count = 1
        for _, values in parameter_options:
            parameter_set_count *= len(values)
            if parameter_set_count > 256:
                raise SystemExit("onnx-light backend: at most 256 parameter sets are supported")
        parameter_sets = [
            dict(zip(parameter_names, values))
            for values in product(*(values for _, values in parameter_options))
        ]
        tuning_comparison = {
            "tunable": tunables[0],
            "parameter_sets": parameter_sets,
            "criterion": args.criterion,
        }
    elif args.criterion is not None:
        raise SystemExit("onnx-light backend: --criterion requires --parameter")

    def progress(completed: int, total: int) -> None:
        width = 20
        filled = width * completed // total
        ending = "\n" if completed == total else "\r"
        print(
            f"[backend tune] [{'#' * filled}{'-' * (width - filled)}] {completed}/{total}",
            file=sys.stderr,
            end=ending,
            flush=True,
        )

    report = _run_backend_test_timing(
        name_regex=args.regex,
        mode=args.mode,
        include_big=args.include_big,
        generate_benchmark_expected_outputs=args.generate_benchmark_expected_outputs,
        repeat=args.repeat,
        warmup=args.warmup,
        max_repeat_time=args.max_repeat_time,
        timeout_seconds=args.timeout,
        tuning_comparison=tuning_comparison,
        save_models=args.save_models,
        progress=progress if tuning_comparison is not None else None,
    )
    if args.output:
        _write_backend_test_output(report, args.output)
        print(f"backend: wrote {args.output}")
        return
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(
        f"mode={report['mode']} selected={report['selected']} "
        f"repeat={report['repeat']} warmup={report['warmup']} "
        f"max_repeat_time={report['max_repeat_time']:g}s "
        f"timeout={report['timeout_seconds']:g}s timed_out={report['timed_out']} "
        f"collection_ms={report['collection_seconds'] * 1000:.3f}"
    )
    if report["tuning_comparison"] is not None:
        comparison = report["tuning_comparison"]
        print(
            f"side by side: {comparison['kernel']} criterion={comparison['criterion']} "
            f"selected={comparison['selected_parameters']}"
        )
        for value in comparison["values"]:
            metric_text = " ".join(
                f"{name}={value[name]:.6g}" if value[name] is not None else f"{name}=unavailable"
                for name in (
                    "average",
                    "sum",
                    "median",
                    "average_speedup",
                    "median_speedup",
                    "max_speedup",
                    "max_latency",
                )
            )
            selected = " selected" if value["selected"] else ""
            print(
                f"  parameters={value['parameters']} {metric_text} "
                f"timed_out={value['timed_out']}{selected}"
            )
    for case in report["cases"]:
        if case["timed_out"]:
            parameter_text = (
                f" {case['parameter_name']}={case['parameter_value']}"
                if case["parameter_name"] is not None
                else ""
            )
            print(f"{case['name']} status=timeout{parameter_text} error={case['error']}")
            continue
        comparison_text = (
            f" {case['parameter_name']}={case['parameter_value']} "
            f"speedup={case['speedup']:.3f}x"
            if case["parameter_name"] is not None and case["speedup"] is not None
            else ""
        )
        print(
            f"{case['name']} status=completed{comparison_text} datasets={case['data_sets']} "
            f"input_shapes={json.dumps(case['input_shapes'], separators=(',', ':'))} "
            f"model={case['model_path'] or '-'} "
            f"materialization_ms={case['materialization_seconds'] * 1000:.3f} "
            f"setup_ms={case['setup_seconds'] * 1000:.3f} "
            f"run_ms={case['run_seconds'] * 1000:.3f} "
            f"mean_ms={case['mean_seconds'] * 1000:.3f}"
        )
    print(f"total_ms={report['total_seconds'] * 1000:.3f}")


def _backend_test_table(report: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Returns the columns and rows exported by the backend command."""
    columns = [
        "name",
        "kind",
        "tag",
        "status",
        "timed_out",
        "error",
        "parameter_name",
        "parameter_value",
        "baseline_value",
        "speedup",
        "data_sets",
        "input_shapes",
        "model_path",
        "mode",
        "repeat",
        "warmup",
        "max_repeat_time",
        "timeout_seconds",
        "collection_seconds",
        "materialization_seconds",
        "setup_seconds",
        "warmup_seconds",
        "run_seconds",
        "mean_seconds",
        "min_seconds",
        "max_seconds",
        "total_seconds",
        "iteration_seconds",
    ]
    rows = []
    for case in report["cases"]:
        values = {
            "parameter_name": None,
            "parameter_value": None,
            "baseline_value": None,
            "speedup": None,
            "model_path": None,
            **case,
            "mode": report["mode"],
            "repeat": report["repeat"],
            "warmup": report["warmup"],
            "max_repeat_time": report["max_repeat_time"],
            "timeout_seconds": report["timeout_seconds"],
            "collection_seconds": report["collection_seconds"],
            "total_seconds": report["total_seconds"],
            "input_shapes": json.dumps(case["input_shapes"]),
            "iteration_seconds": json.dumps(case["iteration_seconds"]),
        }
        rows.append([values[column] for column in columns])
    return columns, rows


def _write_backend_test_output(report: dict[str, Any], output: str) -> None:
    """Writes a backend timing report selected by the output extension."""
    from pathlib import Path

    output_path = Path(output)
    columns, rows = _backend_test_table(report)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        import csv

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(columns)
            writer.writerows(rows)
        return
    if suffix == ".xlsx":
        from openpyxl import Workbook

        from .tools.kernel_baseline import get_cpu_descriptor

        workbook = Workbook()
        summary = workbook.active
        assert summary is not None
        summary.title = "summary"
        summary.append(["property", "value"])
        for name in (
            "name_regex",
            "mode",
            "include_big",
            "repeat",
            "warmup",
            "max_repeat_time",
            "timeout_seconds",
            "save_models",
            "selected",
            "timed_out",
            "collection_seconds",
            "total_seconds",
        ):
            summary.append([name, report[name]])
        if report.get("tuning_comparison") is not None:
            comparison = report["tuning_comparison"]
            for name in (
                "kernel",
                "element_type",
                "implementation",
                "parameter_name",
                "baseline_value",
            ):
                summary.append([f"tuning.{name}", comparison[name]])
        for name, value in sorted(get_cpu_descriptor().items()):
            summary.append([f"cpu.{name}", value])

        worksheet = workbook.create_sheet("backend")
        worksheet.append(columns)
        for row in rows:
            worksheet.append(row)
        workbook.save(output_path)
        return
    raise SystemExit(
        f"onnx-light backend: unsupported output extension {output_path.suffix!r}; "
        "expected .csv or .xlsx"
    )


def _print_shape_inference_events(events: list) -> None:
    """Prints a compact summary of shape-inference events."""
    print(f"[fillshape] shape inference events: {len(events)}")


def _print_shape_inference_events_detailed(events: list) -> None:
    """Prints detailed shape-inference events."""
    for ev in events:
        d = ev.as_dict()
        op = f"{d['op_domain']}::{d['op_type']}" if d["op_type"] else "-"
        print(
            f"[fillshape] node={d['node_index']:<3d} "
            f"action={d['action']:<12s} op={op:<20s} "
            f"name={d['name'] or '-':<16s} shape={d['shape']}"
        )


def _parse_token_spec(token_str: str) -> tuple[str, int, int]:
    """Parses a ``--token`` specification into ``(name, low, high)``.

    The only accepted format is ``NAME=LOW:HIGH`` — an inclusive range where
    ``LOW <= HIGH``.

    Args:
        token_str: The raw argument string, e.g. ``"seq=1:128"``.

    Returns:
        A ``(name, low, high)`` triple where *low* and *high* are
        non-negative integers.

    Raises:
        ValueError: When the format is invalid or the bounds are
            inconsistent.
    """
    _fmt_error = (
        f"--token value {token_str!r} must be in NAME=LOW:HIGH format "
        "(e.g. --token seq=1:128)."
    )
    if "=" not in token_str:
        raise ValueError(_fmt_error)
    name, _, range_str = token_str.partition("=")
    name = name.strip()
    range_str = range_str.strip()
    if ":" not in range_str:
        raise ValueError(_fmt_error)
    low_str, _, high_str = range_str.partition(":")
    low = int(low_str.strip())
    high = int(high_str.strip())
    if low > high:
        raise ValueError(f"--token {name!r}: lower bound {low} must be <= upper bound {high}.")
    return name, low, high


def _seed_context_with_token_ranges(
    ctx: Any, model: Any, token_ranges: dict[str, tuple[int, int]]
) -> None:
    """Seeds the ShapesContext with concrete dim values for symbolic tokens.

    For each graph input whose shape contains a symbolic dim that matches a
    key in *token_ranges*, that dim is replaced with the lower bound of the
    corresponding range before shape inference runs.  This makes the
    shape-inference engine propagate concrete integer shapes through the
    graph for those dimensions.

    Args:
        ctx: The :class:`ShapesContext` to seed.
        model: The ``ModelProto`` whose graph inputs are inspected.
        token_ranges: Mapping from symbolic dim name to ``(low, high)``
            where *low* is used as the concrete substitute value.
    """
    from .onnx_core.shape_inference import SymTensor

    initializer_names = {init.name for init in model.graph.initializer}
    for vi in model.graph.input:
        vi_name = vi.name
        if vi_name in initializer_names:
            continue
        if not vi.type.has_tensor_type():
            continue
        tensor_type = vi.type.tensor_type
        dtype = int(tensor_type.elem_type)
        if not tensor_type.has_shape():
            continue
        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.has_dim_value() and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            elif dim.has_dim_param() and dim.dim_param in token_ranges:
                # Substitute the symbolic token with its lower bound so that
                # shape inference can propagate a concrete integer dimension.
                dims.append(token_ranges[dim.dim_param][0])
            elif dim.has_dim_param() and dim.dim_param:
                dims.append(str(dim.dim_param))
            else:
                # Fully dynamic (no name, no value): leave as 0.
                dims.append(0)
        ctx.set(vi_name, SymTensor(dtype, dims))


def _remove_node_metadata_key(graph: Any, key: str) -> None:
    """Removes a metadata key from all nodes in a graph and nested subgraphs."""

    for node in graph.node:
        props = node.metadata_props
        kept = [(entry.key, entry.value) for entry in props if entry.key != key]
        if len(kept) != len(props):
            del props[:]
            for k, v in kept:
                entry = props.add()
                entry.key = k
                entry.value = v
        for attr in node.attribute:
            if attr.has_g():
                _remove_node_metadata_key(attr.g, key)
            for subgraph in attr.graphs:
                _remove_node_metadata_key(subgraph, key)


def _cmd_fillshape(args: argparse.Namespace) -> None:
    """Implements the ``fillshape`` subcommand."""
    from .onnx import load, save
    from .onnx_lib.external_data_helper import uses_external_data
    from .onnx_core.shape_inference import (
        ComputeContext,
        Device,
        ShapesContext,
        apply_inferred_shapes_to_model,
        compute_shape_model,
        infer_shapes_model,
        write_peak_memory_to_metadata,
        write_value_and_node_tags_to_metadata,
    )
    from .tools.pretty_print import pretty_onnx

    model_path: str = args.model
    output_path: str | None = args.output
    keep: bool = args.keep
    inplace_info: bool = args.inplace_info
    release_info: bool = args.release_info
    peak_memory: bool = args.peak_memory
    shape_tag: bool = args.shape_tag
    show: bool = args.show
    verbose: int = args.verbose

    # Parse --token specifications into a name → (low, high) mapping.
    token_ranges: dict[str, tuple[int, int]] = {}
    for spec in args.token or []:
        name, low, high = _parse_token_spec(spec)
        token_ranges[name] = (low, high)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")

    if verbose:
        print(f"[fillshape] load {model_path!r}")

    # Load without fetching large external tensor bytes. Shape inference only
    # needs the values of small shape-driving tensors (e.g. Reshape's shape
    # input), so parsing inlines only tiny external tensors and leaves larger
    # weights as external references.
    model = load(
        model_path,
        load_external_data=False,
        tiny_external_data_threshold=_FILLSHAPE_TINY_TENSOR_THRESHOLD,
    )

    # Detect whether the model references weights stored in a separate file
    # (must happen before the tiny-tensor step, which inlines small tensors).
    has_external_data = any(uses_external_data(init) for init in model.graph.initializer)

    if inplace_info or release_info or peak_memory or verbose > 0 or token_ranges:
        # Retains the ShapesContext so in-place reuse analysis, peak memory
        # estimation, verbose event logging, and token-range substitution can
        # reuse the already-inferred shape data.
        ctx = ShapesContext()
        ctx.events_enabled = verbose > 0
        if token_ranges:
            # Pre-seed the context with concrete dim values for the specified
            # symbolic tokens so that shape inference propagates concrete
            # shapes for those dimensions.
            _seed_context_with_token_ranges(ctx, model, token_ranges)
        if verbose:
            print("[fillshape] shape inference")
        compute_shape_model(ctx, model, keep)
        apply_inferred_shapes_to_model(ctx, model)
        if verbose:
            events = ctx.events()
            _print_shape_inference_events(events)
            if verbose >= 2:
                _print_shape_inference_events_detailed(events)
        if inplace_info or release_info:
            what = (
                "inplace/release"
                if inplace_info and release_info
                else ("inplace" if inplace_info else "release")
            )
            if verbose:
                print(f"[fillshape] compute {what} info")
            inplace_context = ComputeContext()
            inplace_context.compute_inplace_reuse_graph(model.graph, ctx)
            if verbose:
                print(f"[fillshape] write {what} info in the model")
            inplace_context.write_to_metadata(model.graph)
            if release_info and not inplace_info:
                _remove_node_metadata_key(model.graph, "onnx_light.inplace_reuse")
        if peak_memory:
            if verbose:
                print("[fillshape] write peak memory info in the model")
            write_peak_memory_to_metadata(ctx, model.graph, Device.kUndefined)
    else:
        if verbose:
            print("[fillshape] shape inference only")
        infer_shapes_model(model, prefill_with_value_info_output=keep)

    if shape_tag:
        tag_context = ComputeContext()
        if verbose:
            print("[fillshape] compute shape tags")
        value_tags, node_tags = tag_context.compute_value_and_node_tags(model.graph)
        if verbose:
            tagged_nodes = sum(1 for tag in node_tags if tag)
            print(
                f"[fillshape] computed {len(value_tags)} value tag(s) "
                f"and {tagged_nodes} node tag(s)"
            )
        if verbose:
            print("[fillshape] write shape tags in the model")
        write_value_and_node_tags_to_metadata(model.graph)

    if show:
        print(
            pretty_onnx(
                model,
                include_inplace=inplace_info,
                include_release=(inplace_info or release_info),
                include_node_tags=shape_tag,
            )
        )
        return

    if output_path is not None and has_external_data:
        # The weight-file paths encoded in the proto are relative to the model
        # directory. Place the output file next to the original model (beside
        # the existing weight files) so those paths remain valid. The weight
        # files are NOT written again.
        model_dir = os.path.dirname(os.path.abspath(model_path))
        dest = os.path.join(model_dir, os.path.basename(output_path))
    elif output_path is not None:
        dest = output_path
    else:
        dest = model_path

    if verbose:
        print(f"[fillshape] save into {dest!r}")
    save(model, dest)


def _cmd_show(args: argparse.Namespace) -> None:
    """Implements the ``show`` subcommand."""
    import sys

    from .onnx import load

    model_path: str = args.model
    fmt: str = args.format
    output_path: str | None = args.output
    run_shape_inference: bool = args.shape_inference
    include_shapes: bool = args.include_shapes
    include_attributes: bool = args.include_attributes
    include_inplace: bool = args.include_inplace
    include_release: bool = args.include_release
    include_node_tags: bool = args.include_node_tags
    include_initializers: bool = args.include_initializers
    direction: str = args.direction
    graphviz_format: str | None = args.graphviz

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")

    model = load(model_path, load_external_data=False)

    if run_shape_inference:
        from .onnx_core.shape_inference import (
            ShapesContext,
            apply_inferred_shapes_to_model,
            compute_shape_model,
        )

        ctx = ShapesContext()
        compute_shape_model(ctx, model)
        apply_inferred_shapes_to_model(ctx, model)

    if fmt == "pretty":
        from .tools.pretty_print import pretty_onnx

        text = pretty_onnx(
            model,
            with_attributes=include_attributes,
            include_node_tags=include_node_tags,
            include_inplace=include_inplace,
            include_release=include_release,
        )
    elif fmt == "mermaid":
        from .tools.mermaid import to_mermaid

        text = to_mermaid(
            model,
            direction=direction,
            include_initializers=include_initializers,
            include_shapes=include_shapes,
            include_attributes=include_attributes,
            include_inplace=include_inplace,
        )
    elif fmt == "svg":
        from .tools.svg import to_svg

        text = to_svg(
            model,
            direction=direction,
            layout=args.layout,
            include_initializers=include_initializers,
            include_shapes=include_shapes,
            include_attributes=include_attributes,
            include_inplace=include_inplace,
        )
    elif fmt == "dot":
        from .tools.dot import to_dot

        dot_text = to_dot(
            model,
            direction=direction,
            include_initializers=include_initializers,
            include_shapes=include_shapes,
            include_attributes=include_attributes,
            include_inplace=include_inplace,
        )

        if graphviz_format is not None:
            import subprocess

            # Validates the format string to prevent command injection.
            # Only allows alphanumeric characters, dots, underscores, and hyphens
            # which covers all legitimate Graphviz output formats (e.g. "png",
            # "svg", "pdf", "xdot1.2").
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", graphviz_format):
                raise ValueError(
                    f"Invalid Graphviz output format {graphviz_format!r}. "
                    "The format must start with a letter or digit and contain only "
                    "alphanumeric characters, dots, underscores, or hyphens "
                    "(e.g. 'png', 'svg', 'pdf', 'xdot1.2')."
                )

            result = subprocess.run(
                ["dot", f"-T{graphviz_format}"],
                input=dot_text.encode(),
                capture_output=True,
                check=True,
            )
            if output_path is not None:
                with open(output_path, "wb") as f:
                    f.write(result.stdout)
            else:
                sys.stdout.buffer.write(result.stdout)
            return

        text = dot_text
    elif fmt in ("onnx-compact", "builder", "cpp"):
        from .tools.translate import translate, translate_header

        text = translate_header(fmt) + translate(model, api=fmt)
    else:
        # argparse enforces choices, so this branch is unreachable in normal
        # usage.  It acts as a defensive guard for programmatic callers that
        # bypass the argument parser.
        raise ValueError(
            f"Unknown format {fmt!r}; expected one of 'pretty', 'mermaid', "
            "'svg', 'dot', 'onnx-compact', 'builder' or 'cpp'."
        )

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


def _resolve_input_shape(
    vi_type: _TypeProto | Any, dim_overrides: dict[str, int], input_name: str
) -> list[int]:
    """Resolves the concrete shape for a tensor graph input.

    Args:
        vi_type: The ``TypeProto`` of the ``ValueInfoProto`` for this input.
        dim_overrides: Mapping from symbolic dim parameter name to concrete
            integer value supplied via ``--dim``.
        input_name: Input name, used only for diagnostic messages.

    Returns:
        List of non-negative integer dimension sizes.

    Raises:
        ValueError: When the type is not a tensor type or a dimension cannot
            be resolved.
    """
    if not vi_type.has_tensor_type():
        raise ValueError(
            f"Input {input_name!r} is not a tensor type; only tensor inputs are "
            "supported by the run subcommand."
        )
    tensor_type = vi_type.tensor_type
    if not tensor_type.has_shape():
        raise ValueError(
            f"Input {input_name!r} has no shape information. "
            "Use --dim to specify the size of each dynamic dimension."
        )
    shape: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.has_dim_value() and dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        elif dim.has_dim_param() and dim.dim_param:
            param = dim.dim_param
            if param in dim_overrides:
                shape.append(dim_overrides[param])
            else:
                # Fall back to 1 for unspecified symbolic dims.
                shape.append(1)
        else:
            # Completely unknown dimension (0 or no annotation).
            shape.append(1)
    return shape


def _dump_tensors_as_model(
    tensors: dict[str, object], dump_path: str, *, ir_version: int = 8
) -> None:
    """Writes *tensors* to *dump_path* as an ONNX model with initializers.

    Creates a ``ModelProto`` whose ``graph.initializer`` list contains one
    ``TensorProto`` per entry in *tensors*.  Silently skips non-``numpy.ndarray``
    values.

    Args:
        tensors: Ordered mapping from tensor name to value.  Skips entries
            that are not ``numpy.ndarray``.
        dump_path: Filesystem path where this function writes the resulting
            ``.onnx`` file.
        ir_version: IR version to set on the model (default: 8).
    """
    import numpy as np

    from .onnx import save
    from .onnx.helper import make_graph, make_model, make_opsetid
    from .onnx.numpy_helper import from_array

    initializers = [
        from_array(value, name=name)
        for name, value in tensors.items()
        if isinstance(value, np.ndarray)
    ]
    graph = make_graph([], "dump", [], [], initializer=initializers)
    model = make_model(graph, opset_imports=[make_opsetid("", 21)])
    model.ir_version = ir_version
    save(model, dump_path)


def _cmd_run(args: argparse.Namespace) -> None:
    """Implements the ``run`` subcommand."""
    import numpy as np

    from .onnx import load
    from .onnx.reference import ReferenceEvaluator
    from .onnx.tools import make_random_input

    model_path: str = args.model
    dim_overrides: dict[str, int] = {}
    for entry in args.dim or []:
        if "=" not in entry:
            raise ValueError(
                f"--dim value {entry!r} must be in NAME=VALUE format (e.g. --dim batch=4)."
            )
        name, _, value_str = entry.partition("=")
        dim_overrides[name.strip()] = int(value_str.strip())
    seed: int = args.seed
    verbose: int = args.verbose
    dump_path: str | None = args.dump

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")

    if verbose >= 1:
        print(f"[run] loading {model_path!r}")
    model = load(model_path)

    initializer_names = {init.name for init in model.graph.initializer}

    # Collect every symbolic dim_param used across all non-initializer inputs.
    used_dim_params: set[str] = set()
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue
        if vi.type.has_tensor_type() and vi.type.tensor_type.has_shape():
            for dim in vi.type.tensor_type.shape.dim:
                if dim.has_dim_param() and dim.dim_param:
                    used_dim_params.add(dim.dim_param)

    # Warn about --dim overrides that do not match any symbolic dim in the inputs.
    for dim_name in dim_overrides:
        if dim_name not in used_dim_params:
            warnings.warn(
                f"--dim {dim_name!r} does not match any symbolic dimension in the model inputs.",
                UserWarning,
                stacklevel=2,
            )

    feed: dict[str, object] = {}
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue
        elem_type = int(vi.type.tensor_type.elem_type)
        shape = _resolve_input_shape(vi.type, dim_overrides, vi.name)
        tensor = make_random_input(elem_type, shape, seed)
        feed[vi.name] = tensor
        if verbose >= 1 and isinstance(tensor, np.ndarray):
            print(f"[run]   input {vi.name!r}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    if verbose >= 1:
        print("[run] running inference")
    sess = ReferenceEvaluator(model, events_enabled=(verbose >= 2))
    outputs = sess.run(None, feed)

    if verbose >= 2:
        # Events include tensor add/replace records as well as node-dispatch
        # ("run_node") records. Only run_node events are shown here because
        # they identify which operator was executed and at which graph position.
        for ev in sess.events():
            d = ev.as_dict()
            if d["action"] == _EVENT_ACTION_RUN_NODE:
                op = f"{d['op_domain']}::{d['op_type']}" if d["op_domain"] else d["op_type"]
                print(f"[run]   node={d['node_index']:<3d} op={op}")

    for name, value in zip(sess.output_names, outputs):
        if isinstance(value, np.ndarray):
            print(f"[run] output {name!r}: shape={list(value.shape)}, dtype={value.dtype}")
            if verbose >= 1:
                print(f"[run]   values: {value}")
        elif isinstance(value, list):
            print(f"[run] output {name!r}: sequence of {len(value)} element(s)")
            if verbose >= 1:
                for i, elem in enumerate(value):
                    print(f"[run]   [{i}]: {elem}")
        else:
            print(f"[run] output {name!r}: {value!r}")

    if dump_path is not None:
        all_tensors: dict[str, object] = dict(feed)
        for name, value in zip(sess.output_names, outputs):
            all_tensors[name] = value
        _dump_tensors_as_model(all_tensors, dump_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-light", description="onnx-light command-line utilities."
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # --- fillshape -----------------------------------------------------------
    fillshape_parser = subparsers.add_parser(
        "fillshape",
        help="Fill a model with optimized shape-inference information.",
        description=(
            "Runs onnx_shapes shape inference on a model and writes the inferred "
            "shapes back into graph.value_info and graph.output."
        ),
    )
    fillshape_parser.add_argument("model", help="Path to the input ONNX model file.")
    fillshape_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="OUTPUT",
        help="Path to write the result to. When omitted the input file is overwritten in place.",
    )
    fillshape_parser.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help=(
            "Seed the inference context from shapes already present in the model. "
            "Existing non-conflicting shapes are kept as anchors."
        ),
    )
    fillshape_parser.add_argument(
        "--inplace-info",
        action="store_true",
        default=False,
        dest="inplace_info",
        help=(
            "Also compute in-place buffer-reuse opportunities and record them "
            "in each node's metadata_props under the key "
            "``onnx_light.inplace_reuse``."
        ),
    )
    fillshape_parser.add_argument(
        "--release-info",
        action="store_true",
        default=False,
        dest="release_info",
        help=(
            "Also compute last-use release hints and record them in each "
            "node's metadata_props under the keys "
            "``onnx_light.release_after`` and ``onnx_light.not_used_after``."
        ),
    )
    fillshape_parser.add_argument(
        "--peak-memory",
        action="store_true",
        default=False,
        dest="peak_memory",
        help=(
            "Also compute the estimated peak scratch memory for each node and "
            "record it in each node's metadata_props under the key "
            "``onnx_light.peak_memory``. Only nodes with a registered "
            "peak-memory function and fully concrete input shapes produce a "
            "non-zero entry; all other nodes are left untouched."
        ),
    )
    fillshape_parser.add_argument(
        "--shape-tag",
        action="store_true",
        default=False,
        dest="shape_tag",
        help=(
            "After shape inference, infer semantic shape/axes/weight/ambiguous tags for "
            "every value and node in the graph and record them in metadata_props "
            "(per-value key ``onnx_light.value_tag`` and per-node key "
            "``onnx_light.node_tag``)."
        ),
    )
    fillshape_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Print inferred shapes to stdout; do not save the model.",
    )
    fillshape_parser.add_argument(
        "--token",
        action="append",
        default=None,
        metavar="NAME=LOW:HIGH",
        help=(
            "Bind a symbolic dimension token to an inclusive integer range "
            "before running shape inference. May be given multiple times "
            "(e.g. --token batch=1:8 --token seq=1:128). "
            "The lower bound is used for shape propagation. "
            "Symbolic dims not covered by --token remain symbolic."
        ),
    )
    fillshape_parser.add_argument(
        "--verbose",
        nargs="?",
        const=1,
        default=0,
        metavar="LEVEL",
        type=int,
        help=(
            "Print shape-inference progress; default level is 1 when the option "
            "is present. Level 2 also prints per-event details."
        ),
    )
    fillshape_parser.set_defaults(func=_cmd_fillshape)

    # --- show ----------------------------------------------------------------
    show_parser = subparsers.add_parser(
        "show",
        help="Print a human-readable, Mermaid, SVG or Python-code rendering of a model.",
        description=(
            "Loads an ONNX model and renders it as plain text (pretty), a Mermaid "
            "flowchart, an SVG image, Graphviz DOT source or code that "
            "rebuilds the model (onnx-compact/builder/cpp). The result is written to "
            "stdout by default."
        ),
    )
    show_parser.add_argument("model", help="Path to the input ONNX model file.")
    show_parser.add_argument(
        "--format",
        "-f",
        default="pretty",
        choices=["pretty", "mermaid", "svg", "dot", "onnx-compact", "builder", "cpp"],
        dest="format",
        help=(
            "Output format: 'pretty' (default) for a compact text listing, "
            "'mermaid' for a Mermaid flowchart, 'svg' for an SVG image, "
            "'dot' for Graphviz DOT source, or 'onnx-compact'/'builder'/'cpp' for "
            "code that rebuilds the model (Python for onnx-compact/builder, C++ for cpp; "
            "see onnx_light.tools.translate)."
        ),
    )
    show_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="OUTPUT",
        help="Write the rendered output to OUTPUT instead of printing to stdout.",
    )
    show_parser.add_argument(
        "--shape-inference",
        action="store_true",
        default=False,
        dest="shape_inference",
        help="Run onnx-light shape inference on the model before rendering.",
    )
    show_parser.add_argument(
        "--no-shapes",
        action="store_false",
        default=True,
        dest="include_shapes",
        help=(
            "Suppress shape annotations in the rendered output "
            "(applies to 'mermaid' and 'svg' formats)."
        ),
    )
    show_parser.add_argument(
        "--include-attributes",
        action="store_true",
        default=False,
        dest="include_attributes",
        help="Include node attributes in the rendered output.",
    )
    show_parser.add_argument(
        "--include-inplace",
        action="store_true",
        default=False,
        dest="include_inplace",
        help="Show in-place buffer-reuse annotations (onnx_light.inplace_reuse metadata).",
    )
    show_parser.add_argument(
        "--include-release",
        action="store_true",
        default=False,
        dest="include_release",
        help=(
            "Show post-execution release hints "
            "(onnx_light.release_after and onnx_light.not_used_after metadata). "
            "Only used by the 'pretty' format."
        ),
    )
    show_parser.add_argument(
        "--include-node-tags",
        action="store_true",
        default=False,
        dest="include_node_tags",
        help=(
            "Show semantic shape/axes/weight/ambiguous node-tag annotations "
            "(onnx_light.node_tag metadata). Only used by the 'pretty' format."
        ),
    )
    show_parser.add_argument(
        "--no-initializers",
        action="store_false",
        default=True,
        dest="include_initializers",
        help=(
            "Exclude initializer nodes from the rendered graph "
            "(applies to 'mermaid', 'svg' and 'dot' formats)."
        ),
    )
    show_parser.add_argument(
        "--direction",
        default="TB",
        metavar="DIRECTION",
        help=(
            "Flowchart direction for 'mermaid', 'svg' and 'dot' formats. "
            "One of TB (default), LR, TD or BT (mermaid only)."
        ),
    )
    show_parser.add_argument(
        "--layout",
        default="layered",
        metavar="LAYOUT",
        choices=("layered", "umap"),
        help=(
            "Box positioning strategy for the 'svg' format. "
            "One of layered (default) or umap (requires the 'umap-learn' package)."
        ),
    )
    show_parser.add_argument(
        "--graphviz",
        default=None,
        metavar="GRAPHVIZ_FORMAT",
        dest="graphviz",
        help=(
            "Invoke the Graphviz 'dot' executable on the generated DOT source "
            "and write the result in GRAPHVIZ_FORMAT (e.g. 'png', 'svg', 'pdf'). "
            "Only used when --format dot is given. "
            "Requires Graphviz to be installed and 'dot' to be on PATH."
        ),
    )
    show_parser.set_defaults(func=_cmd_show)

    # --- run -----------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Generate random inputs and run a model.",
        description=(
            "Generates random inputs for every graph input using the onnx-light "
            "deterministic pseudo-random generators and runs the model through "
            "the onnx-light runtime."
        ),
    )
    run_parser.add_argument("model", help="Path to the input ONNX model file.")
    run_parser.add_argument(
        "--dim",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help=(
            "Override a symbolic (dynamic) dimension by name. "
            "May be given multiple times (e.g. --dim batch=4 --dim seq=16). "
            "Dimensions that have no concrete value and are not covered by --dim "
            "default to 1."
        ),
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="SEED",
        help="Integer seed for the random input generator (default: 0).",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        nargs="?",
        const=1,
        default=0,
        metavar="LEVEL",
        type=int,
        help=(
            "Print run progress; default level is 1 when the option is present. "
            "Level 1 prints loading progress, input shapes and output shapes. "
            "Level 2 also prints per-node execution details."
        ),
    )
    run_parser.add_argument(
        "--dump",
        default=None,
        metavar="PATH",
        help=(
            "Write all inputs and outputs to PATH as an ONNX model whose "
            "graph.initializer contains one TensorProto per tensor. "
            "Non-array outputs (e.g. sequences) are skipped."
        ),
    )
    run_parser.set_defaults(func=_cmd_run)

    # --- backend --------------------------------------------------------------
    backend_test_parser = subparsers.add_parser(
        "backend",
        help="Measure regex-selected backend test cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  Measure matching correctness cases:
    python -m onnx_light backend --regex ".*not.*"

  Export timings and machine information to XLSX:
    python -m onnx_light backend --regex ".*not.*" --output backend-not.xlsx

  Save one self-contained ONNX model per selected test:
    python -m onnx_light backend --regex ".*not.*" --save-models backend-models

  Compare tuning values without changing the machine cache:
    python -m onnx_light backend --regex ".*not.*" \\
      --kernel Not --dtype BOOL --impl portable \\
      --parameter parallel.minimum_elements=default,16384,32768 \\
      --criterion average-speedup
""",
    )
    backend_test_parser.add_argument(
        "--regex",
        default="",
        metavar="PATTERN",
        help="Select test case names with an ECMAScript regular expression (default: all).",
    )
    backend_test_parser.add_argument(
        "--mode",
        choices=("test", "benchmark"),
        default="test",
        help="Generate correctness or benchmark-sized cases (default: test).",
    )
    backend_test_parser.add_argument(
        "--generate-benchmark-expected-outputs",
        action="store_true",
        help=(
            "Generate reference outputs for benchmark cases, enabling correctness "
            "validation at the cost of the output oracle. Benchmark cases are "
            "input-only by default."
        ),
    )
    backend_test_parser.add_argument(
        "--include-big", action="store_true", help="Include cases whose name contains '_big_'."
    )
    backend_test_parser.add_argument(
        "--repeat",
        type=_parse_positive_int,
        default=10 * (os.cpu_count() or 1),
        help="Maximum measured iterations per case (default: 10 per CPU).",
    )
    backend_test_parser.add_argument(
        "--warmup",
        type=_parse_nonnegative_int,
        default=2 * (os.cpu_count() or 1),
        help="Maximum unmeasured warm-up iterations per case (default: 2 per CPU).",
    )
    backend_test_parser.add_argument(
        "--max-repeat-time",
        type=_parse_positive_float,
        default=1.0,
        metavar="SECONDS",
        help="Maximum cumulative time for each warm-up and measured phase (default: 1).",
    )
    backend_test_parser.add_argument(
        "--timeout",
        type=_parse_positive_float,
        default=2.0,
        metavar="SECONDS",
        help="Stop and mark a backend case after this many seconds (default: 2).",
    )
    backend_test_parser.add_argument(
        "--parameter",
        action="append",
        metavar="NAME=default,VALUE,...",
        help=(
            "Compare explicit integer tuning values without changing the machine cache; repeat "
            "the option to evaluate the Cartesian product of multiple parameters; "
            "requires --kernel, --dtype, and --impl."
        ),
    )
    backend_test_parser.add_argument(
        "--criterion",
        choices=(
            "average",
            "sum",
            "median",
            "average-speedup",
            "median-speedup",
            "max-speedup",
            "max-latency",
        ),
        help="Metric used to select the best parameter set; required with --parameter.",
    )
    backend_test_parser.add_argument(
        "--kernel", help="Select the kernel tuning schema used by --parameter."
    )
    backend_test_parser.add_argument(
        "--dtype",
        type=_parse_kernel_element_type,
        metavar="DTYPE",
        help="Select the ONNX element type of the tuning schema used by --parameter.",
    )
    backend_test_parser.add_argument(
        "--impl", metavar="IMPLEMENTATION", help="Select the implementation used by --parameter."
    )
    backend_test_parser.add_argument(
        "--library",
        default="onnx_light",
        help="Select the tuning library used by --parameter (default: onnx_light).",
    )
    backend_test_parser.add_argument(
        "--json", action="store_true", help="Print the complete machine-readable report."
    )
    backend_test_parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Write case timings as CSV or XLSX, selected by the file extension.",
    )
    backend_test_parser.add_argument(
        "--save-models",
        metavar="DIRECTORY",
        help="Save each completed test as one self-contained <test-name>.onnx file.",
    )
    backend_test_parser.set_defaults(func=_cmd_backend_test)

    # --- kernel --------------------------------------------------------------
    kernel_parser = subparsers.add_parser(
        "kernel",
        help="List kernels, inspect tunable parameters, or tune one selected kernel.",
        description=(
            "Lists registered kernels, inspects their tunable parameters, or calibrates "
            "and persists one selected kernel with --tune."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  List every registered kernel:
    python -m onnx_light kernel --list

  Inspect one kernel's tunable parameters:
    python -m onnx_light kernel --kernel Gemm --dtype FLOAT --impl portable

  Tune one kernel automatically:
    python -m onnx_light kernel --kernel Gemm --dtype FLOAT --impl portable --tune

  Compare explicit values and persist the fastest:
    python -m onnx_light kernel --kernel Gemm --dtype FLOAT --impl portable --tune \\
      --parameter parallel.minimum_tasks=default,2,4
""",
    )
    kernel_mode = kernel_parser.add_mutually_exclusive_group(required=True)
    kernel_mode.add_argument(
        "--list", action="store_true", help="List every registered native kernel identifier."
    )
    kernel_mode.add_argument(
        "--kernel",
        action="append",
        metavar="NAME",
        help=(
            "Show tunable parameters for a kernel name or domain-qualified identifier; "
            "may be specified multiple times."
        ),
    )
    kernel_parser.add_argument(
        "--json", action="store_true", help="Print a machine-readable JSON report."
    )
    kernel_parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Calibrate and persist exactly one selected kernel, using the duration "
            "and memory budgets below."
        ),
    )
    kernel_parser.add_argument(
        "--parameter",
        metavar="NAME=default,VALUE,...",
        help=(
            "Benchmark explicit integer values side by side while tuning; default resolves "
            "to the current active value and defines the speedup baseline."
        ),
    )
    kernel_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Report tuning progress to stderr."
    )
    kernel_parser.add_argument(
        "--library", help="Restrict tuning schemas to one library (default: all)."
    )
    kernel_parser.add_argument(
        "--device",
        type=_parse_kernel_device,
        metavar="DEVICE",
        help=(
            "Restrict tuning schemas to CPU, Undefined, GPU<N>, or a numeric device "
            "(default: all)."
        ),
    )
    kernel_parser.add_argument(
        "--dtype",
        type=_parse_kernel_element_type,
        metavar="DTYPE",
        help="Restrict tuning schemas to one ONNX element type name or integer (default: all).",
    )
    kernel_parser.add_argument(
        "--impl",
        metavar="IMPLEMENTATION",
        help="Restrict tuning schemas to one implementation (default: all).",
    )
    kernel_parser.add_argument(
        "--cache", help="Read and persist tuning parameters in this cache file."
    )
    kernel_parser.add_argument(
        "--maximum-duration-ms",
        type=_parse_non_negative_int,
        default=0,
        help="Per-key calibration duration budget; 0 uses the callback default.",
    )
    kernel_parser.add_argument(
        "--maximum-memory-mb",
        type=_parse_non_negative_int,
        default=0,
        help="Calibration memory budget in MiB; 0 uses the callback default.",
    )
    kernel_parser.set_defaults(func=_cmd_kernel)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parses *argv* and dispatches to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
