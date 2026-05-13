import os
import pathlib
import re
import shutil
import subprocess


def find_standalone_executable(
    executable_name: str,
    relative_candidates: list[pathlib.Path | str],
    script_file: str | None,
    windows_build_configs: tuple[str, ...] | None = None,
) -> str | None:
    """Locates a standalone executable built from repository examples.

    Args:
        executable_name: Name used for PATH lookup fallback.
        relative_candidates: Candidate executable paths relative to repository root.
        script_file: Path to the calling script file used to locate repository root.
            The repository root is assumed to be three parent directories above
            this path.
        windows_build_configs: Optional Windows build configuration folder names.

    Returns:
        The discovered executable path. Returns ``None`` when the
        ``CI`` environment variable is enabled, or when no candidate file
        exists and PATH lookup does not find the executable.
    """
    ci_env_value = os.environ.get("CI", "").lower()
    if ci_env_value in {"1", "true", "yes"}:
        return None
    if not script_file:
        cwd = pathlib.Path.cwd().resolve()
        script_roots = [cwd, *cwd.parents, pathlib.Path(__file__).resolve().parents[1]]
    else:
        script_roots = [pathlib.Path(script_file).resolve().parents[3]]

    deduplicated_roots = list(dict.fromkeys(script_roots))

    base_candidates = [
        root / candidate for root in deduplicated_roots for candidate in relative_candidates
    ]

    candidates = list(base_candidates)
    if os.name == "nt":
        if windows_build_configs is None:
            windows_build_configs = ("Release", "RelWithDebInfo", "Debug", "MinSizeRel")
        windows_candidates = []
        for candidate in base_candidates:
            windows_candidates.extend(
                [candidate.parent / config / candidate.name for config in windows_build_configs]
            )
        candidates.extend(windows_candidates)
        candidates.extend([path.with_suffix(".exe") for path in candidates])

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which(executable_name)


def measure_cpp_with_example(
    executable: str | None,
    args: list[str],
    metric_pattern: re.Pattern[str],
    result_name: str,
    executable_name: str,
) -> dict | None:
    """Runs a standalone C++ benchmark executable and parses its timing output.

    Args:
        executable: Path to the C++ executable, or ``None`` if unavailable.
        args: Arguments passed to the executable (not including the executable itself).
        metric_pattern: Compiled regex pattern to match metric lines in stdout.
            Must capture the metric label in group 1 and the numeric value in group 2.
            The captured label must produce ``"average"``, ``"median"``, ``"min"``, and ``"max"``
            (case-folded) for the four required metrics, and may also produce
            ``"std"`` or ``"standard deviation"``.
        result_name: Benchmark name stored in the returned dictionary's ``name`` key.
        executable_name: Human-readable executable name used in diagnostic messages.

    Returns:
        A benchmark dictionary with keys ``name``, ``median``, ``avg``, ``min``, ``max``,
        and ``std`` if successful, otherwise ``None``.
    """
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, *args], capture_output=True, text=True, check=True, timeout=300
        )
    except subprocess.CalledProcessError as e:
        print(f"{executable_name} execution failed, skipping C++ benchmark: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        print(f"{executable_name} execution timed out, skipping C++ benchmark.")
        return None
    except OSError as e:
        print(f"Could not execute {executable_name}, skipping C++ benchmark: {e}")
        return None

    values = {}
    for line in completed.stdout.splitlines():
        match = metric_pattern.match(line)
        if match is not None:
            # C++ examples report milliseconds; benchmark table uses seconds.
            label = match.group(1).lower()
            if label in {"std", "standard deviation"}:
                label = "std"
            values[label] = float(match.group(2)) / 1e3

    if not {"average", "median", "min", "max"}.issubset(values):
        print(f"Could not parse {executable_name} output, skipping C++ benchmark.")
        return None

    return {
        "name": result_name,
        "median": values["median"],
        "avg": values["average"],
        "min": values["min"],
        "max": values["max"],
        "std": values.get("std", float("nan")),
    }
