import os
import pathlib
import shutil


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
        Returns the discovered executable path. Returns ``None`` when the
        ``CI`` environment variable is enabled, or when no candidate file
        exists and PATH lookup does not find the executable.
    """
    ci_env_value = os.environ.get("CI", "").lower()
    if ci_env_value in {"1", "true", "yes"}:
        return None
    if not script_file:
        return None

    script_root = pathlib.Path(script_file).resolve().parents[3]
    base_candidates = [script_root / candidate for candidate in relative_candidates]

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
