from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
from typing import Any


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

    unique_roots = list(dict.fromkeys(script_roots))

    base_candidates = [
        root / candidate for root in unique_roots for candidate in relative_candidates
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


# ---------------------------------------------------------------------------
# Operator documentation generation
# ---------------------------------------------------------------------------

_DOMAIN_DISPLAY: dict[str, str] = {
    "": "ai.onnx (default)",
    "ai.onnx.ml": "ai.onnx.ml",
    "ai.onnx.training": "ai.onnx.training",
    "ai.onnx.preview.training": "ai.onnx.preview.training",
}

_ATTR_TYPE_NAMES: dict[int, str] = {
    1: "float",
    2: "int",
    3: "string",
    4: "tensor",
    5: "graph",
    6: "float[]",
    7: "int[]",
    8: "string[]",
    9: "tensor[]",
    10: "graph[]",
    11: "sparse_tensor",
    13: "type_proto",
}


def _option_suffix(option: Any) -> str:
    """Returns a human-readable suffix for a FormalParameter option."""
    name = str(option)
    if "Optional" in name:
        return " (optional)"
    if "Variadic" in name:
        return " (variadic)"
    return ""


def _domain_file_stem(domain: str) -> str:
    """Returns a safe filename stem for *domain*."""
    if domain == "":
        return "ai_onnx"
    return domain.replace(".", "_")


def _domain_title(domain: str) -> str:
    """Returns the display title for *domain*."""
    return _DOMAIN_DISPLAY.get(domain, domain)


def _strip_html(text: str) -> str:
    """Removes simple HTML tags (e.g. ``<br>``) from *text*."""
    return re.sub(r"<[^>]+>", " ", text)


def _format_doc(doc: str, indent: int = 0) -> str:
    """Formats a raw doc-string for inclusion in RST output."""
    if not doc:
        return ""
    # Remove HTML tags that appear in some ONNX doc strings.
    doc = _strip_html(doc)
    lines = doc.strip().splitlines()
    prefix = " " * indent
    return "\n".join(prefix + line for line in lines)


def _schema_to_rst(schema: Any) -> str:
    """Renders a single OpSchema as an RST section."""
    lines: list[str] = []

    # Section header
    title = f"**{schema.name}**-{schema.since_version}"
    lines.append(title)
    lines.append("~" * len(title))
    lines.append("")

    if schema.deprecated:
        lines.append(".. warning::")
        lines.append("   This operator is **deprecated**.")
        lines.append("")

    # Documentation string
    if schema.doc:
        lines.append(_format_doc(schema.doc))
        lines.append("")

    # Inputs
    if schema.inputs:
        lines.append("**Inputs**")
        lines.append("")
        for inp in schema.inputs:
            suffix = _option_suffix(inp.option)
            lines.append(f"- **{inp.name}** (*{inp.type_str}*){suffix}: {inp.description}")
        lines.append("")

    # Outputs
    if schema.outputs:
        lines.append("**Outputs**")
        lines.append("")
        for out in schema.outputs:
            suffix = _option_suffix(out.option)
            lines.append(f"- **{out.name}** (*{out.type_str}*){suffix}: {out.description}")
        lines.append("")

    # Attributes
    if schema.attributes:
        lines.append("**Attributes**")
        lines.append("")
        for attr_name in sorted(schema.attributes):
            attr = schema.attributes[attr_name]
            type_name = _ATTR_TYPE_NAMES.get(int(attr.type), str(attr.type))
            lines.append(f"- **{attr_name}** (*{type_name}*): {attr.description}")
        lines.append("")

    # Type constraints
    if schema.type_constraints:
        lines.append("**Type Constraints**")
        lines.append("")
        for tc in schema.type_constraints:
            allowed = ", ".join(sorted(tc.allowed_type_strs))
            lines.append(f"- **{tc.type_param_str}**: {tc.description}")
            lines.append(f"  Allowed types: {allowed}.")
        lines.append("")

    return "\n".join(lines)


def _domain_page_rst(domain: str, schemas: list[Any], all_schemas_with_history: list[Any]) -> str:
    """Returns full RST content for a single domain page."""
    title = _domain_title(domain)
    lines: list[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    domain_display = domain if domain else "ai.onnx"
    lines.append(f"This page lists all operators in the **{domain_display}** domain.")
    lines.append("")

    # Summary table
    sorted_schemas = sorted(schemas, key=lambda s: s.name)

    lines.append(".. list-table::")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 30 10 10 50")
    lines.append("")
    lines.append("   * - Operator")
    lines.append("     - Since version")
    lines.append("     - Deprecated")
    lines.append("     - Short description")
    for s in sorted_schemas:
        first_line = s.doc.strip().splitlines()[0] if s.doc else ""
        first_line = _strip_html(first_line).strip()
        deprecated = "Yes" if s.deprecated else "No"
        # Truncate long descriptions
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        lines.append(f"   * - :ref:`{s.name} <op_{_domain_file_stem(domain)}_{s.name}>`")
        lines.append(f"     - {s.since_version}")
        lines.append(f"     - {deprecated}")
        lines.append(f"     - {first_line}")
    lines.append("")

    # Detailed sections – one per operator with version history
    history_by_name: dict[str, list[Any]] = {}
    for s in all_schemas_with_history:
        if s.domain == domain:
            history_by_name.setdefault(s.name, []).append(s)

    lines.append("Operator Details")
    lines.append("----------------")
    lines.append("")

    for s in sorted_schemas:
        stem = _domain_file_stem(domain)
        lines.append(f".. _op_{stem}_{s.name}:")
        lines.append("")
        lines.append(_schema_to_rst(s))

        # Older versions (collapsed under a rubric)
        history = sorted(
            history_by_name.get(s.name, []), key=lambda x: x.since_version, reverse=True
        )
        older = [h for h in history if h.since_version != s.since_version]
        if older:
            lines.append(".. rubric:: Version history")
            lines.append("")
            for old in older:
                lines.append(f"  - Version {old.since_version}")
            lines.append("")

    return "\n".join(lines)


def _index_page_rst(domains: list[str]) -> str:
    """Returns RST content for the operators/index.rst page."""
    lines: list[str] = []
    lines.append(".. _l-onnx-operators:")
    lines.append("")
    lines.append("ONNX Operators")
    lines.append("==============")
    lines.append("")
    lines.append(
        "This section lists all ONNX operators grouped by domain.  "
        "Each domain page shows the latest version of every operator "
        "together with its inputs, outputs, attributes and type constraints."
    )
    lines.append("")
    lines.append(".. toctree::")
    lines.append("   :maxdepth: 1")
    lines.append("")
    for domain in sorted(domains):
        stem = _domain_file_stem(domain)
        lines.append(f"   {stem}")
    lines.append("")

    lines.append("Domain overview")
    lines.append("---------------")
    lines.append("")
    for domain in sorted(domains):
        stem = _domain_file_stem(domain)
        domain_title = _domain_title(domain)
        lines.append(f"- :doc:`{domain_title} <{stem}>`")
    lines.append("")

    return "\n".join(lines)


def generate_operators_doc(output_dir: str) -> None:
    """Generates operator RST pages into *output_dir*.

    Reads all ONNX operator schemas from the ``onnx`` package and writes one
    RST file per domain plus a top-level ``index.rst`` toctree.  When the
    ``onnx`` package is not installed a minimal placeholder ``index.rst`` is
    written so Sphinx toctrees do not break.

    Args:
        output_dir: Directory where the generated ``.rst`` files are written.
            It is created if it does not already exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    from onnx_light.onnx import defs as _defs

    schemas = _defs.get_all_schemas()
    schemas_with_history = _defs.get_all_schemas_with_history()
    assert schemas, "No schema detected."

    # Group latest schemas by domain
    by_domain: dict[str, list[Any]] = {}
    for s in schemas:
        by_domain.setdefault(s.domain, []).append(s)

    for domain, domain_schemas in by_domain.items():
        stem = _domain_file_stem(domain)
        path = os.path.join(output_dir, f"{stem}.rst")
        content = _domain_page_rst(domain, domain_schemas, schemas_with_history)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    # Write the top-level index
    index_path = os.path.join(output_dir, "index.rst")
    index_content = _index_page_rst(list(by_domain.keys()))
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(index_content)
