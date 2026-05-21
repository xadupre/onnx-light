from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import textwrap
from typing import Any, Callable


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


_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_RST_ROLE_PREFIX_RE = re.compile(r":[a-zA-Z][a-zA-Z0-9_]*:$")
_RST_CODE_BLOCK_INDENT = " " * 4
_RST_DIRECTIVE_PREFIX = ".. "
_BULLET_MARKERS = ("* ", "- ")
_ELLIPSIS = "..."


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


def _format_markdown_inline(text: str) -> str:
    """Converts inline Markdown constructs into RST equivalents."""
    text = _MARKDOWN_LINK_RE.sub(r"`\1 <\2>`_", text)

    def replace_inline_code(match: re.Match[str]) -> str:
        code_text = match.group(1)
        # Skip already-converted RST links: `label <target>`_.
        if (
            code_text.endswith(">")
            and " <" in code_text
            and text[match.end() : match.end() + 1] == "_"
        ):
            return match.group(0)
        start = match.start()
        prefix = text[:start]
        # Skip explicit RST roles such as :math:`...`.
        role_match = _RST_ROLE_PREFIX_RE.search(prefix)
        if role_match is not None:
            return match.group(0)
        return f"``{code_text}``"

    return _MARKDOWN_INLINE_CODE_RE.sub(replace_inline_code, text)


def _format_doc(doc: str, indent: int = 0) -> str:
    """Formats a raw doc-string for inclusion in RST output."""
    if not doc:
        return ""
    # Remove HTML tags that appear in some ONNX doc strings.
    doc = _strip_html(doc)
    # ONNX op docstrings come from C++ raw string literals and typically have a
    # uniform leading indentation on every content line. Normalizing it avoids
    # spurious literal blocks in the rendered Sphinx output.
    doc = textwrap.dedent(doc)
    raw_lines = doc.strip().splitlines()

    lines: list[str] = []
    in_fenced_code = False
    # Track the indentation of the most recent bullet item to detect when
    # a bullet list ends and a new block starts (RST requires a blank line there).
    last_bullet_indent: int | None = None
    # Track whether we are inside an auto-generated ``.. code-block:: text``
    # directive started because the source contained an indented block.
    in_auto_code_block = False
    auto_code_base_indent = 0

    def end_auto_code_block() -> None:
        nonlocal in_auto_code_block, auto_code_base_indent
        if in_auto_code_block:
            if lines and lines[-1] != "":
                lines.append("")
            in_auto_code_block = False
            auto_code_base_indent = 0

    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            end_auto_code_block()
            if not in_fenced_code:
                # RST expects a blank line before a directive such as ``.. code-block::``.
                if lines and lines[-1] != "":
                    lines.append("")
                language = stripped[3:].strip()
                lines.append(f".. code-block:: {language}".rstrip())
                lines.append("")
                in_fenced_code = True
            else:
                lines.append("")
                in_fenced_code = False
            last_bullet_indent = None
            continue

        if in_fenced_code:
            lines.append(f"{_RST_CODE_BLOCK_INDENT}{line}")
            continue

        if not stripped:
            # A blank line in the source already terminates the bullet list,
            # but is preserved verbatim inside an auto-code-block since RST
            # treats blank lines inside an indented directive content as part
            # of that content.
            last_bullet_indent = None
            lines.append("")
            continue

        cur_indent = len(line) - len(line.lstrip())
        is_bullet = stripped.startswith(_BULLET_MARKERS)

        if is_bullet:
            end_auto_code_block()
            last_bullet_indent = cur_indent
            lines.append(_format_markdown_inline(line))
            continue

        # Continuation of a bullet item: indented more than the bullet marker.
        if last_bullet_indent is not None and cur_indent > last_bullet_indent:
            lines.append(_format_markdown_inline(line))
            continue

        # End any pending bullet list before processing the current line.
        if last_bullet_indent is not None:
            if not lines or lines[-1] != "":
                lines.append("")
            last_bullet_indent = None

        # Indented text outside fenced code, bullets, and bullet continuations
        # is wrapped in a ``.. code-block:: text`` directive so that Sphinx
        # renders it as preformatted output instead of trying to parse it.
        if cur_indent > 0:
            if not in_auto_code_block:
                if lines and lines[-1] != "":
                    lines.append("")
                lines.append(".. code-block:: text")
                lines.append("")
                in_auto_code_block = True
                auto_code_base_indent = cur_indent
            relative_line = (
                line[auto_code_base_indent:]
                if cur_indent >= auto_code_base_indent
                else line.lstrip()
            )
            lines.append(f"{_RST_CODE_BLOCK_INDENT}{relative_line}")
            continue

        # Regular paragraph line: terminate any auto-code-block first.
        end_auto_code_block()
        lines.append(_format_markdown_inline(line))

    end_auto_code_block()

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


def _schema_section_lines(schema: Any) -> list[str]:
    """Returns RST lines for the body of a schema.

    Covers doc string, inputs, outputs, attributes, and type constraints.
    """
    lines: list[str] = []

    if schema.doc:
        lines.append(_format_doc(schema.doc))
        lines.append("")

    if schema.inputs:
        lines.append("**Inputs**")
        lines.append("")
        for inp in schema.inputs:
            suffix = _option_suffix(inp.option)
            _append_operator_field(
                lines, f"**{inp.name}** (*{inp.type_str}*){suffix}", inp.description
            )
        lines.append("")

    if schema.outputs:
        lines.append("**Outputs**")
        lines.append("")
        for out in schema.outputs:
            suffix = _option_suffix(out.option)
            _append_operator_field(
                lines, f"**{out.name}** (*{out.type_str}*){suffix}", out.description
            )
        lines.append("")

    if schema.attributes:
        lines.append("**Attributes**")
        lines.append("")
        for attr_name in sorted(schema.attributes):
            attr = schema.attributes[attr_name]
            type_name = _ATTR_TYPE_NAMES.get(int(attr.type), str(attr.type))
            _append_operator_field(lines, f"**{attr_name}** (*{type_name}*)", attr.description)
        lines.append("")

    if schema.type_constraints:
        lines.append("**Type Constraints**")
        lines.append("")
        for tc in schema.type_constraints:
            allowed = ", ".join(sorted(tc.allowed_type_strs))
            _append_operator_field(lines, f"**{tc.type_param_str}**", tc.description)
            lines.append(f"  Allowed types: {allowed}.")
        lines.append("")

    return lines


def _append_operator_field(lines: list[str], label: str, description: str) -> None:
    """Appends one bullet field and formats multiline descriptions for RST lists."""
    formatted = _format_doc(description, indent=2)
    formatted_lines = formatted.splitlines()
    if not formatted_lines:
        lines.append(f"- {label}")
        return

    first = formatted_lines[0].lstrip()
    if len(formatted_lines) == 1 and not first.startswith(_RST_DIRECTIVE_PREFIX):
        lines.append(f"- {label}: {first}")
        return

    lines.append(f"- {label}:")
    lines.extend(formatted_lines)


def _operator_page_rst(schema: Any, domain: str, all_schemas_with_history: list[Any]) -> str:
    """Returns full RST content for a single operator page (latest version)."""
    stem = _domain_file_stem(domain)
    lines: list[str] = []

    # Anchor label so cross-references from the domain summary table work
    lines.append(f".. _op_{stem}_{schema.name}:")
    lines.append("")

    title = schema.name
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    if schema.deprecated:
        lines.append(".. warning::")
        lines.append("   This operator is **deprecated**.")
        lines.append("")

    domain_display = domain if domain else "ai.onnx"
    lines.append(f"- **Domain**: ``{domain_display}``")
    lines.append(f"- **Since version**: {schema.since_version}")
    lines.append("")

    lines.extend(_schema_section_lines(schema))

    # Version history with links to individual past-version pages
    history = sorted(
        [s for s in all_schemas_with_history if s.domain == domain and s.name == schema.name],
        key=lambda x: x.since_version,
        reverse=True,
    )
    older = [h for h in history if h.since_version != schema.since_version]
    if older:
        lines.append("Version History")
        lines.append("---------------")
        lines.append("")
        for old in older:
            lines.append(
                f"- :doc:`Version {old.since_version} <{schema.name}-{old.since_version}>`"
            )
        lines.append("")

    return "\n".join(lines)


def _operator_version_page_rst(schema: Any, domain: str, latest_schema: Any) -> str:
    """Returns full RST content for a past-version page of an operator.

    Args:
        schema: The historical OpSchema for this specific version.
        domain: The operator domain string.
        latest_schema: The current (latest) OpSchema for the same operator,
            used to generate a back-link to the latest version page.

    Returns:
        RST content as a string.
    """
    stem = _domain_file_stem(domain)
    lines: list[str] = []

    anchor = f"op_{stem}_{schema.name}-{schema.since_version}"
    lines.append(f".. _{anchor}:")
    lines.append("")

    title = f"{schema.name} - version {schema.since_version}"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    lines.append(
        f"This page documents version **{schema.since_version}** of operator "
        f"**{schema.name}**. "
        f"See :doc:`{schema.name}` for the latest version "
        f"(since version {latest_schema.since_version})."
    )
    lines.append("")

    if schema.deprecated:
        lines.append(".. warning::")
        lines.append("   This operator is **deprecated**.")
        lines.append("")

    domain_display = domain if domain else "ai.onnx"
    lines.append(f"- **Domain**: ``{domain_display}``")
    lines.append(f"- **Since version**: {schema.since_version}")
    lines.append("")

    lines.extend(_schema_section_lines(schema))

    return "\n".join(lines)


def _domain_page_rst(domain: str, schemas: list[Any], all_schemas_with_history: list[Any]) -> str:
    """Returns RST content for a single domain overview page.

    Args:
        domain: The operator domain string.
        schemas: Latest-version schemas for this domain.
        all_schemas_with_history: All schemas including historical versions,
            used to add past-version pages to the toctree.

    Returns:
        RST content as a string.
    """
    title = _domain_title(domain)
    lines: list[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    domain_display = domain if domain else "ai.onnx"
    lines.append(f"This page lists all operators in the **{domain_display}** domain.")
    lines.append("")

    sorted_schemas = sorted(schemas, key=lambda s: s.name)
    stem = _domain_file_stem(domain)

    # Build a mapping from operator name -> sorted list of past versions
    past_versions: dict[str, list[Any]] = {}
    for s in sorted_schemas:
        older = sorted(
            [
                h
                for h in all_schemas_with_history
                if h.domain == domain and h.name == s.name and h.since_version != s.since_version
            ],
            key=lambda x: x.since_version,
        )
        if older:
            past_versions[s.name] = older

    # Hidden toctree so Sphinx picks up the individual operator pages
    # (latest versions first, then historical versions sorted by name then version)
    lines.append(".. toctree::")
    lines.append("   :hidden:")
    lines.append("")
    for s in sorted_schemas:
        lines.append(f"   {stem}/{s.name}")
    for s in sorted_schemas:
        for old in past_versions.get(s.name, []):
            lines.append(f"   {stem}/{s.name}-{old.since_version}")
    lines.append("")

    # Summary table with links to individual operator pages
    lines.append(".. list-table::")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 30 10 10 50")
    lines.append("")
    lines.append("   * - Operator")
    lines.append("     - Since version")
    lines.append("     - Deprecated")
    lines.append("     - Short description")
    for s in sorted_schemas:
        first_line = _short_description(s.doc)
        deprecated = "Yes" if s.deprecated else "No"
        lines.append(f"   * - :ref:`{s.name} <op_{stem}_{s.name}>`")
        lines.append(f"     - {s.since_version}")
        lines.append(f"     - {deprecated}")
        lines.append(f"     - {first_line}")
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


def _short_description(doc: str, max_len: int = 80) -> str:
    """Returns a one-line summary with HTML/backticks removed and optional truncation."""
    if not doc:
        return ""
    first_line = doc.strip().splitlines()[0]
    first_line = _strip_html(first_line).strip().replace("`", "")
    if len(first_line) > max_len:
        first_line = first_line[: max_len - len(_ELLIPSIS)] + _ELLIPSIS
    return first_line


def generate_operators_doc(
    output_dir: str, progress_callback: Callable[[str], None] | None = None
) -> None:
    """Generates operator RST pages into *output_dir*.

    Reads all ONNX operator schemas from ``onnx_light.onnx.defs`` and writes
    one RST file per domain plus a top-level ``index.rst`` toctree.

    Args:
        output_dir: Directory where the generated ``.rst`` files are written.
            It is created if it does not already exist.
        progress_callback: Optional callback receiving progress messages while
            pages are generated.
    """

    def _report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    os.makedirs(output_dir, exist_ok=True)

    from onnx_light.onnx import defs as _defs

    schemas = _defs.get_all_schemas()
    schemas_with_history = _defs.get_all_schemas_with_history()
    assert schemas, "No schema detected."

    # Group latest schemas by domain
    by_domain: dict[str, list[Any]] = {}
    for s in schemas:
        by_domain.setdefault(s.domain, []).append(s)

    domains = list(by_domain.items())
    _report(f"Generating operator pages for {len(domains)} domain(s).")

    for domain_index, (domain, domain_schemas) in enumerate(domains, start=1):
        _report(
            f"[{domain_index}/{len(domains)}] Generating domain "
            f"{_domain_title(domain)!r} ({len(domain_schemas)} operators)."
        )
        stem = _domain_file_stem(domain)
        path = os.path.join(output_dir, f"{stem}.rst")
        content = _domain_page_rst(domain, domain_schemas, schemas_with_history)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

        # Build latest-version lookup for this domain
        latest_by_name: dict[str, Any] = {s.name: s for s in domain_schemas}

        # Write one RST page per operator inside a per-domain subdirectory
        op_dir = os.path.join(output_dir, stem)
        os.makedirs(op_dir, exist_ok=True)
        for s in domain_schemas:
            op_path = os.path.join(op_dir, f"{s.name}.rst")
            op_content = _operator_page_rst(s, domain, schemas_with_history)
            with open(op_path, "w", encoding="utf-8") as fh:
                fh.write(op_content)

        # Write one RST page per past version of every operator
        for s in domain_schemas:
            older = [
                h
                for h in schemas_with_history
                if h.domain == domain and h.name == s.name and h.since_version != s.since_version
            ]
            for old in older:
                ver_path = os.path.join(op_dir, f"{s.name}-{old.since_version}.rst")
                ver_content = _operator_version_page_rst(old, domain, latest_by_name[s.name])
                with open(ver_path, "w", encoding="utf-8") as fh:
                    fh.write(ver_content)

    # Write the top-level index
    _report("Writing operators index page.")
    index_path = os.path.join(output_dir, "index.rst")
    index_content = _index_page_rst(list(by_domain.keys()))
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(index_content)
    _report("Finished generating operator pages.")
