from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import textwrap
from types import SimpleNamespace
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
# Matches pipe-delimited tokens such as ``|x|`` or ``|k|`` which RST would
# otherwise interpret as substitution references. Pipes adjacent to other
# pipes (e.g. the ``||X||`` notation) are intentionally not matched.
_PIPE_TOKEN_RE = re.compile(r"(?<!\|)\|([^|\s`]+)\|(?!\|)")
# Matches a word ending with a single ``_`` (e.g. ``nodes_``) where the
# underscore is the last character of the word. RST would otherwise treat
# such a word as an unresolved hyperlink reference. Per the upstream issue
# (``All args with nodes_ are fields``), the trailing ``_`` is stripped.
_TRAILING_UNDERSCORE_WORD_RE = re.compile(r"\b(\w*[A-Za-z0-9])_(?!\w)")
# Matches a single ``*`` not adjacent to another ``*``. Such asterisks appear
# verbatim in some ONNX op doc strings (e.g. Trilu's ``[*, N, M]``) where RST
# would otherwise parse them as inline emphasis start-strings, producing
# "Inline emphasis start-string without end-string" warnings. Double-asterisk
# bold markers (``**bold**``) are intentionally not matched.
_LONE_ASTERISK_RE = re.compile(r"(?<!\*)\*(?!\*)")
_RST_INLINE_CODE_SPLIT_RE = re.compile(r"(``[^`]*``)")
_RST_ROLE_PREFIX_RE = re.compile(r":[a-zA-Z][a-zA-Z0-9_]*:$")
_RST_CODE_BLOCK_INDENT = " " * 4
_RST_DIRECTIVE_PREFIX = ".. "
_BULLET_MARKERS = ("* ", "- ")
# Numbered list markers such as ``1)`` or ``2)`` used in some ONNX op doc
# strings (notably Loop). They are treated like bullet items so their
# indented continuation lines are not mistaken for a literal/code block.
_NUMBERED_BULLET_RE = re.compile(r"^\d+\)\s")
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
        # RST requires whitespace or punctuation after an inline literal's
        # closing ``. When the next character is a word character (e.g.
        # ``NaN``s in the TreeEnsemble docs), insert an escaped space so the
        # literal terminates cleanly without altering the rendered text.
        suffix = text[match.end() : match.end() + 1]
        trailing = "\\ " if suffix.isalnum() or suffix == "_" else ""
        return f"``{code_text}``{trailing}"

    return _strip_trailing_word_underscores(
        _escape_lone_asterisks(
            _escape_pipe_tokens(_MARKDOWN_INLINE_CODE_RE.sub(replace_inline_code, text))
        )
    )


def _escape_lone_asterisks(text: str) -> str:
    """Escapes solitary ``*`` characters outside RST inline code spans.

    ONNX op doc strings occasionally include literal asterisks as wildcards
    (e.g. Trilu's ``[*, N, M]`` or Imputer's ``[*,F]``) which RST otherwise
    parses as inline emphasis start-strings. Leading bullet markers
    (``* `` at the start of a line) and tokens already wrapped in
    ``...`` inline code are left untouched.
    """
    parts = _RST_INLINE_CODE_SPLIT_RE.split(text)
    for i in range(0, len(parts), 2):
        segment = parts[i]
        if i == 0:
            lstripped = segment.lstrip()
            ws_len = len(segment) - len(lstripped)
            if lstripped.startswith("* "):
                parts[i] = (
                    segment[: ws_len + 2] + _LONE_ASTERISK_RE.sub(r"\\*", segment[ws_len + 2 :])
                )
                continue
        parts[i] = _LONE_ASTERISK_RE.sub(r"\\*", segment)
    return "".join(parts)


def _strip_trailing_word_underscores(text: str) -> str:
    """Strips trailing ``_`` from words outside RST inline code spans.

    RST interprets a word terminated by a single ``_`` (e.g. ``nodes_``) as a
    hyperlink reference. ONNX TreeEnsemble docs use such constructs (e.g.
    "All args with nodes_ are fields"), which break Sphinx parsing. Tokens
    already wrapped in ``...`` inline code are left untouched.
    """
    parts = _RST_INLINE_CODE_SPLIT_RE.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = _TRAILING_UNDERSCORE_WORD_RE.sub(r"\1", parts[i])
    return "".join(parts)


def _escape_pipe_tokens(text: str) -> str:
    """Wraps ``|x|`` style tokens in inline code to avoid RST substitution refs.

    Tokens already inside double-backtick inline code spans are left untouched.
    """
    parts = _RST_INLINE_CODE_SPLIT_RE.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = _PIPE_TOKEN_RE.sub(r"``|\1|``", parts[i])
    return "".join(parts)


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
    # Column where the most recent bullet item's text content starts (i.e.
    # indent of the marker plus the marker width). This is the canonical
    # indentation for continuation paragraphs of that bullet.
    last_bullet_content_indent: int = 0
    # Track whether we are inside an auto-generated ``.. code-block:: text``
    # directive started because the source contained an indented block.
    in_auto_code_block = False
    auto_code_base_indent = 0
    # Track whether we are inside a nested ``.. code-block:: text`` directive
    # emitted for a deeply-indented block within a bullet item (see Loop op).
    in_bullet_code_block = False
    bullet_code_base_indent = 0
    bullet_code_prefix = ""

    def end_auto_code_block() -> None:
        nonlocal in_auto_code_block, auto_code_base_indent
        if in_auto_code_block:
            if lines and lines[-1] != "":
                lines.append("")
            in_auto_code_block = False
            auto_code_base_indent = 0

    def end_bullet_code_block() -> None:
        nonlocal in_bullet_code_block, bullet_code_base_indent, bullet_code_prefix
        if in_bullet_code_block:
            if lines and lines[-1] != "":
                lines.append("")
            in_bullet_code_block = False
            bullet_code_base_indent = 0
            bullet_code_prefix = ""

    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            end_auto_code_block()
            end_bullet_code_block()
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
            end_bullet_code_block()
            last_bullet_indent = None
            lines.append("")
            continue

        cur_indent = len(line) - len(line.lstrip())
        numbered_match = _NUMBERED_BULLET_RE.match(stripped)
        is_bullet = stripped.startswith(_BULLET_MARKERS) or bool(numbered_match)

        if is_bullet:
            end_auto_code_block()
            end_bullet_code_block()
            # RST requires a blank line before a bullet list when it follows a
            # paragraph, and whenever the bullet indentation changes (entering
            # a nested sub-list, or closing one back to the outer level).
            # Without it, docutils treats the bullet markers as plain text and
            # emits "Unexpected indentation" warnings on continuation lines.
            needs_blank = (
                last_bullet_indent is None
                or cur_indent != last_bullet_indent
            )
            if needs_blank and lines and lines[-1] != "":
                lines.append("")
            last_bullet_indent = cur_indent
            # Compute the column at which the bullet item's text content starts
            # so continuation paragraphs aligned with that column are recognised
            # as plain text rather than treated as deeply-indented pseudo-code.
            if numbered_match:
                marker_width = len(numbered_match.group(0))
            else:
                marker_width = 2  # "* " or "- "
            last_bullet_content_indent = cur_indent + marker_width
            lines.append(_format_markdown_inline(line))
            continue

        # Continuation of a bullet item: indented more than the bullet marker.
        if last_bullet_indent is not None and cur_indent > last_bullet_indent:
            bullet_content_indent = last_bullet_content_indent
            # A continuation indented well past the bullet's text column is
            # almost always pseudo-code (see the Loop operator). Wrap it in a
            # nested ``.. code-block:: text`` directive so docutils does not
            # emit "Unexpected indentation" / "Block quote ends without a
            # blank line" warnings.
            if cur_indent > bullet_content_indent:
                if not in_bullet_code_block:
                    if lines and lines[-1] != "":
                        lines.append("")
                    indent_prefix = " " * bullet_content_indent
                    lines.append(f"{indent_prefix}.. code-block:: text")
                    lines.append("")
                    in_bullet_code_block = True
                    bullet_code_base_indent = cur_indent
                    bullet_code_prefix = indent_prefix + _RST_CODE_BLOCK_INDENT
                relative_line = (
                    line[bullet_code_base_indent:]
                    if cur_indent >= bullet_code_base_indent
                    else line.lstrip()
                )
                lines.append(f"{bullet_code_prefix}{relative_line}")
                continue
            end_bullet_code_block()
            lines.append(_format_markdown_inline(line))
            continue

        # End any pending bullet list before processing the current line.
        if last_bullet_indent is not None:
            end_bullet_code_block()
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
    end_bullet_code_block()

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


def _previous_version_schema(
    schema: Any, domain: str, all_schemas_with_history: list[Any]
) -> Any | None:
    """Returns the most recent schema strictly older than *schema* for the same
    operator/domain, or ``None`` if *schema* is the earliest known version."""
    older = sorted(
        [
            h
            for h in all_schemas_with_history
            if h.domain == domain
            and h.name == schema.name
            and h.since_version < schema.since_version
        ],
        key=lambda x: x.since_version,
    )
    return older[-1] if older else None


def _differences_section_lines(prev_schema: Any, current_schema: Any) -> list[str]:
    """Returns RST lines for a "Differences with previous version" section.

    Uses :func:`onnx_light.onnx_lib.defs.schema_diff.compare_schemas` to
    produce a structured diff between *prev_schema* and *current_schema*.
    """
    # Local import to avoid a hard dependency on the C-extension at import
    # time of this module (the schema diff module pulls _onnxpy).
    from onnx_light.onnx_lib.defs.schema_diff import compare_schemas

    diff = compare_schemas(prev_schema, current_schema)
    title = f"Differences with previous version ({prev_schema.since_version})"
    lines: list[str] = [title, "-" * len(title), ""]
    lines.append(diff.to_rst())
    lines.append("")
    return lines


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

    # Diff section against the immediately previous opset version, if any.
    prev = _previous_version_schema(schema, domain, all_schemas_with_history)
    if prev is not None:
        lines.extend(_differences_section_lines(prev, schema))

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


def _operator_version_page_rst(
    schema: Any,
    domain: str,
    latest_schema: Any,
    all_schemas_with_history: list[Any] | None = None,
) -> str:
    """Returns full RST content for a past-version page of an operator.

    Args:
        schema: The historical OpSchema for this specific version.
        domain: The operator domain string.
        latest_schema: The current (latest) OpSchema for the same operator,
            used to generate a back-link to the latest version page.
        all_schemas_with_history: Full history of schemas used to locate the
            immediately previous version for the diff section.  When omitted
            (or when *schema* is the earliest known version), no diff section
            is emitted.

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

    # Diff section against the immediately previous opset version, if any.
    if all_schemas_with_history is not None:
        prev = _previous_version_schema(schema, domain, all_schemas_with_history)
        if prev is not None:
            lines.extend(_differences_section_lines(prev, schema))

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


def _load_light_schemas() -> tuple[list[Any], list[Any]]:
    """Loads operator schemas from the onnx_op C extension (LightOpSchema).

    Returns ``(latest_schemas, schemas_with_history)`` where each entry is a
    :class:`types.SimpleNamespace` exposing the same attributes used by the
    rest of this module: ``name``, ``domain``, ``since_version``, ``doc``,
    ``inputs`` (each with ``name``, ``type_str``, ``option``, ``description``),
    ``outputs`` (same shape), ``attributes`` (empty mapping; LightOpSchema does
    not carry attribute metadata), ``type_constraints`` (each with
    ``type_param_str``, ``allowed_type_strs`` as plain strings, and
    ``description``) and ``deprecated`` (always ``False``).
    """
    from onnx_light.onnx_py._onnxpy import onnx_op as _op  # type: ignore[attr-defined]

    raw_schemas = _op.GetAllOnnxOpSchemasWithHistory()

    def _adapt(s: Any) -> Any:
        inputs = [
            SimpleNamespace(
                name=p.name, type_str=p.type, option="Single", description=p.description
            )
            for p in s.inputs
        ]
        outputs = [
            SimpleNamespace(
                name=p.name, type_str=p.type, option="Single", description=p.description
            )
            for p in s.outputs
        ]
        type_constraints = [
            SimpleNamespace(
                type_param_str=t.type_param_str,
                allowed_type_strs=[_op.ToTypeString(x) for x in t.allowed_type_strs],
                description=t.description,
            )
            for t in s.type_constraints
        ]
        return SimpleNamespace(
            name=s.name,
            domain=s.domain,
            since_version=s.since_version,
            doc=s.doc,
            inputs=inputs,
            outputs=outputs,
            attributes={},
            type_constraints=type_constraints,
            deprecated=False,
        )

    schemas_with_history = [_adapt(s) for s in raw_schemas]
    latest: dict[tuple[str, str], Any] = {}
    for s in schemas_with_history:
        key = (s.domain, s.name)
        if key not in latest or s.since_version > latest[key].since_version:
            latest[key] = s
    return list(latest.values()), schemas_with_history


def generate_operators_doc(
    output_dir: str, progress_callback: Callable[[str], None] | None = None
) -> None:
    """Generates operator RST pages into *output_dir*.

    Reads all ONNX operator schemas from the lightweight ``onnx_op`` C
    extension (``LightOpSchema``) and writes one RST file per domain plus a
    top-level ``index.rst`` toctree.

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

    skipped = 0
    written = 0

    def _write_if_missing(path: str, content_factory: Callable[[], str]) -> None:
        """Writes *content_factory()* to *path* if the file does not already exist."""
        nonlocal skipped, written
        if os.path.exists(path):
            skipped += 1
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content_factory())
        written += 1

    schemas, schemas_with_history = _load_light_schemas()
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

        def _make_domain_page(
            domain: str = domain, domain_schemas: list[Any] = domain_schemas
        ) -> str:
            return _domain_page_rst(domain, domain_schemas, schemas_with_history)

        _write_if_missing(path, _make_domain_page)

        # Build latest-version lookup for this domain
        latest_by_name: dict[str, Any] = {s.name: s for s in domain_schemas}

        # Write one RST page per operator inside a per-domain subdirectory
        op_dir = os.path.join(output_dir, stem)
        os.makedirs(op_dir, exist_ok=True)
        for s in domain_schemas:
            op_path = os.path.join(op_dir, f"{s.name}.rst")

            def _make_operator_page(s: Any = s, domain: str = domain) -> str:
                return _operator_page_rst(s, domain, schemas_with_history)

            _write_if_missing(op_path, _make_operator_page)

        # Write one RST page per past version of every operator
        for s in domain_schemas:
            older = [
                h
                for h in schemas_with_history
                if h.domain == domain and h.name == s.name and h.since_version != s.since_version
            ]
            for old in older:
                ver_path = os.path.join(op_dir, f"{s.name}-{old.since_version}.rst")
                latest = latest_by_name[s.name]

                def _make_version_page(
                    old: Any = old, domain: str = domain, latest: Any = latest
                ) -> str:
                    return _operator_version_page_rst(
                        old, domain, latest, schemas_with_history
                    )

                _write_if_missing(ver_path, _make_version_page)

    # Write the top-level index
    _report("Writing operators index page.")
    index_path = os.path.join(output_dir, "index.rst")

    def _make_index_page() -> str:
        return _index_page_rst(list(by_domain.keys()))

    _write_if_missing(index_path, _make_index_page)
    _report(
        f"Finished generating operator pages "
        f"({written} written, {skipped} skipped because already present)."
    )
