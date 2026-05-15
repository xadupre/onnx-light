"""Generates RST documentation pages for ONNX operator schemas.

This module is called from ``docs/conf.py`` during the Sphinx build to produce
one RST file per operator domain (e.g. ``ai.onnx``, ``ai.onnx.ml``) plus a
top-level ``index.rst`` that references them all.  The generated files are
written to ``docs/operators/`` so they can be discovered by Sphinx toctrees.

Schema data is obtained from the ``onnx`` package when it is available.  If
``onnx`` is not installed the generation is skipped and no files are written.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
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

_OPTION_SUFFIX: dict[Any, str] = {}


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


def _escape_rst(text: str) -> str:
    """Lightly escapes characters that can confuse RST parsers."""
    # Backslash-escape lone backticks to avoid accidental interpreted text.
    return text.replace("\\", "\\\\")


def _strip_html(text: str) -> str:
    """Removes simple HTML tags (e.g. ``<br>``) from *text*."""
    import re

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


# ---------------------------------------------------------------------------
# RST generation for a single operator schema
# ---------------------------------------------------------------------------


def _schema_to_rst(schema: Any, include_history: bool = False) -> str:
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


# ---------------------------------------------------------------------------
# Domain-level RST page
# ---------------------------------------------------------------------------


def _domain_page_rst(domain: str, schemas: list[Any], all_schemas_with_history: list[Any]) -> str:
    """Returns full RST content for a single domain page."""
    title = _domain_title(domain)
    header_char = "="
    lines: list[str] = []
    lines.append(title)
    lines.append(header_char * len(title))
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

    # Detailed sections – latest version first per operator, then older versions
    history_by_name: dict[str, list[Any]] = {}
    for s in all_schemas_with_history:
        if s.domain == domain:
            history_by_name.setdefault(s.name, []).append(s)

    lines.append("Operator Details")
    lines.append("----------------")
    lines.append("")

    for s in sorted_schemas:
        # Anchor label so summary table links work
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


# ---------------------------------------------------------------------------
# Top-level index page
# ---------------------------------------------------------------------------


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
        title = _domain_title(domain)
        lines.append(f"- :doc:`{title} <{stem}>`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate(output_dir: str) -> None:
    """Generates operator RST pages into *output_dir*.

    Args:
        output_dir: Directory where the generated ``.rst`` files are written.
            It is created if it does not already exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import onnx.defs as _defs
    except ImportError:
        import warnings

        warnings.warn(
            "The 'onnx' package is not installed; operator documentation pages "
            "will not be generated.",
            stacklevel=2,
        )
        # Write a minimal placeholder index so the Sphinx toctree does not break.
        index_path = os.path.join(output_dir, "index.rst")
        if not os.path.exists(index_path):
            with open(index_path, "w", encoding="utf-8") as fh:
                fh.write(
                    ".. _l-onnx-operators:\n\n"
                    "ONNX Operators\n"
                    "==============\n\n"
                    "Operator documentation is not available because the ``onnx`` "
                    "package is not installed in this build environment.\n"
                )
        return

    schemas = _defs.get_all_schemas()
    schemas_with_history = _defs.get_all_schemas_with_history()

    if not schemas:
        return

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
