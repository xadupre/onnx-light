#!/usr/bin/env python3
"""Generates onnx_light/onnx/defs/all_schemas_registration.cc from the onnx package.

Run this script whenever the onnx package is updated to a new version to keep
the registered schemas in sync with the reference implementation.

Usage::

    python tools/gen_all_schemas_registration.py

Requires ``onnx`` to be installed (``pip install onnx``).
"""

import pathlib
import sys

try:
    import onnx.defs
except ModuleNotFoundError:
    print("ERROR: onnx package is not installed. Run: pip install onnx", file=sys.stderr)
    sys.exit(1)

OUTPUT = (
    pathlib.Path(__file__).parent.parent
    / "onnx_light"
    / "onnx"
    / "defs"
    / "all_schemas_registration.cc"
)


def escape_raw(s: str) -> str:
    """Returns s as a C++ raw string literal with a unique delimiter."""
    for delim in ["DOC", "RAWDOC", "STR", "S", ""]:
        if f'){delim}"' not in s:
            return f'R"{delim}({s}){delim}"'
    raise ValueError(f"Cannot find a safe C++ raw-string delimiter for: {s[:50]!r}")


def esc(s: str) -> str:
    """Escapes a string for use inside a C++ double-quoted string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")


def opt_cpp(opt_str: str) -> str:
    """Converts FormalParameterOption enum to its C++ onnx-light name."""
    return {
        "FormalParameterOption.Single": "OpSchema::Single",
        "FormalParameterOption.Optional": "OpSchema::Optional",
        "FormalParameterOption.Variadic": "OpSchema::Variadic",
    }.get(str(opt_str), "OpSchema::Single")


def generate() -> str:
    """Generates the C++ source text for all_schemas_registration.cc."""
    schemas = onnx.defs.get_all_schemas_with_history()
    lines = [
        "// Auto-generated from onnx package: full ONNX operator schema registration.",
        f"// Total schemas: {len(schemas)}",
        "// DO NOT EDIT -- regenerate with tools/gen_all_schemas_registration.py",
        "",
        '#include "onnx/defs/schema.h"',
        "",
        "namespace ONNX_LIGHT_NAMESPACE {",
        "",
        "void RegisterAllOnnxOperatorSchemas() {",
        "",
    ]

    for s in schemas:
        parts = [
            "  RegisterSchema(OpSchema()",
            f'      .SetName("{esc(s.name)}")',
            f'      .SetDomain("{esc(s.domain)}")',
            f"      .SinceVersion({s.since_version})",
        ]
        for i, inp in enumerate(s.inputs):
            parts.append(
                f'      .Input({i}, "{esc(inp.name)}", "{esc(inp.description)}",'
                f' "{esc(inp.type_str)}", {opt_cpp(inp.option)})'
            )
        for i, out in enumerate(s.outputs):
            parts.append(
                f'      .Output({i}, "{esc(out.name)}", "{esc(out.description)}",'
                f' "{esc(out.type_str)}", {opt_cpp(out.option)})'
            )
        for tc in s.type_constraints:
            allowed = "{" + ", ".join(f'"{esc(t)}"' for t in tc.allowed_type_strs) + "}"
            parts.append(
                f'      .TypeConstraint("{esc(tc.type_param_str)}", {allowed},'
                f' "{esc(tc.description)}")'
            )
        if s.doc:
            parts.append(f"      .SetDoc({escape_raw(s.doc)})")
        if str(s.support_level) != "SupportType.COMMON":
            parts.append("      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)")
        if s.non_deterministic:
            parts.append("      .SetNodeDeterminism(OpSchema::NodeDeterminism::NonDeterministic)")
        if s.deprecated:
            parts.append("      .Deprecate()")
        if s.has_function:
            # Add an empty function body so HasFunction() returns true, matching
            # the reference onnx package behaviour.
            parts.append("      .FunctionBody({})")
        lines.append("\n".join(parts) + ", 0, false, false);")
        lines.append("")

    lines += ["}", "", "} // namespace ONNX_LIGHT_NAMESPACE"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Generates and writes all_schemas_registration.cc."""
    content = generate()
    OUTPUT.write_text(content, encoding="utf-8")
    n = content.count("\n")
    print(f"Written {n} lines to {OUTPUT}")


if __name__ == "__main__":
    main()
