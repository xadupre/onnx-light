# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Parses vendored C++ schema definitions and registration calls."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS_DIR = REPO_ROOT / "onnx_light" / "onnx_lib" / "defs"

# Preview and preview-training share a class-name macro but use separate headers.
MACRO_INFO = {
    "ONNX_OPERATOR_SET_SCHEMA": ("Onnx", "", "operator_sets.h"),
    "ONNX_ML_OPERATOR_SET_SCHEMA": ("OnnxML", "ai.onnx.ml", "operator_sets_ml.h"),
    "ONNX_TRAINING_OPERATOR_SET_SCHEMA": (
        "OnnxTraining",
        "ai.onnx.training",
        "operator_sets_training.h",
    ),
    "ONNX_PREVIEW_OPERATOR_SET_SCHEMA": (
        "OnnxPreview",
        "ai.onnx.preview",
        "operator_sets_preview.h",
    ),
    "ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA": (
        "OnnxPreview",
        "ai.onnx.preview.training",
        "operator_sets_training.h",
    ),
}

_DEFINITION_RE = re.compile(
    r"\b(" + "|".join(MACRO_INFO) + r")\s*\(\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,"
)
# Only a full fn(GetOpSchema<...>()) call counts, not a forward declaration.
_REGISTRATION_RE = re.compile(
    r"\bfn\s*\(\s*GetOpSchema\s*<\s*([A-Za-z0-9_]+)\s*\(([^()]*)\)\s*>\s*\(\s*\)\s*\)"
)
_CLASS_NAME_MACROS = {
    "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME": re.compile(
        r"([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,\s*([A-Za-z0-9_]+)"
    ),
    "ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME": re.compile(r"(\d+)\s*,\s*([A-Za-z0-9_]+)"),
}
_QUOTED_RE = re.compile(
    r'(?:u8|u|U|L)?(?:R"([^\s()\\]{0,16})\(|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')'
)


def strip_comments(text: str) -> str:
    """Removes C++ comments while preserving quoted and raw string literals."""
    out: list[str] = []
    pos = 0
    while pos < len(text):
        if text.startswith("//", pos):
            end = text.find("\n", pos + 2)
            pos = len(text) if end == -1 else end
            out.append(" ")
        elif text.startswith("/*", pos):
            end = text.find("*/", pos + 2)
            pos = len(text) if end == -1 else end + 2
            out.append(" ")
        else:
            match = _QUOTED_RE.match(text, pos)
            if match is None:
                out.append(text[pos])
                pos += 1
                continue
            start = pos
            pos = match.end()
            if match.group(1) is not None:
                close = f'){match.group(1)}"'
                end = text.find(close, pos)
                pos = len(text) if end == -1 else end + len(close)
            out.append(text[start:pos])
    return "".join(out)


def defined_schemas(source: str) -> dict[str, set[tuple[str, int]]]:
    """Returns the (name, version) pairs defined by each schema macro."""
    defined: dict[str, set[tuple[str, int]]] = {macro: set() for macro in MACRO_INFO}
    for match in _DEFINITION_RE.finditer(strip_comments(source)):
        macro, name, version = match.groups()
        defined[macro].add((name, int(version)))
    return defined


def all_defined_schemas() -> dict[str, set[tuple[str, int]]]:
    """Collects schema definitions from the vendored C++ sources."""
    defined = {macro: set() for macro in MACRO_INFO}
    for source_file in DEFS_DIR.rglob("*.cc"):
        for macro, schemas in defined_schemas(source_file.read_text(encoding="utf-8")).items():
            defined[macro].update(schemas)
    return defined


def registered_schemas(source: str, domain: str) -> set[tuple[str, int]]:
    """Returns the (name, version) pairs passed to fn for the given domain."""
    registered = set()
    for macro, args in _REGISTRATION_RE.findall(strip_comments(source)):
        if macro not in _CLASS_NAME_MACROS:
            raise AssertionError(
                f"Unknown class-name macro {macro}; extend _CLASS_NAME_MACROS "
                "so registrations are not silently ignored."
            )
        parsed = _CLASS_NAME_MACROS[macro].fullmatch(args.strip())
        if parsed is None:
            raise AssertionError(f"Cannot parse the arguments of {macro}({args}).")
        if macro == "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME":
            dom, version, name = parsed.groups()
        else:
            dom = "OnnxPreview"
            version, name = parsed.groups()
        if dom == domain:
            registered.add((name, int(version)))
    return registered
