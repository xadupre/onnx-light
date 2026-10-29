# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Guards against defined operator schemas missing from the registration headers.

Adapted from onnx/onnx#8373 and #8635. Unlike the debug-only runtime count assertion, this
source-level check runs without a compiled extension and reports missing
(name, version) pairs in every build configuration. The companion binding test
also compares these definitions with the runtime registry, catching missing
opset registration calls and schemas registered under the wrong domain.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS_DIR = REPO_ROOT / "onnx_light" / "onnx_lib" / "defs"

# Preview and preview-training share a class-name macro but use separate headers.
_MACRO_INFO = {
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
_MACROS_WITHOUT_SCHEMAS = {"ONNX_TRAINING_OPERATOR_SET_SCHEMA"}

_DEFINITION_RE = re.compile(
    r"\b(" + "|".join(_MACRO_INFO) + r")\s*\(\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,"
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
_QUOTED_RE = re.compile(r'R"([A-Za-z0-9_]*)\(|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def _strip_comments(text: str) -> str:
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


def _defined_schemas(source: str) -> dict[str, set[tuple[str, int]]]:
    """Returns the (name, version) pairs defined by each schema macro."""
    defined: dict[str, set[tuple[str, int]]] = {macro: set() for macro in _MACRO_INFO}
    for match in _DEFINITION_RE.finditer(_strip_comments(source)):
        macro, name, version = match.groups()
        defined[macro].add((name, int(version)))
    return defined


def _all_defined_schemas() -> dict[str, set[tuple[str, int]]]:
    """Collects schema definitions from the vendored C++ sources."""
    defined = {macro: set() for macro in _MACRO_INFO}
    for source_file in DEFS_DIR.rglob("*.cc"):
        for macro, schemas in _defined_schemas(source_file.read_text(encoding="utf-8")).items():
            defined[macro].update(schemas)
    return defined


def _registered_schemas(source: str, domain: str) -> set[tuple[str, int]]:
    """Returns the (name, version) pairs passed to fn for the given domain."""
    registered = set()
    for macro, args in _REGISTRATION_RE.findall(_strip_comments(source)):
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


class TestSchemaRegistration(unittest.TestCase):
    def test_all_defined_schemas_are_registered(self):
        defined = _all_defined_schemas()

        for macro, (domain, _, header_name) in _MACRO_INFO.items():
            with self.subTest(macro=macro):
                header = DEFS_DIR / header_name
                registered = _registered_schemas(header.read_text(encoding="utf-8"), domain)
                if macro in _MACROS_WITHOUT_SCHEMAS:
                    self.assertFalse(
                        defined[macro],
                        f"{macro} now defines schemas; remove it from _MACROS_WITHOUT_SCHEMAS.",
                    )
                else:
                    self.assertTrue(
                        defined[macro],
                        f"Found no {macro} definitions under {DEFS_DIR}; "
                        "check the regex or glob.",
                    )
                    self.assertTrue(
                        registered,
                        f"Found no {domain} registrations in {header}; check the regex.",
                    )

                missing = defined[macro] - registered
                self.assertFalse(
                    missing,
                    f"{len(missing)} schema(s) defined via {macro} are missing from "
                    f"{header.relative_to(REPO_ROOT)}: {sorted(missing)}. "
                    "Each schema must be forward-declared and passed to "
                    "fn(GetOpSchema<...>()) inside the header's ForEachSchema.",
                )

    def test_definitions_ignore_comments(self):
        for macro in _MACRO_INFO:
            with self.subTest(macro=macro):
                source = f"""
                // {macro}(LineComment, 1, OpSchema());
                /* {macro}(BlockComment, 2,
                    OpSchema()); */
                {macro} (
                    Live, /* version */ 3, OpSchema());
                """
                self.assertEqual(_defined_schemas(source)[macro], {("Live", 3)})

    def test_strip_comments_preserves_literals(self):
        for literal in (
            '"https://onnx.ai/*docs*/"',
            r'"escaped \" quote // still a string"',
            "'/'",
            r"'\''",
            'R"(https://onnx.ai/"quoted" /*docs*/)"',
            'R"DOC(https://onnx.ai/"quoted"\n/*docs*/)DOC"',
        ):
            with self.subTest(literal=literal):
                source = f"{literal} // removed\n/* removed */ live"
                self.assertEqual(_strip_comments(source), f"{literal}  \n  live")
                source = f"SetDoc({literal}); ONNX_OPERATOR_SET_SCHEMA(Live, 3, OpSchema());"
                self.assertEqual(
                    _defined_schemas(source)["ONNX_OPERATOR_SET_SCHEMA"], {("Live", 3)}
                )

    def test_strip_comments_handles_unterminated_input(self):
        self.assertEqual(_strip_comments("live // comment"), "live  ")
        self.assertEqual(_strip_comments("live /* comment"), "live  ")
        self.assertEqual(_strip_comments('R"DOC(// comment'), 'R"DOC(// comment')

    def test_registrations_reject_unknown_macro(self):
        source = "fn(GetOpSchema<UNKNOWN_CLASS_NAME(Onnx, 1, Example)>());"
        with self.assertRaisesRegex(
            AssertionError, "Unknown class-name macro UNKNOWN_CLASS_NAME"
        ):
            _registered_schemas(source, "Onnx")

    def test_registrations_reject_malformed_arguments(self):
        for macro, args in (
            ("ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME", "Onnx, Example"),
            ("ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME", "invalid, Example"),
        ):
            with (
                self.subTest(macro=macro),
                self.assertRaisesRegex(AssertionError, "Cannot parse the arguments"),
            ):
                _registered_schemas(f"fn(GetOpSchema<{macro}({args})>());", "Onnx")

    def test_registrations_require_live_calls_and_matching_domain(self):
        source = """
        class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Declared);
        GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Referenced)>();
        // fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 3, LineComment)>());
        /* fn(GetOpSchema<
            ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 4, BlockComment)>()); */
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 5, OtherDomain)>());
        fn (
            GetOpSchema < ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME (
                Onnx, /* version */ 6, Live) > ( ) );
        """
        self.assertEqual(_registered_schemas(source, "Onnx"), {("Live", 6)})
        self.assertEqual(_registered_schemas(source, "OnnxML"), {("OtherDomain", 5)})

    def test_preview_registration_spellings(self):
        source = """
        class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Declared);
        // fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, LineComment)>());
        /* fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, BlockComment)>()); */
        fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Alias)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxPreview, 2, Explicit)>());
        """
        self.assertEqual(
            _registered_schemas(source, "OnnxPreview"), {("Alias", 1), ("Explicit", 2)}
        )
        self.assertEqual(_registered_schemas(source, "Onnx"), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
