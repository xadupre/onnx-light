# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Guards against defined operator schemas missing from the registration headers.

Adapted from onnx/onnx#8373. Unlike the debug-only runtime count assertion, this
source-level check runs without a compiled extension and reports missing
(name, version) pairs in every build configuration.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS_DIR = REPO_ROOT / "onnx_light" / "onnx_lib" / "defs"

# Preview and preview-training share a class-name macro but use separate headers.
_MACRO_DOMAIN_AND_HEADER = {
    "ONNX_OPERATOR_SET_SCHEMA": ("Onnx", "operator_sets.h"),
    "ONNX_ML_OPERATOR_SET_SCHEMA": ("OnnxML", "operator_sets_ml.h"),
    "ONNX_TRAINING_OPERATOR_SET_SCHEMA": ("OnnxTraining", "operator_sets_training.h"),
    "ONNX_PREVIEW_OPERATOR_SET_SCHEMA": ("OnnxPreview", "operator_sets_preview.h"),
    "ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA": ("OnnxPreview", "operator_sets_training.h"),
}
_MACROS_WITHOUT_SCHEMAS = {"ONNX_TRAINING_OPERATOR_SET_SCHEMA"}

_DEFINITION_RE = re.compile(
    r"\b(" + "|".join(_MACRO_DOMAIN_AND_HEADER) + r")\s*\(\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,"
)
# Only a full fn(GetOpSchema<...>()) call counts, not a forward declaration.
_REGISTRATION_RE = re.compile(
    r"\bfn\s*\(\s*GetOpSchema\s*<\s*ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME\s*\("
    r"\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*>\s*\(\s*\)\s*\)"
)
_PREVIEW_REGISTRATION_RE = re.compile(
    r"\bfn\s*\(\s*GetOpSchema\s*<\s*ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME\s*\("
    r"\s*(\d+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*>\s*\(\s*\)\s*\)"
)
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def _defined_schemas(source: str) -> dict[str, set[tuple[str, int]]]:
    """Returns the (name, version) pairs defined by each schema macro."""
    defined: dict[str, set[tuple[str, int]]] = {
        macro: set() for macro in _MACRO_DOMAIN_AND_HEADER
    }
    for match in _DEFINITION_RE.finditer(_COMMENT_RE.sub(" ", source)):
        macro, name, version = match.groups()
        defined[macro].add((name, int(version)))
    return defined


def _registered_schemas(source: str, domain: str) -> set[tuple[str, int]]:
    """Returns the (name, version) pairs passed to fn for the given domain."""
    source = _COMMENT_RE.sub(" ", source)
    registered = {
        (name, int(version))
        for dom, version, name in _REGISTRATION_RE.findall(source)
        if dom == domain
    }
    if domain == "OnnxPreview":
        registered.update(
            (name, int(version)) for version, name in _PREVIEW_REGISTRATION_RE.findall(source)
        )
    return registered


class TestSchemaRegistration(unittest.TestCase):
    def test_all_defined_schemas_are_registered(self):
        defined = {macro: set() for macro in _MACRO_DOMAIN_AND_HEADER}
        for source_file in DEFS_DIR.rglob("*.cc"):
            for macro, schemas in _defined_schemas(
                source_file.read_text(encoding="utf-8")
            ).items():
                defined[macro].update(schemas)

        for macro, (domain, header_name) in _MACRO_DOMAIN_AND_HEADER.items():
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
        for macro in _MACRO_DOMAIN_AND_HEADER:
            with self.subTest(macro=macro):
                source = f"""
                // {macro}(LineComment, 1, OpSchema());
                /* {macro}(BlockComment, 2,
                    OpSchema()); */
                {macro} (
                    Live, /* version */ 3, OpSchema());
                """
                self.assertEqual(_defined_schemas(source)[macro], {("Live", 3)})

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
