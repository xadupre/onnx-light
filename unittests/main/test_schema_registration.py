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

import unittest

from unittests.main.schema_registration_utils import (
    DEFS_DIR,
    MACRO_INFO,
    REPO_ROOT,
    all_defined_schemas,
    defined_schemas,
    registered_schemas,
    strip_comments,
)

_MACROS_WITHOUT_SCHEMAS = {"ONNX_TRAINING_OPERATOR_SET_SCHEMA"}


class TestSchemaRegistration(unittest.TestCase):
    def test_all_defined_schemas_are_registered(self):
        defined = all_defined_schemas()

        for macro, (domain, _, header_name) in MACRO_INFO.items():
            with self.subTest(macro=macro):
                header = DEFS_DIR / header_name
                registered = registered_schemas(header.read_text(encoding="utf-8"), domain)
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
        for macro in MACRO_INFO:
            with self.subTest(macro=macro):
                source = f"""
                // {macro}(LineComment, 1, OpSchema());
                /* {macro}(BlockComment, 2,
                    OpSchema()); */
                {macro} (
                    Live, /* version */ 3, OpSchema());
                """
                self.assertEqual(defined_schemas(source)[macro], {("Live", 3)})

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
                self.assertEqual(strip_comments(source), f"{literal}  \n  live")
                source = f"SetDoc({literal}); ONNX_OPERATOR_SET_SCHEMA(Live, 3, OpSchema());"
                self.assertEqual(
                    defined_schemas(source)["ONNX_OPERATOR_SET_SCHEMA"], {("Live", 3)}
                )

    def test_strip_comments_preserves_prefixed_literals_and_raw_delimiters(self):
        for prefix in ("", "u8", "u", "U", "L"):
            literals = [f'{prefix}"// literal /* literal */"', f"{prefix}'/'"]
            for delimiter in ("tag-!", "0123456789ABC!#?", 'a"b'):
                literals.append(f'{prefix}R"{delimiter}(" // literal\n/* literal */){delimiter}"')
            for literal in literals:
                with self.subTest(literal=literal):
                    source = f"{literal} // removed\n/* removed */ live"
                    self.assertEqual(strip_comments(source), f"{literal}  \n  live")
                    source = f"SetDoc({literal}); ONNX_OPERATOR_SET_SCHEMA(Live, 3, OpSchema());"
                    self.assertEqual(
                        defined_schemas(source)["ONNX_OPERATOR_SET_SCHEMA"], {("Live", 3)}
                    )

    def test_strip_comments_handles_unterminated_input(self):
        self.assertEqual(strip_comments("live // comment"), "live  ")
        self.assertEqual(strip_comments("live /* comment"), "live  ")
        self.assertEqual(strip_comments('R"DOC(// comment'), 'R"DOC(// comment')

    def test_registrations_reject_unknown_macro(self):
        source = "fn(GetOpSchema<UNKNOWN_CLASS_NAME(Onnx, 1, Example)>());"
        with self.assertRaisesRegex(
            AssertionError, "Unknown class-name macro UNKNOWN_CLASS_NAME"
        ):
            registered_schemas(source, "Onnx")

    def test_registrations_reject_malformed_arguments(self):
        for macro, args in (
            ("ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME", "Onnx, Example"),
            ("ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME", "invalid, Example"),
        ):
            with (
                self.subTest(macro=macro),
                self.assertRaisesRegex(AssertionError, "Cannot parse the arguments"),
            ):
                registered_schemas(f"fn(GetOpSchema<{macro}({args})>());", "Onnx")

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
        self.assertEqual(registered_schemas(source, "Onnx"), {("Live", 6)})
        self.assertEqual(registered_schemas(source, "OnnxML"), {("OtherDomain", 5)})

    def test_preview_registration_spellings(self):
        source = """
        class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Declared);
        // fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, LineComment)>());
        /* fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, BlockComment)>()); */
        fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Alias)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxPreview, 2, Explicit)>());
        """
        self.assertEqual(
            registered_schemas(source, "OnnxPreview"), {("Alias", 1), ("Explicit", 2)}
        )
        self.assertEqual(registered_schemas(source, "Onnx"), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
