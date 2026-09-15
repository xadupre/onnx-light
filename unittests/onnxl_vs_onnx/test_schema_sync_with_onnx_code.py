import re
import unittest
from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path
from tempfile import TemporaryDirectory
import onnx
import onnx.defs as onnx_defs
import onnx_light.onnx.defs


class TestSchemaSyncWithOnnxCode(ExtTestCase):
    _REGISTERED_SCHEMA_PATTERNS = (
        (
            "operator_sets.h",
            r"class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME\(Onnx,\s*(\d+),\s*(\w+)\);",
            "",
            "version_op",
        ),
        (
            "operator_sets_ml.h",
            r"class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME\(OnnxML,\s*(\d+),\s*(\w+)\);",
            "ai.onnx.ml",
            "version_op",
        ),
        (
            "preview/defs.cc",
            r"ONNX_PREVIEW_OPERATOR_SET_SCHEMA\(\s*(\w+),\s*(\d+),",
            "ai.onnx.preview",
            "op_version",
        ),
        (
            "training/defs.cc",
            r"ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA\(\s*(\w+),\s*(\d+),",
            "ai.onnx.preview.training",
            "op_version",
        ),
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        onnx_light.onnx.defs.register_onnx_operator_set_schema()

    @staticmethod
    def _io_signature(parameters) -> tuple[tuple[str, str, int, int], ...]:
        return tuple(
            (parameter.name, parameter.type_str, int(parameter.option), parameter.min_arity)
            for parameter in parameters
        )

    @classmethod
    def _collect_operator_schemas(
        cls,
        defs_root: Path,
        max_opset_version: int,
        operator_versions: dict[str, int] | None = None,
    ) -> dict[str, tuple[int, set[str]]]:
        """Collects the latest operator schemas up to a maximum opset version."""
        schemas: dict[str, tuple[int, set[str]]] = {}
        sources = {
            path: path.read_text(encoding="utf-8", errors="ignore")
            for path in sorted(defs_root.rglob("*"))
            if path.suffix in {".cc", ".h"}
        }
        helper_sources = tuple(sources.values())
        for source_file, source in sources.items():
            if source_file.suffix != ".cc":
                continue
            for op_name, opset_version, attributes in cls._extract_schema_blocks(
                source, helper_sources
            ):
                version_limit = min(
                    max_opset_version, (operator_versions or {}).get(op_name, max_opset_version)
                )
                if opset_version > version_limit:
                    continue
                if op_name not in schemas or opset_version > schemas[op_name][0]:
                    schemas[op_name] = (opset_version, attributes)
                elif opset_version == schemas[op_name][0]:
                    schemas[op_name] = (opset_version, schemas[op_name][1] | attributes)
        return schemas

    @classmethod
    def _extract_schema_blocks(
        cls, source: str, helper_sources: tuple[str, ...] = ()
    ) -> list[tuple[str, int, set[str]]]:
        """Extracts schema macro tuples: operator name, opset version, and explicit attributes."""
        token = "ONNX_OPERATOR_SET_SCHEMA("
        blocks: list[tuple[str, int, set[str]]] = []
        index = 0
        while True:
            start = source.find(token, index)
            if start < 0:
                return blocks

            open_paren = start + len(token) - 1
            close_paren = cls._find_matching_parenthesis(source, open_paren)
            if close_paren < 0:
                return blocks

            args = source[open_paren + 1 : close_paren]
            split = cls._split_first_arguments(args)
            if split is not None:
                op_name, version_text, schema_body = split
                try:
                    opset_version = int(version_text)
                except ValueError:
                    index = close_paren + 1
                    continue

                attributes = cls._extract_attributes(source, schema_body, helper_sources)
                blocks.append((op_name, opset_version, attributes))

            index = close_paren + 1

    @classmethod
    def _find_matching_parenthesis(cls, source: str, open_paren: int) -> int:
        """Finds the matching closing parenthesis index and returns -1 if not found."""
        depth = 0
        in_string = False
        escaped = False
        quote = ""
        position = open_paren
        while position < len(source):
            if not in_string and source.startswith("//", position):
                newline = source.find("\n", position + 2)
                if newline < 0:
                    return -1
                position = newline + 1
                continue
            if not in_string and source.startswith("/*", position):
                comment_end = source.find("*/", position + 2)
                if comment_end < 0:
                    return -1
                position = comment_end + 2
                continue
            if not in_string and source.startswith('R"', position):
                delimiter_end = source.find("(", position + 2)
                if delimiter_end < 0:
                    return -1
                delimiter = source[position + 2 : delimiter_end]
                closing = ")" + delimiter + '"'
                raw_end = source.find(closing, delimiter_end + 1)
                if raw_end < 0:
                    return -1
                position = raw_end + len(closing)
                continue

            char = source[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    in_string = False
            elif char in {'"', "'"}:
                in_string = True
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return position
            position += 1

        return -1

    @classmethod
    def _extract_attributes(
        cls, source: str, schema_body: str, helper_sources: tuple[str, ...] = ()
    ) -> set[str]:
        """Extracts inline attributes and attributes from schema helpers and fillers."""
        attributes = set(re.findall(r'\.Attr\(\s*"([^"]+)"', schema_body))
        body = schema_body.strip()
        helper_match = re.match(r"((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*\(", body)
        helper_names = re.findall(
            r"\.FillUsing\(\s*((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*\(", schema_body
        )
        if helper_match is not None:
            close_paren = cls._find_matching_parenthesis(body, body.find("("))
            if close_paren >= 0 and not body[close_paren + 1 :].strip(";\n\r\t "):
                helper_names.append(helper_match.group(1))
        for qualified_name in helper_names:
            helper_name = qualified_name.rsplit("::", 1)[-1]
            for helper_source in (source, *helper_sources):
                helper_body = cls._find_helper_body(helper_source, helper_name)
                if helper_body is not None:
                    attributes.update(re.findall(r'\.Attr\(\s*"([^"]+)"', helper_body))
                    break
            else:
                raise ValueError(f"Cannot resolve schema helper {qualified_name!r}.")
        return attributes

    @classmethod
    def _find_helper_body(cls, source: str, helper_name: str) -> str | None:
        """Finds a helper definition, excluding declarations and call sites."""
        for candidate in re.finditer(rf"\b{re.escape(helper_name)}\s*\(", source):
            open_paren = source.find("(", candidate.start())
            close_paren = cls._find_matching_parenthesis(source, open_paren)
            if close_paren < 0:
                continue
            open_brace = close_paren + 1
            while open_brace < len(source) and source[open_brace].isspace():
                open_brace += 1
            if open_brace >= len(source) or source[open_brace] != "{":
                continue
            close_brace = cls._find_matching_brace(source, open_brace)
            if close_brace < 0:
                raise ValueError(f"Unterminated schema helper {helper_name!r}.")
            return source[open_brace + 1 : close_brace]
        return None

    @classmethod
    def _find_matching_brace(cls, source: str, open_brace: int) -> int:
        """Finds the matching closing brace while ignoring strings and comments."""
        depth = 0
        position = open_brace
        while position < len(source):
            if source.startswith("//", position):
                newline = source.find("\n", position + 2)
                if newline < 0:
                    return -1
                position = newline + 1
                continue
            if source.startswith("/*", position):
                comment_end = source.find("*/", position + 2)
                if comment_end < 0:
                    return -1
                position = comment_end + 2
                continue
            if source.startswith('R"', position):
                delimiter_end = source.find("(", position + 2)
                if delimiter_end < 0:
                    return -1
                delimiter = source[position + 2 : delimiter_end]
                closing = ")" + delimiter + '"'
                raw_end = source.find(closing, delimiter_end + 1)
                if raw_end < 0:
                    return -1
                position = raw_end + len(closing)
                continue
            if source[position] in {'"', "'"}:
                quote = source[position]
                position += 1
                escaped = False
                while position < len(source):
                    char = source[position]
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        position += 1
                        break
                    position += 1
                continue
            if source[position] == "{":
                depth += 1
            elif source[position] == "}":
                depth -= 1
                if depth == 0:
                    return position
            position += 1
        return -1

    @classmethod
    def _split_first_arguments(cls, args: str) -> tuple[str, str, str] | None:
        """Splits the first three top-level macro arguments or returns None."""
        parts = []
        current = []
        depth = 0
        in_string = False
        escaped = False
        quote = ""

        for char in args:
            if in_string:
                current.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    in_string = False
                continue

            if char in {'"', "'"}:
                in_string = True
                quote = char
                current.append(char)
                continue

            if char in "([{<":
                depth += 1
            elif char in ")]}>":
                depth = max(0, depth - 1)

            if char == "," and depth == 0 and len(parts) < 2:
                parts.append("".join(current).strip())
                current = []
                continue

            current.append(char)

        parts.append("".join(current).strip())
        if len(parts) < 3:
            return None
        return parts[0], parts[1], parts[2]

    @classmethod
    def _collect_registered_schema_keys(cls, defs_root: Path) -> set[tuple[str, str, int]]:
        """Collects the schema keys declared by the vendored registration headers."""
        keys: set[tuple[str, str, int]] = set()
        for schema_pattern in cls._REGISTERED_SCHEMA_PATTERNS:
            if len(schema_pattern) != 4:
                raise ValueError(f"Unexpected schema pattern {schema_pattern!r}.")
            relative_path, pattern, domain, field_order = schema_pattern
            source = (defs_root / relative_path).read_text(encoding="utf-8")
            matches = re.findall(pattern, source)
            if field_order == "version_op":
                keys.update(
                    (domain, op_name, int(version_text)) for version_text, op_name in matches
                )
            elif field_order == "op_version":
                keys.update(
                    (domain, op_name, int(version_text)) for op_name, version_text in matches
                )
            else:
                raise ValueError(f"Unexpected field order {field_order!r} for {relative_path!r}.")
        return keys

    def test_onnx_light_operator_and_attribute_signatures_match_onnx(self):
        target_version = onnx_defs.onnx_opset_version()
        onnx_schemas = self._collect_operator_schemas(
            Path(onnx.__file__).resolve().parent / "defs", target_version
        )
        onnx_light_schemas = self._collect_operator_schemas(
            Path(__file__).resolve().parents[2] / "onnx_light" / "onnx_lib" / "defs",
            target_version,
            {name: version for name, (version, _) in onnx_schemas.items()},
        )

        # Recently merged schemas may not yet be present in the installed onnx source package.
        if "SwiGLU" not in onnx_schemas:
            onnx_light_schemas.pop("SwiGLU", None)
        if "BitShift" not in onnx_schemas:
            onnx_light_schemas.pop("BitShift", None)
        if "DynamicQuantizeLinear" not in onnx_schemas:
            onnx_light_schemas.pop("DynamicQuantizeLinear", None)

        self.assertEqual(set(onnx_light_schemas), set(onnx_schemas))
        for op_name in sorted(onnx_schemas):
            with self.subTest(op_name=op_name):
                self.assertEqual(onnx_light_schemas[op_name], onnx_schemas[op_name])

    def test_registered_onnx_ops_match_onnx_code(self):
        defs_root = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx_lib" / "defs"
        onnx_light_schema_keys = {
            (schema.domain, schema.name, schema.since_version)
            for schema in onnx_light.onnx.defs.get_all_schemas_with_history()
        }
        source_schema_keys = self._collect_registered_schema_keys(defs_root)
        self.assertEqual(onnx_light_schema_keys, source_schema_keys)

    def test_extract_schema_attributes_from_helper(self):
        source = """
        static OpSchema MakeExampleSchema() {
          return OpSchema()
              .Attr("alpha", "first", AttributeProto::FLOAT)
              .Attr("beta", R"DOC({ ignored })DOC", AttributeProto::INT);
        }
        ONNX_OPERATOR_SET_SCHEMA(Example, 25, MakeExampleSchema());
        """
        self.assertEqual(
            self._extract_schema_blocks(source), [("Example", 25, {"alpha", "beta"})]
        )

    def test_extract_schema_attributes_from_namespaced_filler(self):
        source = """
        ONNX_OPERATOR_SET_SCHEMA(
            Example, 28,
            OpSchema().Attr("axis", "axis", AttributeProto::INT)
                .FillUsing(defs::math::utils::ExampleGenerator(AllTypes())));
        """
        helper_source = """
        std::function<void(OpSchema&)> ExampleGenerator(std::vector<std::string> types);
        void Other() { ExampleGenerator(AllTypes()); }
        std::function<void(OpSchema&)> ExampleGenerator(std::vector<std::string> types) {
            return [](OpSchema& schema) {
                schema.Attr("equation", "expression with { braces }", AttributeProto::STRING);
            };
        }
        """
        self.assertEqual(
            self._extract_schema_blocks(source, (helper_source,)),
            [("Example", 28, {"axis", "equation"})],
        )

    def test_extract_schema_attributes_from_parameterized_helper(self):
        source = """
        OpSchema MakeExampleSchema(bool recent) {
            return OpSchema().Attr("axis", "axis", AttributeProto::INT);
        }
        ONNX_OPERATOR_SET_SCHEMA(Example, 11, MakeExampleSchema(false));
        ONNX_OPERATOR_SET_SCHEMA(Example, 28, MakeExampleSchema(true));
        """
        self.assertEqual(
            self._extract_schema_blocks(source),
            [("Example", 11, {"axis"}), ("Example", 28, {"axis"})],
        )

    def test_extract_schema_attributes_rejects_missing_filler(self):
        source = "ONNX_OPERATOR_SET_SCHEMA(Example, 28, OpSchema().FillUsing(Missing()));"
        with self.assertRaisesRegex(ValueError, "Cannot resolve schema helper 'Missing'"):
            self._extract_schema_blocks(source)

    def test_collect_schema_uses_actual_historical_attributes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "defs.cc").write_text(
                'ONNX_OPERATOR_SET_SCHEMA(Example, 11, OpSchema().Attr("axis", "old"));\n'
                'ONNX_OPERATOR_SET_SCHEMA(Example, 28, OpSchema().Attr("mode", "new"));\n',
                encoding="utf-8",
            )
            self.assertEqual(
                self._collect_operator_schemas(root, 28, {"Example": 11}),
                {"Example": (11, {"axis"})},
            )
            self.assertEqual(
                self._collect_operator_schemas(root, 28), {"Example": (28, {"mode"})}
            )

    def test_preview_operators_separated_from_preview_training(self):
        defs_root = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx_lib" / "defs"
        preview_operator_pattern = (
            r"GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME\(1,\s*(\w+)\)>\(\)"
        )

        preview_source = (defs_root / "operator_sets_preview.h").read_text(encoding="utf-8")
        preview_ops = set(re.findall(preview_operator_pattern, preview_source))
        self.assertEqual(preview_ops, {"FlexAttention"})

        training_source = (defs_root / "operator_sets_training.h").read_text(encoding="utf-8")
        preview_training_ops = set(re.findall(preview_operator_pattern, training_source))
        self.assertEqual(preview_training_ops, {"Gradient", "Momentum", "Adagrad", "Adam"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
