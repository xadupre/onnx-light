import re
import unittest
from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path
import onnx
import onnx.defs as onnx_defs
import onnx_light.onnx.defs


class TestSchemaSyncWithOnnxCode(ExtTestCase):
    _KNOWN_MISSING_ONNX_28_SCHEMAS = {"Cast", "DequantizeLinear", "QuantizeLinear"}

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
        cls, defs_root: Path, max_opset_version: int
    ) -> dict[str, tuple[int, set[str]]]:
        """Collects the latest operator schemas up to a maximum opset version."""
        schemas: dict[str, tuple[int, set[str]]] = {}
        for source_file in sorted(defs_root.rglob("*.cc")):
            source = source_file.read_text(encoding="utf-8", errors="ignore")
            for op_name, opset_version, attributes in cls._extract_schema_blocks(source):
                if opset_version > max_opset_version:
                    continue
                if op_name not in schemas or opset_version > schemas[op_name][0]:
                    schemas[op_name] = (opset_version, attributes)
                elif opset_version == schemas[op_name][0]:
                    schemas[op_name] = (opset_version, schemas[op_name][1] | attributes)
        return schemas

    @classmethod
    def _extract_schema_blocks(cls, source: str) -> list[tuple[str, int, set[str]]]:
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

                attributes = cls._extract_attributes(source, schema_body)
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
    def _extract_attributes(cls, source: str, schema_body: str) -> set[str]:
        """Extracts attributes from an inline schema or a zero-argument schema helper."""
        attributes = set(re.findall(r'\.Attr\(\s*"([^"]+)"', schema_body))
        helper_match = re.fullmatch(r"([A-Za-z_]\w*)\s*\(\s*\)\s*;?", schema_body.strip())
        if attributes or helper_match is None:
            return attributes

        helper_name = helper_match.group(1)
        definition = re.search(
            rf"\b(?:static\s+)?OpSchema\s+{re.escape(helper_name)}\s*\(\s*\)\s*\{{", source
        )
        if definition is None:
            return attributes

        open_brace = source.find("{", definition.start())
        close_brace = cls._find_matching_brace(source, open_brace)
        if close_brace < 0:
            return attributes
        return set(re.findall(r'\.Attr\(\s*"([^"]+)"', source[open_brace + 1 : close_brace]))

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
        onnx_light_schemas = self._collect_operator_schemas(
            Path(__file__).resolve().parents[2] / "onnx_light" / "onnx_lib" / "defs",
            target_version,
        )
        onnx_schemas = self._collect_operator_schemas(
            Path(onnx.__file__).resolve().parent / "defs", target_version
        )

        # SwiGLU (onnx#8202) and DynamicQuantizeLinear may not yet be present in
        # the installed onnx source package.
        if "SwiGLU" not in onnx_schemas:
            onnx_light_schemas.pop("SwiGLU", None)
        if "DynamicQuantizeLinear" not in onnx_schemas:
            onnx_light_schemas.pop("DynamicQuantizeLinear", None)
        for op_name in self._KNOWN_MISSING_ONNX_28_SCHEMAS:
            if (
                onnx_schemas.get(op_name, (0, set()))[0] == 28
                and onnx_light_schemas.get(op_name, (0, set()))[0] < 28
            ):
                onnx_schemas.pop(op_name)
                onnx_light_schemas.pop(op_name)

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
