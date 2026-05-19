import re
import unittest
from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path
import onnx
import onnx.defs as onnx_defs
import onnx_light.onnx


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

                attributes = set(re.findall(r'\.Attr\(\s*"([^"]+)"', schema_body))
                blocks.append((op_name, opset_version, attributes))

            index = close_paren + 1

    @classmethod
    def _find_matching_parenthesis(cls, source: str, open_paren: int) -> int:
        """Finds the matching closing parenthesis index and returns -1 if not found."""
        depth = 0
        in_string = False
        escaped = False
        quote = ""

        for position in range(open_paren, len(source)):
            char = source[position]
            if in_string:
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
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return position

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
