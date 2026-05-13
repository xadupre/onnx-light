import re
import unittest
from pathlib import Path

import onnx
import onnx.defs as onnx_defs
import onnx_light.onnx.helper as onnx_light_helper


class TestSchemaSyncWithOnnx(unittest.TestCase):
    def test_onnx_light_versions_match_onnx(self):
        self.assertEqual(onnx_light_helper._onnx_ir_version(), onnx.IR_VERSION)
        self.assertEqual(onnx_light_helper._onnx_opset_version(), onnx_defs.onnx_opset_version())

    def test_onnx_light_operator_and_attribute_signatures_match_onnx(self):
        target_version = onnx_defs.onnx_opset_version()
        onnx_light_schemas = self._collect_operator_schemas(
            Path(__file__).resolve().parents[2] / "onnx_light" / "onnx" / "defs", target_version
        )
        onnx_schemas = self._collect_operator_schemas(
            Path(onnx.__file__).resolve().parent / "defs", target_version
        )

        self.assertEqual(set(onnx_light_schemas), set(onnx_schemas))
        for op_name in sorted(onnx_schemas):
            with self.subTest(op_name=op_name):
                self.assertEqual(onnx_light_schemas[op_name], onnx_schemas[op_name])

    @classmethod
    def _collect_operator_schemas(
        cls, defs_root: Path, max_opset_version: int
    ) -> dict[str, tuple[int, set[str]]]:
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

    @staticmethod
    def _find_matching_parenthesis(source: str, open_paren: int) -> int:
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

    @staticmethod
    def _split_first_arguments(args: str) -> tuple[str, str, str] | None:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
