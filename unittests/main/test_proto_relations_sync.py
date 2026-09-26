# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path
import unittest
import xml.etree.ElementTree


class TestProtoRelationsSync(unittest.TestCase):
    """Checks the proto containment diagram against the message declarations."""

    ROOT = Path(__file__).resolve().parents[2]
    STATIC = ROOT / "docs/api/protos/_static"

    def _declared_edges(self):
        """Returns message-valued fields, resolving nested C++ message names."""
        header = (self.ROOT / "onnx_light/onnx_proto/onnx.h").read_text(encoding="utf-8")
        header = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', "", header)
        declarations = re.compile(
            r"^\s*(?:(BEGIN_PROTO(?:_NOINIT)?)\s*\(\s*(\w+)"
            r"|(END_PROTO)\s*\(\s*\)"
            r"|(FIELD(?:_[A-Z_]+)?)\s*\(\s*([\w:]+)\s*,\s*(\w+))",
            re.MULTILINE,
        )
        stack = []
        messages = set()
        fields = []
        for match in declarations.finditer(header):
            begin, name, end, field_macro, field_type, field_name = match.groups()
            if begin:
                stack.append(name)
                messages.add(".".join(stack))
            elif end:
                self.assertTrue(stack, "Unbalanced END_PROTO")
                stack.pop()
            elif field_macro:
                self.assertTrue(stack, "Field outside a proto")
                fields.append((".".join(stack), field_type.replace("::", "."), field_name))
        self.assertFalse(stack, "Unclosed proto declaration")
        self.assertGreater(len(messages), 30)
        aliases = {"TensorProto.Segment": "Segment", "TensorShapeProto.Dimension": "Dimension"}
        edges = set()
        for owner, field_type, field_name in fields:
            scope = owner.split(".")
            for depth in range(len(scope), -1, -1):
                target = ".".join([*scope[:depth], field_type])
                if target in messages:
                    edges.add((owner, aliases.get(target, target), field_name))
                    break
        return edges

    def _diagram_edges(self):
        """Returns DOT edges with their grouped labels and reference markers."""
        dot = (self.STATIC / "protos_relations.dot").read_text(encoding="utf-8")
        return [
            (source, target, label, "style=dashed" in attributes)
            for source, target, label, attributes in re.findall(
                r'"?([\w.]+)"?\s*->\s*"?([\w.]+)"?\s*\[label="([^"]+)"([^\]]*)\];', dot
            )
        ]

    def test_dot_matches_message_fields(self):
        diagram = {
            (source, target, field.strip())
            for source, target, label, reference in self._diagram_edges()
            if not reference
            for field in label.split(",")
        }
        self.assertSetEqual(self._declared_edges(), diagram)

    def test_svg_matches_dot(self):
        svg = xml.etree.ElementTree.parse(self.STATIC / "protos_relations.svg")
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        rendered = set()
        for group in svg.findall(".//svg:g[@class='edge']", namespace):
            title = group.find("svg:title", namespace)
            self.assertIsNotNone(title)
            source, target = title.text.split("->")
            label = "".join(
                "".join(text.itertext()) for text in group.findall("svg:text", namespace)
            )
            reference = any(
                "stroke-dasharray" in path.attrib for path in group.findall("svg:path", namespace)
            )
            rendered.add((source, target, label, reference))
        self.assertSetEqual(set(self._diagram_edges()), rendered)

    def test_text_trees_match_dot(self):
        index = (self.STATIC.parent / "index.rst").read_text(encoding="utf-8")
        tree = set()
        stack = []
        for line in index.splitlines():
            if line.strip() in {"ModelProto", "SequenceProto"}:
                stack = [line.strip()]
                continue
            edge = re.match(r"([ \u2502]*)[\u251c\u2514]\u2500\u2500 (.+) \u2192 (\S+)(.*)", line)
            if not edge:
                continue
            indent, fields, target, annotation = edge.groups()
            depth = (len(indent) - 3) // 4
            self.assertLess(depth, len(stack), f"Missing parent in {line!r}")
            for field in fields.split(","):
                tree.add((stack[depth], target, field.strip(), "reference" in annotation))
            stack = [*stack[: depth + 1], target]
        diagram = {
            (source, target, field.strip(), reference)
            for source, target, label, reference in self._diagram_edges()
            for field in label.partition(" (")[0].split(",")
        }
        self.assertSetEqual(diagram, tree)


if __name__ == "__main__":
    unittest.main()
