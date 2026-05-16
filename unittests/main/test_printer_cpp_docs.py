import unittest
from pathlib import Path


class TestPrinterCppDocs(unittest.TestCase):
    def test_printer_header_has_file_level_documentation(self):
        repo = Path(__file__).resolve().parents[2]
        printer_header = repo / "onnx_light" / "onnx" / "defs" / "printer.h"
        content = printer_header.read_text(encoding="utf-8")
        self.assertIn("@file printer.h", content)
        self.assertIn(
            "@brief Declares stream-formatting helpers for ONNX protobuf structures.", content
        )

    def test_printer_header_has_operator_documentation(self):
        """Verifies that each operator<< overload has Doxygen return documentation."""
        repo = Path(__file__).resolve().parents[2]
        printer_header = repo / "onnx_light" / "onnx" / "defs" / "printer.h"
        content = printer_header.read_text(encoding="utf-8")
        self.assertIn("Formats and writes a ModelProto to an output stream.", content)
        self.assertIn("Formats and writes a GraphProto to an output stream.", content)
        self.assertIn("Formats and writes a NodeProto to an output stream.", content)
        self.assertIn("Formats and writes an AttributeProto to an output stream.", content)
        self.assertIn("@return A reference to os.", content)

    def test_printer_header_proto_to_string_has_tparam(self):
        """Verifies that ProtoToString has @tparam and @return Doxygen tags."""
        repo = Path(__file__).resolve().parents[2]
        printer_header = repo / "onnx_light" / "onnx" / "defs" / "printer.h"
        content = printer_header.read_text(encoding="utf-8")
        self.assertIn("@tparam ProtoType", content)
        self.assertIn("@return The formatted textual representation of proto.", content)

    def test_printer_cpp_page_has_intro(self):
        repo = Path(__file__).resolve().parents[2]
        printer_page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "printer.rst"
        content = printer_page.read_text(encoding="utf-8")
        self.assertIn("Printing helpers for ONNX protobuf objects", content)
        self.assertIn(":cpp:func:`onnx::ProtoToString`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
