import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestDoxygenConfig(ExtTestCase):
    def test_doxygen_input_includes_onnx_tree(self):
        """Tests that Doxygen indexes the ONNX tree used by C++ docs pages."""
        doxygen_path = Path(__file__).resolve().parents[2] / "docs" / "Doxyfile"
        content = doxygen_path.read_text(encoding="utf-8")
        self.assertIn("../onnx_light/onnx", content)
        self.assertIn("ONNX_API=", content)
        self.assertIn("ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain,ver,name)=name", content)

    def test_define_data_predefined_macro_is_single_value(self):
        """Verifies that the define_data PREDEFINED macro remains on a single Doxygen value."""
        doxygen_path = Path(__file__).resolve().parents[2] / "docs" / "Doxyfile"
        content = doxygen_path.read_text(encoding="utf-8")
        lines = [line for line in content.splitlines() if "define_data(type, field)=" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn(
            "define_data(type, field)=/** doc */ template <> inline type *Tensor::data<type>() {",
            content,
        )
        self.assertIn(
            "return field.data(); } template <> inline const type *Tensor::data<type>() const {",
            content,
        )
        self.assertIn('return field.data(); }"', lines[0])

    def test_doxygen_warnings_for_legacy_onnx_docs_are_suppressed(self):
        """Verifies that noisy legacy doc warnings are disabled for C++ docs generation."""
        doxygen_path = Path(__file__).resolve().parents[2] / "docs" / "Doxyfile"
        content = doxygen_path.read_text(encoding="utf-8")
        self.assertIn("WARN_IF_UNDOCUMENTED   = NO", content)
        self.assertIn("WARN_IF_DOC_ERROR      = NO", content)
        self.assertIn("WARN_NO_PARAMDOC       = NO", content)
        self.assertIn("WARN_IF_INCOMPLETE_DOC = NO", content)
        self.assertIn("DOT_GRAPH_MAX_NODES    = 100", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
