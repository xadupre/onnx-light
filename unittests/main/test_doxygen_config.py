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
        """Tests that define_data PREDEFINED macro stays on a single Doxygen value."""
        doxygen_path = Path(__file__).resolve().parents[2] / "docs" / "Doxyfile"
        content = doxygen_path.read_text(encoding="utf-8")
        self.assertIn(
            "define_data(type, field)=/** doc */ template <> inline type *Tensor::data<type>() {",
            content,
        )
        self.assertIn(
            "return field.data(); } template <> inline const type *Tensor::data<type>() const {",
            content,
        )
        self.assertNotIn("return field.data\n", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
