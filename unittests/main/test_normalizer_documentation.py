import unittest
from pathlib import Path


class TestNormalizerDocumentation(unittest.TestCase):
    def test_normalizer_formulas(self):
        """Keeps both Normalizer documentation sources aligned with upstream formulas."""
        root = Path(__file__).resolve().parents[2]
        sources = (
            ("onnx_light/onnx_lib/defs/doc_strings.cc", "const char kDoc_Normalizer_ver1[]"),
            ("onnx_light/onnx_op/operator_sets_traditionalml_doc.cc", "MakeNormalizerDoc()"),
        )
        docs = []
        for relative_path, marker in sources:
            with self.subTest(source=relative_path):
                content = (root / relative_path).read_text(encoding="utf-8")
                doc = content.split(marker, 1)[1].split('R"DOC(', 1)[1].split(')DOC"', 1)[0]
                docs.append(doc)
                self.assertIn("L1:  Y = X / sum(abs(X))<br>", doc)
                self.assertIn("L2:  Y = X / sqrt(sum(X^2))<br>", doc)
                self.assertIn("Max: Y = X / max(X)<br>", doc)
                self.assertIn("if the divisor is zero, Y == X.", doc)
                self.assertIn("'max' and 'sum' reduce along the normalization axis", doc)
                self.assertIn("normalization is done along the C axis", doc)
        self.assertEqual(docs[0], docs[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
