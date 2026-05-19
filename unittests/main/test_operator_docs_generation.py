"""Tests for the operator documentation generator (onnx_light.doc)."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.doc as doc_module


class TestGenOperators(ExtTestCase):
    """Tests that generate_operators_doc generates valid RST operator documentation."""

    def _init(self, clean=False):
        from onnx_light.onnx.defs import register_onnx_operator_set_schema

        folder = self.get_dump_folder("test_gen_operators", clean=clean)
        register_onnx_operator_set_schema()
        doc_module.generate_operators_doc(folder)
        self.tmp_dir = folder

    def test_a_start(self):
        self._init(True)

    def test_generate_creates_files(self):
        self._init()
        files = os.listdir(self.tmp_dir)
        self.assertIn("index.rst", files, "index.rst must be generated")
        self.assertIn("ai_onnx.rst", files, "ai_onnx.rst must be generated")
        self.assertIn("ai_onnx_ml.rst", files, "ai_onnx_ml.rst must be generated")

    def test_index_lists_all_domains(self):
        self._init()
        index = Path(self.tmp_dir, "index.rst").read_text(encoding="utf-8")
        self.assertIn("ai_onnx", index)
        self.assertIn("ai_onnx_ml", index)
        self.assertIn("ai_onnx_preview", index)

    def test_ml_domain_page_contains_operators(self):
        self._init()
        content = Path(self.tmp_dir, "ai_onnx_ml.rst").read_text(encoding="utf-8")
        for name in ("Binarizer", "LabelEncoder", "TreeEnsembleClassifier"):
            self.assertIn(name, content, f"Expected operator {name!r} in ai_onnx_ml.rst")

        page = Path(self.tmp_dir, "ai_onnx_ml", "Binarizer.rst").read_text(encoding="utf-8")
        self.assertIn(".. _op_ai_onnx_ml_Binarizer:", page)

    def test_domain_page_contains_operators(self):
        self._init()
        content = Path(self.tmp_dir, "ai_onnx.rst").read_text(encoding="utf-8")
        for name in ("Abs", "Add", "Conv", "Relu"):
            self.assertIn(name, content, f"Expected operator {name!r} in ai_onnx.rst")

    def test_domain_page_contains_anchors(self):
        self._init()
        content = Path(self.tmp_dir, "ai_onnx", "Abs.rst").read_text(encoding="utf-8")
        self.assertIn(".. _op_ai_onnx_Abs:", content)
        content = Path(self.tmp_dir, "ai_onnx", "Add.rst").read_text(encoding="utf-8")
        self.assertIn(".. _op_ai_onnx_Add:", content)

    def test_operator_section_contains_inputs_outputs(self):
        self._init()
        # Check that operator pages include domain and version metadata
        content = Path(self.tmp_dir, "ai_onnx", "Add.rst").read_text(encoding="utf-8")
        self.assertIn("**Domain**", content)
        self.assertIn("**Since version**", content)
        self.assertIn("**Inputs**", content)
        self.assertIn("**Outputs**", content)
        self.assertIn("**Type Constraints**", content)
        self.assertIn("This operator supports **multidirectional", content)

        cast_content = Path(self.tmp_dir, "ai_onnx", "Cast.rst").read_text(encoding="utf-8")
        self.assertIn("**Attributes**", cast_content)
        self.assertIn("**to**", cast_content)
        self.assertIn("The operator casts the elements of ", cast_content)

    def test_individual_operator_pages_created(self):
        self._init()
        op_dir = Path(self.tmp_dir, "ai_onnx")
        self.assertTrue(op_dir.is_dir(), "ai_onnx/ subdirectory must exist")
        for name in ("Abs", "Add", "Conv", "Relu"):
            op_file = op_dir / f"{name}.rst"
            self.assertTrue(op_file.exists(), f"Individual page {name}.rst must exist")

    def test_past_version_pages_created(self):
        self._init()
        op_dir = Path(self.tmp_dir, "ai_onnx")
        # Add has multiple historical versions (1, 6, 7, 13, 14); at least one should exist
        past_files = list(op_dir.glob("Add-*.rst"))
        self.assertTrue(
            past_files, "At least one past-version page (e.g. Add-1.rst) must be created for Add"
        )
        # Each past-version file should link back to the latest version page
        for f in past_files:
            content = f.read_text(encoding="utf-8")
            self.assertIn(":doc:`Add`", content, f"{f.name} must link back to Add (latest)")
            self.assertIn("**Since version**", content)

    def test_latest_version_links_to_past_versions(self):
        self._init()
        content = Path(self.tmp_dir, "ai_onnx", "Add.rst").read_text(encoding="utf-8")
        self.assertIn("Version History", content)
        # Version history entries must be doc links, not plain text
        self.assertIn(":doc:`Version", content)
        # E.g. Add-1.rst should be referenced
        self.assertRegex(content, r":doc:`Version \d+ <Add-\d+>`")

    def test_domain_toctree_includes_past_versions(self):
        self._init()
        content = Path(self.tmp_dir, "ai_onnx.rst").read_text(encoding="utf-8")
        # The toctree should reference past-version pages (e.g. ai_onnx/Add-1)
        self.assertRegex(content, r"ai_onnx/Add-\d+")

    def test_domain_file_stem(self):
        self._init()
        self.assertEqual(doc_module._domain_file_stem(""), "ai_onnx")
        self.assertEqual(doc_module._domain_file_stem("ai.onnx.ml"), "ai_onnx_ml")
        self.assertEqual(
            doc_module._domain_file_stem("ai.onnx.preview.training"), "ai_onnx_preview_training"
        )

    def test_main_docs_index_references_operators(self):
        self._init()
        index_path = Path(__file__).resolve().parents[2] / "docs" / "index.rst"
        content = index_path.read_text(encoding="utf-8")
        self.assertIn("operators/index", content)

    def test_generate_reports_progress(self):
        from onnx_light.onnx.defs import register_onnx_operator_set_schema

        folder = self.get_dump_folder("test_gen_operators_progress", clean=True)
        register_onnx_operator_set_schema()
        messages = []
        doc_module.generate_operators_doc(folder, progress_callback=messages.append)
        self.assertTrue(messages)
        self.assertIn("Generating operator pages for", messages[0])
        self.assertTrue(any("Generating domain" in message for message in messages))
        self.assertEqual(messages[-1], "Finished generating operator pages.")

    def test_format_doc_translates_markdown_links_and_code(self):
        content = doc_module._format_doc(
            "See [the doc](Broadcasting.md).\nUse `X` and `Y` to compute `f(x)`."
        )
        self.assertIn("See `the doc <Broadcasting.md>`_.", content)
        self.assertIn("Use ``X`` and ``Y`` to compute ``f(x)``.", content)

    def test_format_doc_translates_fenced_code_block(self):
        content = doc_module._format_doc("Examples:\n```python\nx = 1\n```\nDone.")
        self.assertIn(".. code-block:: python", content)
        self.assertIn("    x = 1", content)
        self.assertIn("Done.", content)

    def test_format_doc_blank_line_after_bullet_list(self):
        # A bullet list immediately followed by a paragraph needs a blank line in RST.
        doc = "The following formats are supported:\n* BMP\n* PNG\nDecoded images follow."
        content = doc_module._format_doc(doc)
        # The paragraph must be preceded by a blank line.
        self.assertIn("* PNG\n\nDecoded images follow.", content)

    def test_format_doc_blank_line_not_added_when_already_present(self):
        # When a blank line already exists between bullet and paragraph, no duplicate is added.
        doc = "Supported types:\n* float\n* int\n\nMore details."
        content = doc_module._format_doc(doc)
        self.assertNotIn("\n\n\n", content)
        self.assertIn("* int\n\nMore details.", content)

    def test_format_doc_blank_line_after_bullet_before_code_block(self):
        # A bullet list immediately followed by a fenced code block needs a blank line.
        doc = "Notes:\n* pad_shape[i] is sum of pads\n```\nx = 1\n```"
        content = doc_module._format_doc(doc)
        self.assertIn("* pad_shape[i] is sum of pads\n\n.. code-block::", content)

    def test_format_doc_blank_line_before_code_block_after_paragraph(self):
        # A paragraph immediately followed by a fenced code block also needs a blank line.
        doc = "Examples:\nUse this:\n```python\nx = 1\n```"
        content = doc_module._format_doc(doc)
        self.assertIn("Use this:\n\n.. code-block:: python", content)

    def test_format_doc_no_blank_line_for_indented_continuation(self):
        # An indented continuation of a bullet item should NOT get an extra blank line.
        doc = "List:\n- Per-axis: scale is 1-D\n  with length Di.\nMore text."
        content = doc_module._format_doc(doc)
        # The continuation line should appear directly under the bullet
        self.assertIn("- Per-axis: scale is 1-D\n  with length Di.", content)
        # Followed by a blank line before 'More text.'
        self.assertIn("  with length Di.\n\nMore text.", content)

    def test_short_description_removes_inline_code_markers(self):
        doc = "Reverse batch of sequences having different lengths specified by `sequence_lens`."
        content = doc_module._short_description(doc)
        self.assertNotIn("`", content)
        self.assertEqual(
            content,
            "Reverse batch of sequences having different lengths specified by sequence_lens.",
        )

    def test_schema_section_multiline_descriptions_are_indented(self):
        schema = SimpleNamespace(
            doc="",
            inputs=[
                SimpleNamespace(
                    name="X",
                    type_str="tensor(float)",
                    option="Single",
                    description="Main input.\nUse values from the previous node.",
                )
            ],
            outputs=[],
            attributes={
                "alpha": SimpleNamespace(
                    type=1, description="Scaling factor.\n```python\nalpha = 0.5\n```"
                )
            },
            type_constraints=[],
        )
        content = "\n".join(doc_module._schema_section_lines(schema))
        self.assertIn("- **X** (*tensor(float)*):\n  Main input.\n  Use values", content)
        self.assertIn("- **alpha** (*float*):\n  Scaling factor.\n  \n  .. code-block::", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
