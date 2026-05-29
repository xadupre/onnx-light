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
        folder = self.get_dump_folder("test_gen_operators", clean=clean)
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
        for name in ("LabelEncoder", "ZipMap"):
            self.assertIn(name, content, f"Expected operator {name!r} in ai_onnx_ml.rst")

        page = Path(self.tmp_dir, "ai_onnx_ml", "LabelEncoder.rst").read_text(encoding="utf-8")
        self.assertIn(".. _op_ai_onnx_ml_LabelEncoder:", page)

    def test_domain_page_contains_operators(self):
        self._init()
        content = Path(self.tmp_dir, "ai_onnx.rst").read_text(encoding="utf-8")
        for name in ("Abs", "Add", "Cast", "Mul"):
            self.assertIn(name, content, f"Expected operator {name!r} in ai_onnx.rst")

    def test_domain_page_table_is_sortable_and_searchable(self):
        self._init()
        # The domain summary table must carry the ``sphinx-datatable``
        # class so the ``sphinx_datatables`` extension renders it as an
        # interactive (sortable, searchable) table.
        for stem in ("ai_onnx.rst", "ai_onnx_ml.rst"):
            content = Path(self.tmp_dir, stem).read_text(encoding="utf-8")
            self.assertIn(".. list-table::", content)
            self.assertIn(":class: sphinx-datatable", content)

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
        self.assertIn("multidirectional", content)

    def test_individual_operator_pages_created(self):
        self._init()
        op_dir = Path(self.tmp_dir, "ai_onnx")
        self.assertTrue(op_dir.is_dir(), "ai_onnx/ subdirectory must exist")
        for name in ("Abs", "Add", "Cast", "Mul"):
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
        folder = self.get_dump_folder("test_gen_operators_progress", clean=True)
        messages = []
        doc_module.generate_operators_doc(folder, progress_callback=messages.append)
        self.assertTrue(messages)
        self.assertIn("Generating operator pages for", messages[0])
        self.assertTrue(any("Generating domain" in message for message in messages))
        self.assertIn("Finished generating operator pages", messages[-1])
        # On a clean run all pages are written and none should be skipped.
        self.assertIn("written", messages[-1])
        self.assertIn("0 skipped", messages[-1])

    def test_generate_skips_existing_pages(self):
        folder = self.get_dump_folder("test_gen_operators_skip", clean=True)
        # First call generates everything from scratch.
        doc_module.generate_operators_doc(folder)
        op_path = Path(folder, "ai_onnx", "Abs.rst")
        self.assertTrue(op_path.exists())
        # Replace the file content with a sentinel; a second call must NOT overwrite it.
        sentinel = "SENTINEL CONTENT - must not be overwritten"
        op_path.write_text(sentinel, encoding="utf-8")
        messages = []
        doc_module.generate_operators_doc(folder, progress_callback=messages.append)
        self.assertEqual(op_path.read_text(encoding="utf-8"), sentinel)
        # The final progress message reports how many pages were skipped.
        self.assertIn("skipped", messages[-1])

    def test_latest_page_contains_diff_with_previous_version(self):
        self._init()
        # Add v14 has a previous version (v13); the latest page must contain
        # a "Differences with previous version" section produced by
        # compare_schemas.
        content = Path(self.tmp_dir, "ai_onnx", "Add.rst").read_text(encoding="utf-8")
        self.assertIn("Differences with previous version (13)", content)
        self.assertIn("**SchemaDiff**", content)

    def test_version_page_contains_diff_with_previous_version(self):
        self._init()
        # Add v7's previous version is v6; the historical page must include
        # the corresponding diff section, and Add v6 a diff against v1.
        content_v7 = Path(self.tmp_dir, "ai_onnx", "Add-7.rst").read_text(encoding="utf-8")
        self.assertIn("Differences with previous version (6)", content_v7)
        self.assertIn("**SchemaDiff**", content_v7)

    def test_earliest_version_has_no_diff_section(self):
        self._init()
        # Add v1 is the earliest known version; no "Differences with previous
        # version" section should be emitted.
        content_v1 = Path(self.tmp_dir, "ai_onnx", "Add-1.rst").read_text(encoding="utf-8")
        self.assertNotIn("Differences with previous version", content_v1)

    def test_reducesum_diff_includes_attribute_changes(self):
        self._init()
        # ReduceSum v13 moved ``axes`` from an attribute to an optional input
        # and added the ``noop_with_empty_axes`` attribute. The diff section
        # on the latest (v13) page must report both changes — previously, no
        # attribute metadata was carried by ``LightOpSchema``.
        content = Path(self.tmp_dir, "ai_onnx", "ReduceSum.rst").read_text(encoding="utf-8")
        self.assertIn("Differences with previous version (11)", content)
        self.assertIn("**Attributes:**", content)
        self.assertIn("removed 'axes'", content)
        self.assertIn("added 'noop_with_empty_axes'", content)

    def test_operator_page_contains_backend_test_examples(self):
        self._init()
        # The Abs latest-version page should display the example sourced from
        # the C++ backend test case ``test_cc_abs``, including a code block
        # with the input/output tensors.
        content = Path(self.tmp_dir, "ai_onnx", "Abs.rst").read_text(encoding="utf-8")
        self.assertIn("Examples", content)
        self.assertIn("test_cc_abs", content)
        self.assertIn(".. code-block:: text", content)
        self.assertIn("Inputs:", content)
        self.assertIn("Outputs:", content)
        # The Abs backend test uses input ``x`` and output ``y``.
        self.assertIn("x: shape=(2, 3)", content)
        self.assertIn("y: shape=(2, 3)", content)

    def test_operator_page_without_backend_test_has_no_examples(self):
        self._init()
        # Pick an operator that has no backend test case registered (e.g.
        # ``Sin``).  Its page must not contain the "Examples" section.
        content = Path(self.tmp_dir, "ai_onnx", "Sin.rst").read_text(encoding="utf-8")
        self.assertNotIn("Examples\n--------", content)

    def test_format_doc_translates_markdown_links_and_code(self):
        content = doc_module._format_doc(
            "See [the doc](Broadcasting.md).\nUse `X` and `Y` to compute `f(x)`."
        )
        self.assertIn("See `the doc <Broadcasting.md>`_.", content)
        self.assertIn("Use ``X`` and ``Y`` to compute ``f(x)``.", content)

    def test_format_doc_separates_inline_code_followed_by_word_char(self):
        # ``NaN`` immediately followed by ``s`` (as in the TreeEnsemble
        # ``membership_values`` attribute description) would trigger a Sphinx
        # "Inline literal start-string without end-string" warning. RST
        # requires an escaped space between the closing ```` `` ```` and a
        # following word character.
        content = doc_module._format_doc("delimited by `NaN`s.")
        self.assertIn("``NaN``\\ s.", content)

    def test_format_doc_escapes_pipe_tokens(self):
        # |x| would be interpreted as an RST substitution reference and must be
        # wrapped in inline code so Sphinx renders it as ``|x|``.
        content = doc_module._format_doc("Compute softsign (x/(1+|x|)) and use |k| diagonals.")
        self.assertIn("(x/(1+``|x|``))", content)
        self.assertIn("use ``|k|`` diagonals", content)
        # Double-pipe norm notation must remain untouched.
        content = doc_module._format_doc("Norm ||X||_2^2 stays.")
        self.assertIn("||X||_2^2", content)

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

    def test_format_doc_blank_line_before_bullet_list_after_paragraph(self):
        # A bullet list immediately following a paragraph (no blank line in the
        # source) must be preceded by a blank line in the RST output. Without
        # it, docutils emits "Unexpected indentation" / "Block quote ends
        # without a blank line" warnings on bullet continuation lines (see
        # QuantizeLinear operator docs).
        doc = (
            "There are three supported granularities.\n"
            "In all cases, shapes match.\n"
            "- Per-tensor: scalar.\n"
            "- Per-axis: 1-D tensor of length Di. For an input shape\n"
            "  ``(D0, ..., Di, ..., Dn)``.\n"
            "- Blocked: shape identical except one dim."
        )
        content = doc_module._format_doc(doc)
        self.assertIn("In all cases, shapes match.\n\n- Per-tensor: scalar.", content)

    def test_format_doc_blank_line_before_nested_bullet_list(self):
        # Nested bullet lists (a deeper-indented bullet following a parent bullet)
        # require a blank line in RST. Without it, docutils emits
        # "Unexpected indentation" / "Block quote ends without a blank line"
        # warnings (see Cast operator docs).
        doc = (
            "Casting rules:\n"
            "* Casting from floating point to:\n"
            "  * floating point: OOR.\n"
            "  * fixed point: undefined.\n"
            "* Casting from bool to:\n"
            "  * floating point: 1.0/0.0.\n"
            "Then continue."
        )
        content = doc_module._format_doc(doc)
        # Blank line before opening the nested list.
        self.assertIn("* Casting from floating point to:\n\n  * floating point: OOR.", content)
        # Blank line when closing the nested list back to the outer level.
        self.assertIn("  * fixed point: undefined.\n\n* Casting from bool to:", content)
        self.assertIn("  * floating point: 1.0/0.0.\n\nThen continue.", content)

    def test_format_doc_no_blank_line_for_indented_continuation(self):
        # An indented continuation of a bullet item should NOT get an extra blank line.
        doc = "List:\n- Per-axis: scale is 1-D\n  with length Di.\nMore text."
        content = doc_module._format_doc(doc)
        # The continuation line should appear directly under the bullet
        self.assertIn("- Per-axis: scale is 1-D\n  with length Di.", content)
        # Followed by a blank line before 'More text.'
        self.assertIn("  with length Di.\n\nMore text.", content)

    def test_format_doc_dedents_uniformly_indented_input(self):
        # ONNX C++ raw doc strings often have a uniform leading indent on every
        # content line.  The formatter must dedent them so Sphinx doesn't render
        # the body as a literal block.
        doc = "\n    First paragraph.\n    Second line.\n\n    Another paragraph.\n    "
        content = doc_module._format_doc(doc)
        self.assertEqual(content, "First paragraph.\nSecond line.\n\nAnother paragraph.")

    def test_format_doc_wraps_indented_block_in_code_block_text(self):
        # Indented blocks (e.g. pseudo-code) outside fenced code should be
        # wrapped in a ``.. code-block:: text`` directive.
        doc = (
            "Pseudo code follows:\n"
            "  r = R / (1 + T);\n"
            "  H_new = H + G * G;\n"
            "Then continue."
        )
        content = doc_module._format_doc(doc)
        self.assertIn("Pseudo code follows:\n\n.. code-block:: text\n", content)
        self.assertIn("    r = R / (1 + T);", content)
        self.assertIn("    H_new = H + G * G;", content)
        # The directive must end with a blank line before the next paragraph.
        self.assertIn("    H_new = H + G * G;\n\nThen continue.", content)

    def test_format_doc_wraps_deeply_indented_bullet_continuation_in_code_block(self):
        # Some ONNX operators (notably Loop) have bullet items whose
        # "continuation" text is indented far past the bullet's content column
        # because it is actually pseudo-code. Without special handling docutils
        # emits "Unexpected indentation" and "Block quote ends without a blank
        # line" warnings.
        doc = (
            "Modes:\n"
            '* input ("", ""):\n'
            "        for (int i=0; ; ++i) {\n"
            "          cond = ...\n"
            "        }\n"
            "\n"
            '* input ("", cond):\n'
            "        bool cond = ...;\n"
            "Then continue."
        )
        content = doc_module._format_doc(doc)
        # Each deeply-indented continuation must be wrapped in a nested
        # ``.. code-block:: text`` directive indented at the bullet content column.
        self.assertIn('* input ("", ""):\n\n  .. code-block:: text\n', content)
        self.assertIn("      for (int i=0; ; ++i) {", content)
        self.assertIn("        cond = ...", content)
        self.assertIn('* input ("", cond):\n\n  .. code-block:: text\n', content)
        self.assertIn("      bool cond = ...;", content)
        # The trailing paragraph must follow a blank line.
        self.assertIn("\n\nThen continue.", content)

        # An indented block that contains blank lines should remain a single
        # ``.. code-block:: text`` directive with the blank lines preserved.
        doc = "Example:\n  step_1();\n\n  step_2();\nDone."
        content = doc_module._format_doc(doc)
        # Only one auto code-block directive should be emitted.
        self.assertEqual(content.count(".. code-block:: text"), 1)
        self.assertIn("    step_1();\n\n    step_2();", content)
        self.assertIn("    step_2();\n\nDone.", content)

    def test_format_doc_treats_numbered_items_as_list(self):
        # Some ONNX operators (notably Loop) use ``1) ...``, ``2) ...`` style
        # numbered enumerations whose continuation lines are aligned with the
        # bullet's text column (3 spaces). They must be rendered as a list with
        # the continuations kept as plain text, not wrapped in a code-block.
        doc = (
            "Modes:\n"
            "1) Trip count. Iteration count specified at runtime.\n"
            "   Note that a static trip count can be specified.\n"
            "2) Loop termination condition. Provided as input.\n"
            "   The body graph must yield a value for it.\n"
        )
        content = doc_module._format_doc(doc)
        # No auto code-block must be emitted for the regular continuations.
        self.assertNotIn(".. code-block:: text", content)
        # Both numbered items and their continuation lines are present verbatim.
        self.assertIn("1) Trip count. Iteration count specified at runtime.", content)
        self.assertIn("   Note that a static trip count can be specified.", content)
        self.assertIn("2) Loop termination condition. Provided as input.", content)
        self.assertIn("   The body graph must yield a value for it.", content)

    def test_format_doc_wraps_deeply_indented_numbered_continuation_in_code_block(self):
        # A numbered item whose continuation is indented well past the bullet's
        # text column is pseudo-code and must be wrapped in a nested
        # ``.. code-block:: text`` directive (same as starred bullets).
        doc = (
            "Modes:\n"
            "1) Loop with code:\n"
            "       for (int i=0; i < n; ++i) {\n"
            "         do_stuff();\n"
            "       }\n"
            "Then continue."
        )
        content = doc_module._format_doc(doc)
        self.assertIn("1) Loop with code:\n\n   .. code-block:: text\n", content)
        self.assertIn("       for (int i=0; i < n; ++i) {", content)
        self.assertIn("\n\nThen continue.", content)

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
            outputs=[
                SimpleNamespace(
                    name="Y",
                    type_str="tensor(float)",
                    option="Single",
                    description="Main output.\nProduced by the compute graph.",
                )
            ],
            attributes={
                "alpha": SimpleNamespace(
                    type=1, description="Scaling factor.\n```python\nalpha = 0.5\n```"
                )
            },
            type_constraints=[
                SimpleNamespace(
                    type_param_str="T",
                    description="Constrain input type.\nSupports float and int.",
                    allowed_type_strs=["tensor(float)", "tensor(int64)"],
                )
            ],
        )
        content = "\n".join(doc_module._schema_section_lines(schema))
        self.assertIn("- **X** (*tensor(float)*):\n  Main input.\n  Use values", content)
        self.assertIn("- **Y** (*tensor(float)*):\n  Main output.\n  Produced by", content)
        self.assertIn(
            "- **alpha** (*float*):\n"
            "  Scaling factor.\n"
            "  \n"
            "  .. code-block:: python\n"
            "  \n"
            "      alpha = 0.5",
            content,
        )
        self.assertIn(
            "- **T**:\n"
            "  Constrain input type.\n"
            "  Supports float and int.\n"
            "  Allowed types: tensor(float), tensor(int64).",
            content,
        )

    def test_schema_section_emits_output_arity_hint_when_variadic(self):
        schema = SimpleNamespace(
            doc="",
            inputs=[],
            outputs=[
                SimpleNamespace(
                    name="Y", type_str="tensor(float)", option="Variadic", description="Outputs."
                )
            ],
            attributes={},
            type_constraints=[],
            min_output=1,
            max_output=2**31 - 1,
        )
        content = "\n".join(doc_module._schema_section_lines(schema))
        self.assertIn("Between 1 and \u221e outputs.", content)

    def test_schema_section_omits_output_arity_hint_when_fixed(self):
        schema = SimpleNamespace(
            doc="",
            inputs=[],
            outputs=[
                SimpleNamespace(
                    name="Y", type_str="tensor(float)", option="Single", description="Output."
                )
            ],
            attributes={},
            type_constraints=[],
            min_output=1,
            max_output=1,
        )
        content = "\n".join(doc_module._schema_section_lines(schema))
        self.assertNotIn("Between", content)

    def test_format_doc_strips_trailing_underscore_in_words(self):
        # Words ending with a single ``_`` (e.g. ``nodes_``) would be parsed by
        # RST as unresolved hyperlink references. The formatter strips the
        # trailing underscore so docs for TreeEnsemble operators (which contain
        # phrases such as "All args with nodes_ are fields") render correctly.
        content = doc_module._format_doc("All args with nodes_ are fields.")
        self.assertEqual(content, "All args with nodes are fields.")
        # Internal underscores are preserved; only the trailing one is removed.
        content = doc_module._format_doc("Use classlabels_int64s_ here.")
        self.assertEqual(content, "Use classlabels_int64s here.")
        # Inline code spans are not modified.
        content = doc_module._format_doc("See `nodes_` and nodes_.")
        self.assertIn("``nodes_``", content)
        self.assertIn(" nodes.", content)
        # Words with multiple trailing underscores (e.g. ``__init__``) are
        # untouched since RST does not treat them as hyperlink references.
        content = doc_module._format_doc("Method __init__ stays as is.")
        self.assertIn("__init__", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
