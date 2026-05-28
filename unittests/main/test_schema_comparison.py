"""Tests for :mod:`onnx_light.schema_comparison`."""

import os
import re
import unittest

from onnx_light.ext_test_case import ExtTestCase
from onnx_light import schema_comparison as sc


class TestSchemaComparison(ExtTestCase):
    """Validates the schema comparison helpers."""

    def test_compute_returns_nonempty_rows(self):
        comparison = sc.compute_schema_comparison()
        self.assertGreater(len(comparison.rows), 0)
        # Either onnx is available -> there are many onnx schemas, or it is
        # not -> total_onnx is 0 but onnx_light still contributes rows.
        if comparison.onnx_available:
            self.assertGreater(comparison.total_onnx, 0)
        self.assertGreater(comparison.total_onnx_light, 0)

    def test_known_operators_present(self):
        comparison = sc.compute_schema_comparison()
        by_key = {(r.domain, r.name): r for r in comparison.rows}
        # Abs, Add and And are exposed by LightOpSchema and have onnx_optim
        # shape inference.
        for name in ("Abs", "Add", "And"):
            row = by_key.get(("ai.onnx", name))
            self.assertIsNotNone(row, f"Missing row for {name}")
            self.assertTrue(row.in_onnx_light, f"{name} should be in onnx_light")
            self.assertTrue(
                row.onnx_light_shape_inference, f"{name} should have onnx_light shape inference"
            )

    def test_render_rst_summary_is_directive(self):
        comparison = sc.compute_schema_comparison()
        text = sc.render_rst_summary(comparison)
        self.assertIn(".. list-table::", text)
        self.assertIn("Operators with a schema", text)
        self.assertIn("Operators with shape inference", text)
        self.assertIn("Node backend tests", text)

    def test_render_rst_table_has_all_columns(self):
        comparison = sc.compute_schema_comparison()
        text = sc.render_rst_table(comparison)
        self.assertIn(".. list-table::", text)
        self.assertIn(".. rubric:: ai.onnx", text)
        # By default no :class: option is emitted.
        self.assertNotIn(":class:", text)
        for header in (
            "Operator",
            "``onnx``",
            "``onnx_light``",
            "shape inference",
            "backend tests",
        ):
            self.assertIn(header, text)
        self.assertNotIn("    * - Domain", text)
        # The table must contain at least one row per operator that appears
        # in either onnx or onnx_light.
        n_total_rows = text.count("    * - ")
        n_header_rows = text.count("    * - Operator")
        n_data_rows = n_total_rows - n_header_rows
        n_ops_in_either = sum(1 for r in comparison.rows if r.in_onnx or r.in_onnx_light)
        n_domains = len({r.domain for r in comparison.rows if r.in_onnx or r.in_onnx_light})
        self.assertEqual(n_data_rows, n_ops_in_either)
        self.assertEqual(n_header_rows, n_domains)

    def test_render_rst_table_with_css_class(self):
        """The ``css_class`` argument opts the table into ``sphinx-datatables``."""
        comparison = sc.compute_schema_comparison()
        text = sc.render_rst_table(comparison, css_class="sphinx-datatable")
        self.assertIn(":class: sphinx-datatable", text)
        # Same row count as without the option.
        n_total_rows = text.count("    * - ")
        n_header_rows = text.count("    * - Operator")
        n_data_rows = n_total_rows - n_header_rows
        n_ops_in_either = sum(1 for r in comparison.rows if r.in_onnx or r.in_onnx_light)
        n_domains = len({r.domain for r in comparison.rows if r.in_onnx or r.in_onnx_light})
        self.assertEqual(n_data_rows, n_ops_in_either)
        self.assertEqual(n_header_rows, n_domains)

    def test_op_name_forms_handles_camelcase_and_acronyms(self):
        from onnx_light.schema_comparison import _op_name_forms

        # Plain lowercase op: a single form.
        self.assertEqual(_op_name_forms("Abs"), ("abs",))
        # CamelCase op: lowercased + snake_case forms.
        self.assertEqual(_op_name_forms("ReduceL1"), ("reducel1", "reduce_l1"))
        # Acronym + CamelCase op (matches the upstream squashed test-data
        # naming convention as well as snake_case).
        self.assertEqual(_op_name_forms("QLinearConv"), ("qlinearconv", "q_linear_conv"))

    def test_attribute_test_name_uses_longest_op_match(self):
        from onnx_light.schema_comparison import _attribute_test_name, _build_op_form_index

        keys = {
            ("ai.onnx", "Abs"),
            ("ai.onnx", "ReduceL1"),
            ("ai.onnx", "ReduceSum"),
            ("ai.onnx", "Softsign"),
            ("ai.onnx", "AveragePool"),
            ("ai.onnx.ml", "ArrayFeatureExtractor"),
        }
        idx = _build_op_form_index(keys)
        # Both ``test_`` and ``test_cc_`` prefixes resolve to Abs.
        self.assertEqual(_attribute_test_name("test_abs", idx), ("ai.onnx", "Abs"))
        self.assertEqual(_attribute_test_name("test_cc_abs", idx), ("ai.onnx", "Abs"))
        # ReduceL1 wins over a hypothetical shorter ``Reduce`` form
        # for ``test_reduce_l1_*_expanded``.
        self.assertEqual(
            _attribute_test_name("test_reduce_l1_default_axes_keepdims_example_expanded", idx),
            ("ai.onnx", "ReduceL1"),
        )
        # Softsign expanded variants attribute to Softsign (not to whatever
        # Abs-based decomposition the model uses internally).
        self.assertEqual(
            _attribute_test_name("test_softsign_expanded_ver18", idx), ("ai.onnx", "Softsign")
        )
        # ai.onnx.ml ops accept the upstream domain-prefixed test name.
        self.assertEqual(
            _attribute_test_name("test_ai_onnx_ml_array_feature_extractor", idx),
            ("ai.onnx.ml", "ArrayFeatureExtractor"),
        )
        # ``test_cc_`` cases follow the same rule for onnx_light's own cases.
        self.assertEqual(
            _attribute_test_name("test_cc_averagepool_2d_ceil", idx), ("ai.onnx", "AveragePool")
        )
        # No known op matches -> None (caller falls back to first-node op_type).
        self.assertIsNone(_attribute_test_name("test_unknown_op", idx))

    def test_onnx_optim_shape_inference_list_matches_source(self):
        """Hardcoded list of onnx_optim shape inference ops must match the
        dispatch table declared in ``dispatch_table.cc`` (when reachable)."""
        # Locate the C++ dispatch source file inside the source tree.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cc_path = os.path.join(
            repo_root, "onnx_light", "onnx_optim", "shapes", "dispatch_table.cc"
        )
        if not os.path.exists(cc_path):
            self.skipTest("dispatch_table.cc not available in this install")
        with open(cc_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        # Pull (domain, op_type) keys from the dispatch table block
        # (entries of the form ``{"<domain>:<OpName>",`` immediately
        # followed by a lambda).
        ops_in_source = set(
            re.findall(r'\{"([A-Za-z][A-Za-z0-9_.]*):([A-Za-z][A-Za-z0-9_]*)",\s*\[\]', source)
        )
        ops_in_module = set(sc.ONNX_OPTIM_SHAPE_INFERENCE_OPS)
        self.assertEqual(
            ops_in_source,
            ops_in_module,
            "ONNX_OPTIM_SHAPE_INFERENCE_OPS in onnx_light.schema_comparison must be kept "
            "in sync with the dispatch table in dispatch_table.cc",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
