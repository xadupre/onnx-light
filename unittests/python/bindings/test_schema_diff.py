"""Tests for onnx_light.compatibility.compare_schemas."""

import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx.defs as defs
from onnx_light.tools.compatibility import (
    ConstraintDiff,
    DeprecationDiff,
    DocDiff,
    SchemaDiff,
    compare_schemas,
)


class TestCompareSchemasBuiltin(ExtTestCase):
    """Tests that exercise compare_schemas on built-in ONNX operator schemas."""

    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    # ------------------------------------------------------------------
    # Return type and basic structure
    # ------------------------------------------------------------------

    def test_returns_schema_diff_instance(self):
        old = defs.get_schema("Relu", 6)
        new = defs.get_schema("Relu", 14)
        diff = compare_schemas(old, new)
        self.assertIsInstance(diff, SchemaDiff)

    def test_schema_diff_metadata(self):
        old = defs.get_schema("Relu", 6)
        new = defs.get_schema("Relu", 14)
        diff = compare_schemas(old, new)
        self.assertEqual(diff.op_name, "Relu")
        self.assertEqual(diff.domain, defs.ONNX_DOMAIN)
        self.assertEqual(diff.old_version, old.since_version)
        self.assertEqual(diff.new_version, new.since_version)

    # ------------------------------------------------------------------
    # Relu v6 → v14: type constraint expanded (non-breaking)
    # ------------------------------------------------------------------

    def test_relu_v6_to_v14_type_constraint_expanded(self):
        old = defs.get_schema("Relu", 6)
        new = defs.get_schema("Relu", 14)
        diff = compare_schemas(old, new)
        # Relu v14 adds more types (int32, int8, …) → non-breaking
        self.assertFalse(diff.is_breaking)
        self.assertEqual(len(diff.constraints), 1)
        cdiff = diff.constraints[0]
        self.assertIsInstance(cdiff, ConstraintDiff)
        self.assertEqual(cdiff.name, "T")
        self.assertEqual(cdiff.kind, "changed")
        self.assertFalse(cdiff.is_breaking)
        self.assertTrue(len(cdiff.added_types) > 0)
        self.assertEqual(len(cdiff.removed_types), 0)

    # ------------------------------------------------------------------
    # Identical schema → no diffs
    # ------------------------------------------------------------------

    def test_identical_schemas_no_diff(self):
        s = defs.get_schema("Add")
        diff = compare_schemas(s, s)
        self.assertFalse(diff.is_breaking)
        self.assertEqual(diff.inputs, [])
        self.assertEqual(diff.outputs, [])
        self.assertEqual(diff.attributes, [])
        self.assertEqual(diff.constraints, [])

    # ------------------------------------------------------------------
    # str() output
    # ------------------------------------------------------------------

    def test_str_representation(self):
        old = defs.get_schema("Relu", 6)
        new = defs.get_schema("Relu", 14)
        diff = compare_schemas(old, new)
        s = str(diff)
        self.assertIn("Relu", s)
        self.assertIn("breaking", s)


class TestCompareSchemasCustom(ExtTestCase):
    """Tests that build custom OpSchema objects to exercise all diff branches."""

    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_fp(self, name, type_str="tensor(float)", option=None):
        if option is None:
            option = defs.OpSchema.FormalParameterOption.Single
        return defs.OpSchema.FormalParameter(name, type_str, param_option=option)

    def _make_attr_required(self, name, attr_type=None):
        from onnx_light.onnx import AttributeProto  # type: ignore

        return defs.OpSchema.Attribute(name, AttributeProto.INT, required=True)

    def _make_schema(
        self, name, version, inputs=None, outputs=None, attrs=None, constraints=None
    ):
        inputs = inputs or []
        outputs = outputs or []
        attrs = attrs or []
        constraints = constraints or []
        return defs.OpSchema(
            name,
            defs.ONNX_DOMAIN,
            version,
            inputs=inputs,
            outputs=outputs,
            attributes=attrs,
            type_constraints=constraints,
        )

    # ------------------------------------------------------------------
    # Input changes
    # ------------------------------------------------------------------

    def test_input_removed_is_breaking(self):
        fp_x = self._make_fp("X")
        fp_y = self._make_fp("Y")
        old = self._make_schema("MyOp", 1, inputs=[fp_x, fp_y], outputs=[self._make_fp("Z")])
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[self._make_fp("Z")])
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        removed = [d for d in diff.inputs if d.kind == "removed"]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].name, "Y")
        self.assertTrue(removed[0].is_breaking)

    def test_optional_input_added_is_not_breaking(self):
        fp_x = self._make_fp("X")
        fp_y = self._make_fp("Y", option=defs.OpSchema.FormalParameterOption.Optional)
        old = self._make_schema("MyOp", 1, inputs=[fp_x], outputs=[self._make_fp("Z")])
        new = self._make_schema("MyOp", 2, inputs=[fp_x, fp_y], outputs=[self._make_fp("Z")])
        diff = compare_schemas(old, new)
        self.assertFalse(diff.is_breaking)
        added = [d for d in diff.inputs if d.kind == "added"]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].name, "Y")
        self.assertFalse(added[0].is_breaking)

    def test_required_input_added_is_breaking(self):
        fp_x = self._make_fp("X")
        fp_y = self._make_fp("Y")  # Single (required by default)
        old = self._make_schema("MyOp", 1, inputs=[fp_x], outputs=[self._make_fp("Z")])
        new = self._make_schema("MyOp", 2, inputs=[fp_x, fp_y], outputs=[self._make_fp("Z")])
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        added = [d for d in diff.inputs if d.kind == "added"]
        self.assertEqual(len(added), 1)
        self.assertTrue(added[0].is_breaking)

    # ------------------------------------------------------------------
    # Output changes
    # ------------------------------------------------------------------

    def test_output_removed_is_breaking(self):
        fp_x = self._make_fp("X")
        fp_z1 = self._make_fp("Z1")
        fp_z2 = self._make_fp("Z2")
        old = self._make_schema("MyOp", 1, inputs=[fp_x], outputs=[fp_z1, fp_z2])
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[fp_z1])
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        removed = [d for d in diff.outputs if d.kind == "removed"]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].name, "Z2")
        self.assertTrue(removed[0].is_breaking)

    def test_optional_output_added_is_not_breaking(self):
        fp_x = self._make_fp("X")
        fp_z1 = self._make_fp("Z1")
        fp_z2 = self._make_fp("Z2", option=defs.OpSchema.FormalParameterOption.Optional)
        old = self._make_schema("MyOp", 1, inputs=[fp_x], outputs=[fp_z1])
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[fp_z1, fp_z2])
        diff = compare_schemas(old, new)
        self.assertFalse(diff.is_breaking)
        added = [d for d in diff.outputs if d.kind == "added"]
        self.assertEqual(len(added), 1)
        self.assertFalse(added[0].is_breaking)

    # ------------------------------------------------------------------
    # Attribute changes
    # ------------------------------------------------------------------

    def test_attribute_removed_is_breaking(self):
        from onnx_light.onnx import AttributeProto  # type: ignore

        attr = defs.OpSchema.Attribute("mode", AttributeProto.INT, required=False)
        fp_x = self._make_fp("X")
        fp_z = self._make_fp("Z")
        old = self._make_schema("MyOp", 1, inputs=[fp_x], outputs=[fp_z], attrs=[attr])
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[fp_z])
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        removed = [d for d in diff.attributes if d.kind == "removed"]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].name, "mode")
        self.assertTrue(removed[0].is_breaking)

    def test_required_attribute_added_is_breaking(self):
        from onnx_light.onnx import AttributeProto  # type: ignore

        attr = defs.OpSchema.Attribute("kernel_shape", AttributeProto.INTS, required=True)
        fp_x = self._make_fp("X")
        fp_z = self._make_fp("Z")
        old = self._make_schema("MyOp", 1, inputs=[fp_x], outputs=[fp_z])
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[fp_z], attrs=[attr])
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        added = [d for d in diff.attributes if d.kind == "added"]
        self.assertEqual(len(added), 1)
        self.assertTrue(added[0].is_breaking)

    def test_optional_attribute_added_is_not_breaking(self):
        from onnx_light.onnx import AttributeProto as AP  # type: ignore

        fp_x = self._make_fp("X")
        fp_z = self._make_fp("Z")
        old = self._make_schema("MyOp", 1, inputs=[fp_x], outputs=[fp_z])
        # Use the constructor that takes a default AttributeProto value.
        default_val = AP()
        default_val.type = AP.FLOAT
        default_val.f = 1e-5
        attr_opt = defs.OpSchema.Attribute("epsilon", default_val, description="small constant")
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[fp_z], attrs=[attr_opt])
        diff = compare_schemas(old, new)
        self.assertFalse(diff.is_breaking)
        added = [d for d in diff.attributes if d.kind == "added"]
        self.assertEqual(len(added), 1)
        self.assertFalse(added[0].is_breaking)

    def test_attribute_default_value_changed_is_breaking(self):
        from onnx_light.onnx import AttributeProto as AP  # type: ignore

        fp_x = self._make_fp("X")
        fp_z = self._make_fp("Z")

        dv_old = AP()
        dv_old.type = AP.INT
        dv_old.i = 1
        attr_old = defs.OpSchema.Attribute("group", dv_old, description="group")

        dv_new = AP()
        dv_new.type = AP.INT
        dv_new.i = 2
        attr_new = defs.OpSchema.Attribute("group", dv_new, description="group")

        old = self._make_schema("MyOp", 1, inputs=[fp_x], outputs=[fp_z], attrs=[attr_old])
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[fp_z], attrs=[attr_new])
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        changed = [d for d in diff.attributes if d.kind == "changed"]
        self.assertEqual(len(changed), 1)
        self.assertTrue(changed[0].is_breaking)
        self.assertTrue(any("default value changed" in det for det in changed[0].details))

    # ------------------------------------------------------------------
    # Type constraint changes
    # ------------------------------------------------------------------

    def test_constraint_type_removed_is_breaking(self):
        fp_x = defs.OpSchema.FormalParameter(
            "X", "T", param_option=defs.OpSchema.FormalParameterOption.Single
        )
        fp_z = defs.OpSchema.FormalParameter(
            "Z", "T", param_option=defs.OpSchema.FormalParameterOption.Single
        )
        old = self._make_schema(
            "MyOp",
            1,
            inputs=[fp_x],
            outputs=[fp_z],
            constraints=[("T", ["tensor(float)", "tensor(double)"], "float types")],
        )
        new = self._make_schema(
            "MyOp",
            2,
            inputs=[fp_x],
            outputs=[fp_z],
            constraints=[("T", ["tensor(float)"], "float types")],
        )
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        changed = [d for d in diff.constraints if d.kind == "changed"]
        self.assertEqual(len(changed), 1)
        self.assertTrue(changed[0].is_breaking)
        self.assertIn("tensor(double)", changed[0].removed_types)

    def test_constraint_type_added_is_not_breaking(self):
        fp_x = defs.OpSchema.FormalParameter(
            "X", "T", param_option=defs.OpSchema.FormalParameterOption.Single
        )
        fp_z = defs.OpSchema.FormalParameter(
            "Z", "T", param_option=defs.OpSchema.FormalParameterOption.Single
        )
        old = self._make_schema(
            "MyOp",
            1,
            inputs=[fp_x],
            outputs=[fp_z],
            constraints=[("T", ["tensor(float)"], "float types")],
        )
        new = self._make_schema(
            "MyOp",
            2,
            inputs=[fp_x],
            outputs=[fp_z],
            constraints=[("T", ["tensor(float)", "tensor(double)"], "float types")],
        )
        diff = compare_schemas(old, new)
        self.assertFalse(diff.is_breaking)
        changed = [d for d in diff.constraints if d.kind == "changed"]
        self.assertEqual(len(changed), 1)
        self.assertFalse(changed[0].is_breaking)
        self.assertIn("tensor(double)", changed[0].added_types)

    def test_constraint_removed_entirely_is_breaking(self):
        fp_x = defs.OpSchema.FormalParameter(
            "X", "T", param_option=defs.OpSchema.FormalParameterOption.Single
        )
        fp_z = defs.OpSchema.FormalParameter(
            "Z", "T", param_option=defs.OpSchema.FormalParameterOption.Single
        )
        old = self._make_schema(
            "MyOp",
            1,
            inputs=[fp_x],
            outputs=[fp_z],
            constraints=[("T", ["tensor(float)"], "float types")],
        )
        new_fp_x = defs.OpSchema.FormalParameter(
            "X", "tensor(float)", param_option=defs.OpSchema.FormalParameterOption.Single
        )
        new_fp_z = defs.OpSchema.FormalParameter(
            "Z", "tensor(float)", param_option=defs.OpSchema.FormalParameterOption.Single
        )
        new = self._make_schema("MyOp", 2, inputs=[new_fp_x], outputs=[new_fp_z])
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        removed = [d for d in diff.constraints if d.kind == "removed"]
        self.assertEqual(len(removed), 1)
        self.assertTrue(removed[0].is_breaking)

    # ------------------------------------------------------------------
    # breaking_reasons list
    # ------------------------------------------------------------------

    def test_breaking_reasons_populated(self):
        fp_x = self._make_fp("X")
        fp_y = self._make_fp("Y")
        fp_z = self._make_fp("Z")
        old = self._make_schema("MyOp", 1, inputs=[fp_x, fp_y], outputs=[fp_z])
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[fp_z])
        diff = compare_schemas(old, new)
        self.assertTrue(diff.is_breaking)
        self.assertTrue(len(diff.breaking_reasons) > 0)
        self.assertTrue(any("Y" in r for r in diff.breaking_reasons))

    # ------------------------------------------------------------------
    # str() output contains relevant info
    # ------------------------------------------------------------------

    def test_str_shows_breaking(self):
        fp_x = self._make_fp("X")
        fp_y = self._make_fp("Y")
        fp_z = self._make_fp("Z")
        old = self._make_schema("MyOp", 1, inputs=[fp_x, fp_y], outputs=[fp_z])
        new = self._make_schema("MyOp", 2, inputs=[fp_x], outputs=[fp_z])
        diff = compare_schemas(old, new)
        s = str(diff)
        self.assertIn("MyOp", s)
        self.assertIn("True", s)
        self.assertIn("removed", s)


class TestDocDiff(ExtTestCase):
    """Tests for :class:`DocDiff` (line-level documentation diff)."""

    def test_doc_diff_unchanged(self):
        d = DocDiff.compare("hello\nworld", "hello\nworld")
        self.assertFalse(d.changed)
        self.assertEqual(d.similarity, 1.0)
        self.assertEqual(d.unified_diff, [])
        self.assertEqual(d.added_lines, 0)
        self.assertEqual(d.removed_lines, 0)
        self.assertEqual(str(d), "doc unchanged")

    def test_doc_diff_none_inputs_unchanged(self):
        d = DocDiff.compare(None, None)
        self.assertFalse(d.changed)
        self.assertEqual(d.similarity, 1.0)

    def test_doc_diff_line_level_not_char_level(self):
        # A single-character difference inside one line must count as exactly
        # one removed line and one added line, not as many character edits.
        old_doc = "line a\nline b\nline c"
        new_doc = "line a\nline B\nline c"
        d = DocDiff.compare(old_doc, new_doc)
        self.assertTrue(d.changed)
        self.assertEqual(d.added_lines, 1)
        self.assertEqual(d.removed_lines, 1)
        # Three lines, one of which changed -> SequenceMatcher ratio = 2*2/6
        self.assertAlmostEqual(d.similarity, 2 / 3, atol=1e-3)
        # The unified diff must contain proper +/- markers on lines.
        self.assertTrue(any(line.startswith("-line b") for line in d.unified_diff))
        self.assertTrue(any(line.startswith("+line B") for line in d.unified_diff))

    def test_doc_diff_unified_diff_header(self):
        d = DocDiff.compare("a", "b", old_label="A", new_label="B")
        self.assertTrue(d.changed)
        self.assertEqual(d.unified_diff[0], "--- A")
        self.assertEqual(d.unified_diff[1], "+++ B")

    def test_doc_diff_added_only_lines(self):
        d = DocDiff.compare("a\nb", "a\nb\nc\nd")
        self.assertTrue(d.changed)
        self.assertEqual(d.added_lines, 2)
        self.assertEqual(d.removed_lines, 0)

    def test_doc_diff_removed_only_lines(self):
        d = DocDiff.compare("a\nb\nc\nd", "a\nb")
        self.assertTrue(d.changed)
        self.assertEqual(d.added_lines, 0)
        self.assertEqual(d.removed_lines, 2)

    def test_doc_diff_str_contains_unified_block(self):
        d = DocDiff.compare("line 1\nline 2\nline 3", "line 1\nLINE 2\nline 3")
        s = str(d)
        self.assertIn("line similarity", s)
        self.assertIn("-line 2", s)
        self.assertIn("+LINE 2", s)


class TestSchemaDiffDocIntegration(ExtTestCase):
    """Tests that :func:`compare_schemas` exposes a line-level doc diff."""

    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    def _make_fp(self, name, type_str="tensor(float)"):
        return defs.OpSchema.FormalParameter(
            name, type_str, param_option=defs.OpSchema.FormalParameterOption.Single
        )

    def test_doc_unchanged_for_identical_schemas(self):
        s = defs.get_schema("Add")
        diff = compare_schemas(s, s)
        self.assertIsInstance(diff.doc, DocDiff)
        self.assertFalse(diff.doc.changed)

    def test_doc_changed_in_custom_schemas(self):
        fp_x = self._make_fp("X")
        fp_z = self._make_fp("Z")
        old = defs.OpSchema(
            "MyOp", defs.ONNX_DOMAIN, 1, "line 1\nline 2\nline 3", inputs=[fp_x], outputs=[fp_z]
        )
        new = defs.OpSchema(
            "MyOp",
            defs.ONNX_DOMAIN,
            2,
            "line 1\nLINE 2\nline 3\nline 4",
            inputs=[fp_x],
            outputs=[fp_z],
        )
        diff = compare_schemas(old, new)
        # A documentation-only change is not breaking on its own.
        self.assertFalse(diff.is_breaking)
        self.assertTrue(diff.doc.changed)
        self.assertEqual(diff.doc.removed_lines, 1)
        self.assertEqual(diff.doc.added_lines, 2)
        self.assertLess(diff.doc.similarity, 1.0)
        # The unified diff renders line-by-line with +/- prefixes.
        self.assertTrue(any(line.startswith("-line 2") for line in diff.doc.unified_diff))
        self.assertTrue(any(line.startswith("+LINE 2") for line in diff.doc.unified_diff))
        # Plain str() and RST renderings include the diff.
        s = str(diff)
        self.assertIn("Documentation", s)
        self.assertIn("-line 2", s)
        rst = diff.to_rst()
        self.assertIn("**Documentation:**", rst)
        self.assertIn(".. code-block:: diff", rst)
        self.assertIn("-line 2", rst)
        self.assertIn("+LINE 2", rst)


class TestCompareSchemasLightOpSchema(ExtTestCase):
    """Tests that :func:`compare_schemas` also accepts ``LightOpSchema`` objects."""

    @classmethod
    def setUpClass(cls):
        from onnx_light.onnx_py._onnxpyprotoop import onnx_op  # type: ignore

        cls.onnx_op = onnx_op
        cls.schemas = onnx_op.GetAllOnnxOpSchemasWithHistory(init_doc=True)
        from collections import defaultdict

        by_name: dict = defaultdict(list)
        for s in cls.schemas:
            by_name[(s.domain, s.name)].append(s)
        for v in by_name.values():
            v.sort(key=lambda s: s.since_version)
        cls.by_name = by_name

    def _get_light(self, name, version, domain="ai.onnx"):
        for s in self.schemas:
            if s.name == name and s.since_version == version and s.domain == domain:
                return s
        raise self.failureException(f"No LightOpSchema {name} v{version} in {domain}")

    def test_identical_light_schemas_no_diff(self):
        s = self._get_light("Add", 14)
        diff = compare_schemas(s, s)
        self.assertIsInstance(diff, SchemaDiff)
        self.assertFalse(diff.is_breaking)
        self.assertEqual(diff.op_name, "Add")
        self.assertEqual(diff.old_version, 14)
        self.assertEqual(diff.new_version, 14)
        self.assertEqual(diff.inputs, [])
        self.assertEqual(diff.outputs, [])
        self.assertEqual(diff.attributes, [])
        self.assertEqual(diff.constraints, [])
        self.assertFalse(diff.doc.changed)

    def test_light_add_v1_to_v14_widens_type_constraint(self):
        old = self._get_light("Add", 1)
        new = self._get_light("Add", 14)
        diff = compare_schemas(old, new)
        # Add v14 widens the supported types -> non-breaking change.
        self.assertFalse(diff.is_breaking)
        self.assertEqual(diff.op_name, "Add")
        self.assertEqual(diff.old_version, 1)
        self.assertEqual(diff.new_version, 14)
        # Attributes are not exposed by LightOpSchema -> always empty.
        self.assertEqual(diff.attributes, [])
        # A single type-constraint widening should be reported.
        self.assertEqual(len(diff.constraints), 1)
        cdiff = diff.constraints[0]
        self.assertIsInstance(cdiff, ConstraintDiff)
        self.assertEqual(cdiff.name, "T")
        self.assertEqual(cdiff.kind, "changed")
        self.assertFalse(cdiff.is_breaking)
        # The added types come back as plain ONNX type strings, not enum values.
        for t in cdiff.added_types:
            self.assertIsInstance(t, str)
        self.assertEqual(cdiff.removed_types, [])
        # Documentation changed between v1 and v14.
        self.assertTrue(diff.doc.changed)

    def test_light_compare_all_pairs_succeeds(self):
        """compare_schemas works on every (oldest, newest) LightOpSchema pair."""
        compared = 0
        for (_domain, _name), versions in self.by_name.items():
            if len(versions) < 2:
                continue
            diff = compare_schemas(versions[0], versions[-1])
            self.assertIsInstance(diff, SchemaDiff)
            compared += 1
        self.assertGreater(compared, 0)

    def test_light_compare_optional_get_element(self):
        """Covers schemas that use ``optional(...)`` type strings."""
        versions = self.by_name[("ai.onnx", "OptionalGetElement")]
        if len(versions) < 2:
            self.skipTest("Not enough OptionalGetElement versions.")
        diff = compare_schemas(versions[0], versions[-1])
        # All added/removed types should be serialised as strings.
        for cdiff in diff.constraints:
            for t in cdiff.added_types + cdiff.removed_types:
                self.assertIsInstance(t, str)


class TestDeprecationDiff(ExtTestCase):
    """Tests for :class:`DeprecationDiff` (operator deprecation status diff)."""

    class _Stub:
        """Minimal schema-like object exposing only ``deprecated``."""

        def __init__(self, deprecated):
            self.deprecated = deprecated

    def test_unchanged_not_deprecated(self):
        d = DeprecationDiff.compare(self._Stub(False), self._Stub(False))
        self.assertFalse(d.changed)
        self.assertFalse(d.is_breaking)

    def test_unchanged_deprecated(self):
        d = DeprecationDiff.compare(self._Stub(True), self._Stub(True))
        self.assertFalse(d.changed)
        self.assertFalse(d.is_breaking)

    def test_newly_deprecated_is_breaking(self):
        d = DeprecationDiff.compare(self._Stub(False), self._Stub(True))
        self.assertTrue(d.changed)
        self.assertTrue(d.is_breaking)
        self.assertIn("[BREAKING]", str(d))
        self.assertIn("False -> True", str(d))

    def test_undeprecated_is_not_breaking(self):
        d = DeprecationDiff.compare(self._Stub(True), self._Stub(False))
        self.assertTrue(d.changed)
        self.assertFalse(d.is_breaking)

    def test_missing_attribute_defaults_to_not_deprecated(self):
        d = DeprecationDiff.compare(object(), object())
        self.assertFalse(d.changed)
        self.assertFalse(d.is_breaking)


class TestCompareSchemasDeprecation(ExtTestCase):
    """Tests that :func:`compare_schemas` reports a deprecation change."""

    @classmethod
    def setUpClass(cls):
        from collections import defaultdict

        from onnx_light.onnx_py._onnxpyprotoop import onnx_op  # type: ignore

        cls.schemas = onnx_op.GetAllOnnxOpSchemasWithHistory(init_doc=True)
        by_name: dict = defaultdict(list)
        for s in cls.schemas:
            by_name[(s.domain, s.name)].append(s)
        for v in by_name.values():
            v.sort(key=lambda s: s.since_version)
        cls.by_name = by_name

    def _find_deprecated_pair(self):
        for versions in self.by_name.values():
            deprecated = [s for s in versions if getattr(s, "deprecated", False)]
            active = [s for s in versions if not getattr(s, "deprecated", False)]
            if deprecated and active:
                return active[0], deprecated[-1]
        return None

    def test_no_deprecation_change_is_not_reported(self):
        s = next(iter(self.by_name.values()))[0]
        diff = compare_schemas(s, s)
        self.assertIsInstance(diff.deprecation, DeprecationDiff)
        self.assertFalse(diff.deprecation.changed)
        self.assertNotIn("Deprecation", str(diff))

    def test_newly_deprecated_operator_is_breaking(self):
        pair = self._find_deprecated_pair()
        if pair is None:
            self.skipTest("No operator with both active and deprecated versions.")
        old, new = pair
        diff = compare_schemas(old, new)
        self.assertTrue(diff.deprecation.changed)
        self.assertTrue(diff.deprecation.is_breaking)
        self.assertTrue(diff.is_breaking)
        self.assertTrue(any("deprecated" in r for r in diff.breaking_reasons))
        # Both plain text and RST renderings surface the deprecation change.
        self.assertIn("Deprecation", str(diff))
        self.assertIn("**Deprecation:**", diff.to_rst())


if __name__ == "__main__":
    unittest.main(verbosity=2)
