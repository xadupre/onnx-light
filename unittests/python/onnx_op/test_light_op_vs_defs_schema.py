# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests that every LightOpSchema and its matching onnx_light defs OpSchema share the same attributes."""  # noqa: E501

from __future__ import annotations

import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx.defs as defs
import onnx_light.onnx_op as op


def _normalize_domain(domain: str) -> str:
    """Normalizes the ONNX default domain.

    Both ``""`` (empty string used by defs OpSchema) and ``"ai.onnx"``
    (used by LightOpSchema) represent the same default ONNX domain.
    Returns ``"ai.onnx"`` for both so that dict keys compare equal.
    """
    return "ai.onnx" if domain == "" else domain


class TestLightOpVsDefsSchemaAttributes(ExtTestCase):
    """Verifies that LightOpSchema and registered onnx_light OpSchema share the same attributes.

    Both ``onnx_light.onnx_op.get_all_schemas_with_history()`` and
    ``onnx_light.onnx.defs.get_all_schemas_with_history()`` expose the same
    operator definitions through different C++ types.  This test asserts that
    the attribute metadata (names, required-flag, and attribute type) is
    consistent between the two representations, without requiring the upstream
    ``onnx`` package.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        defs.register_onnx_operator_set_schema()

        light_hist = op.get_all_schemas_with_history()
        cls.light_dict: dict[tuple[str, str, int], op.LightOpSchema] = {
            (_normalize_domain(s.domain), s.name, s.since_version): s for s in light_hist
        }

        defs_hist = defs.get_all_schemas_with_history()
        cls.defs_dict: dict[tuple[str, str, int], defs.OpSchema] = {
            (_normalize_domain(s.domain), s.name, s.since_version): s for s in defs_hist
        }

    def test_same_operator_keys(self) -> None:
        # The defs registry includes all historical opset versions, while
        # LightOpSchema only covers versions that onnx-light implements.
        # Custom domains (e.g. "ai.rt") are light-only and have no defs entry.
        # We verify two things:
        #   1. Every standard-domain key in light_dict exists in defs_dict.
        #   2. Custom-domain keys in light_dict are acceptable (no assertion).
        standard_domains = {"ai.onnx", "ai.onnx.ml", "ai.onnx.preview.training"}
        light_standard = {k for k in self.light_dict if k[0] in standard_domains}
        defs_standard = {k for k in self.defs_dict if k[0] in standard_domains}
        extra_in_light = light_standard - defs_standard
        self.assertEqual(
            extra_in_light,
            set(),
            msg=f"Light has standard-domain operators not present in defs: {extra_in_light}",
        )

    def test_attributes_same_names(self) -> None:
        # Verify that every attribute declared in light_schema also exists in defs_schema.
        # Defs may declare additional attributes that light has not yet annotated; that is
        # acceptable. What must not happen is light declaring an attribute name that does not
        # exist in defs (which would indicate a typo or API drift).
        for key, light_schema in self.light_dict.items():
            defs_schema = self.defs_dict.get(key)
            if defs_schema is None:
                continue
            with self.subTest(key=key):
                light_attr_names = {a.name for a in light_schema.attributes}
                defs_attr_names = set(defs_schema.attributes.keys())
                extra_in_light = light_attr_names - defs_attr_names
                self.assertEqual(
                    extra_in_light,
                    set(),
                    msg=f"Light declares attributes absent from defs for {key}: {extra_in_light}",
                )

    def test_attributes_required_flag(self) -> None:
        for key, light_schema in self.light_dict.items():
            defs_schema = self.defs_dict.get(key)
            if defs_schema is None:
                continue
            with self.subTest(key=key):
                light_by_name = {a.name: a for a in light_schema.attributes}
                for name, defs_attr in defs_schema.attributes.items():
                    light_attr = light_by_name.get(name)
                    if light_attr is None:
                        continue
                    with self.subTest(attribute=name):
                        self.assertEqual(
                            defs_attr.required,
                            light_attr.required,
                            msg=f"required mismatch for attribute {name!r} in {key}",
                        )

    def test_attributes_type(self) -> None:
        for key, light_schema in self.light_dict.items():
            defs_schema = self.defs_dict.get(key)
            if defs_schema is None:
                continue
            with self.subTest(key=key):
                light_by_name = {a.name: a for a in light_schema.attributes}
                for name, defs_attr in defs_schema.attributes.items():
                    light_attr = light_by_name.get(name)
                    if light_attr is None:
                        continue
                    with self.subTest(attribute=name):
                        # AttributeProto::AttributeType (defs) and AttributeType (op)
                        # use the same enum names ("INT", "FLOAT", "STRING", etc.).
                        self.assertEqual(
                            defs_attr.type.name,
                            light_attr.type.name,
                            msg=f"type mismatch for attribute {name!r} in {key}: "
                            f"defs={defs_attr.type.name!r}, light={light_attr.type.name!r}",
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
