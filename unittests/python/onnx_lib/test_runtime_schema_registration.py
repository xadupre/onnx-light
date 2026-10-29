# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Compares source schema definitions with the actual runtime registry (onnx/onnx#8635)."""

import unittest

from onnx_light.onnx import defs
from unittests.main.test_schema_registration import _MACRO_INFO, _all_defined_schemas


class TestRuntimeSchemaRegistration(unittest.TestCase):
    def test_runtime_registry_matches_defined_schemas(self):
        defs.register_onnx_operator_set_schema()
        runtime: dict[str, set[tuple[str, int]]] = {}
        for schema in defs.get_all_schemas_with_history():
            runtime.setdefault(schema.domain, set()).add((schema.name, schema.since_version))

        defined = _all_defined_schemas()
        for macro, (_, registry_domain, _) in _MACRO_INFO.items():
            with self.subTest(macro=macro, domain=registry_domain):
                registered = runtime.get(registry_domain, set())
                missing = defined[macro] - registered
                self.assertFalse(
                    missing,
                    f"{len(missing)} schema(s) defined via {macro} are missing from the "
                    f"runtime registry for domain {registry_domain!r}: {sorted(missing)}. "
                    "Check fn(GetOpSchema<...>()) and RegisterOpSetSchema<...>() calls.",
                )
                unexpected = registered - defined[macro]
                self.assertFalse(
                    unexpected,
                    f"{len(unexpected)} schema(s) in the runtime registry for domain "
                    f"{registry_domain!r} were not parsed as {macro} definitions: "
                    f"{sorted(unexpected)}. Check the registry domain and definition parser.",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
