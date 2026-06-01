"""Compare backend test names between ONNX and ``onnx_light``.

For every node-level test discovered by ``onnx.backend.test`` (ignoring CUDA
variants), strips the ``test_`` prefix and the ``_cpu`` suffix and checks that
the remaining string is found as a substring in the name of at least one
``onnx_light`` backend test case.

The set of ONNX test names that are not yet covered by ``onnx_light`` is
maintained as a snapshot in ``_backend_test_known_missing.txt`` so the test
fails when new ONNX tests appear (or when ``onnx_light`` gains coverage that
makes a previously-missing entry stale).
"""

import os
import unittest

import onnx.backend.base
from onnx.backend.test import BackendTest

from onnx_light.backend.test.case.base import collect_test_case
from onnx_light.ext_test_case import ExtTestCase


_KNOWN_MISSING_FILE = os.path.join(
    os.path.dirname(__file__), "_backend_test_known_missing.txt"
)


class _DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def supports_device(cls, device):  # pragma: no cover - trivial
        return True


def _onnx_node_cpu_test_names() -> list[str]:
    """Returns all ONNX node-level backend test method names ending in ``_cpu``."""
    bt = BackendTest(_DummyBackend, "dummy")
    cls = bt.test_cases["OnnxBackendNodeModelTest"]
    return sorted(
        m
        for m in dir(cls)
        if m.startswith("test_") and m.endswith("_cpu") and callable(getattr(cls, m))
    )


def _load_known_missing() -> set[str]:
    with open(_KNOWN_MISSING_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


class TestBackendTestNamesOnnxVsOnnxLight(ExtTestCase):
    """Ensures ONNX node backend tests have a counterpart in ``onnx_light``."""

    def test_onnx_backend_test_names_found_in_onnx_light(self):
        light_names = set(collect_test_case().keys())
        self.assertGreater(len(light_names), 0)

        missing: list[str] = []
        for method_name in _onnx_node_cpu_test_names():
            # The ``_cuda`` variants are excluded by construction.
            assert not method_name.endswith("_cuda")
            stripped = method_name[len("test_") : -len("_cpu")]
            if not any(stripped in light_name for light_name in light_names):
                missing.append(stripped)

        known_missing = _load_known_missing()
        missing_set = set(missing)

        # Names that are newly missing (i.e. ONNX added a test we do not cover).
        new_missing = sorted(missing_set - known_missing)
        # Names recorded as missing that ``onnx_light`` now covers.
        stale_entries = sorted(known_missing - missing_set)

        msg_parts = []
        if new_missing:
            msg_parts.append(
                "ONNX backend tests without a matching onnx-light test "
                f"({len(new_missing)}): {new_missing[:20]}"
                + (" ..." if len(new_missing) > 20 else "")
                + f"\nAdd the matching cases to onnx_light or append these names to "
                f"{os.path.basename(_KNOWN_MISSING_FILE)}."
            )
        if stale_entries:
            msg_parts.append(
                "Entries in known-missing snapshot that are now covered by "
                f"onnx-light ({len(stale_entries)}): {stale_entries[:20]}"
                + (" ..." if len(stale_entries) > 20 else "")
                + f"\nRemove these names from {os.path.basename(_KNOWN_MISSING_FILE)}."
            )
        if msg_parts:
            self.fail("\n\n".join(msg_parts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
