"""Verifies security defaults when the upstream onnx package is installed.

Investigates GHSA-cjhm-j56f-fj5v: ExternalDataInfo attribute injection and
resource exhaustion via crafted external_data entries in TensorProto.

Findings:
- onnx-light is NOT affected: ExternalDataInfo uses an explicit key whitelist
  rather than setattr(), so unknown keys are silently ignored.
- _load_external_data_for_tensor validates offset/length against actual file size.
- When upstream onnx IS installed, it must be >= 1.21.0 which contains the fix.

All tests in this module are skipped when onnx is not installed.
"""

from __future__ import annotations

import importlib.util
import unittest


def _has_onnx() -> bool:
    """Returns True if the upstream onnx package is importable."""
    return importlib.util.find_spec("onnx") is not None


@unittest.skipUnless(_has_onnx(), "upstream onnx is not installed")
class TestUpstreamOnnxSecurityVersion(unittest.TestCase):
    """Checks that the installed upstream onnx version includes GHSA-cjhm-j56f-fj5v fix."""

    def test_onnx_version_at_least_1_21(self) -> None:
        """Verifies that upstream onnx is >= 1.21.0 which contains the GHSA-cjhm-j56f-fj5v fix.

        GHSA-cjhm-j56f-fj5v: ExternalDataInfo attribute injection via unvalidated
        setattr() calls.  Fixed in onnx 1.21.0 by switching to an allowlist approach.
        """
        import onnx

        version_str = onnx.__version__
        # Strip pre-release and local suffixes (e.g. "1.21.0rc1", "1.21.0.dev20250101")
        # to get a numeric-only tuple for comparison.
        numeric_part = version_str.split("+")[0]  # drop local label
        segments = []
        for seg in numeric_part.split("."):
            # Keep only the leading digits from each segment (e.g. "0rc1" → 0).
            digits = ""
            for ch in seg:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                segments.append(int(digits))
            if len(segments) == 2:
                break
        major, minor = [*segments, 0, 0][:2]
        self.assertGreaterEqual(
            (major, minor),
            (1, 21),
            f"Installed onnx version {version_str!r} is older than 1.21.0 which "
            "contains the security fix for GHSA-cjhm-j56f-fj5v "
            "(ExternalDataInfo attribute injection). "
            "Upgrade onnx to >= 1.21.0.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
