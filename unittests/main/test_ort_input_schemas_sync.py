"""Checks the generated ORT input metadata against the registered schema history."""

from pathlib import Path
import subprocess
import sys
import unittest

from onnx_light.ext_test_case import ExtTestCase


class TestOrtInputSchemasSync(ExtTestCase):
    def test_snapshot_matches_registered_schemas(self):
        root = Path(__file__).resolve().parents[2]
        # A fresh registry avoids schemas registered or removed by other tests.
        result = subprocess.run(
            [
                sys.executable,
                str(root / ".github" / "scripts" / "generate_ort_input_schemas.py"),
                "--check",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
