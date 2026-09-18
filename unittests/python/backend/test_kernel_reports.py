# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
from onnx_light.tools import kernel_reports  # noqa: E402


class TestKernelReports(ExtTestCase):
    def prepare_profile(self, output):
        """Persists a real profile using the current schema rather than fixed ABI values."""
        cache = str(output / "kernel_tuning.cache")
        schema = runtime.kernel_tuning_parameters(kernel="Abs", element_type=1, path=cache)[
            "kernels"
        ][0]
        key = {field: schema[field] for field in kernel_reports.KEY_FIELDS}
        result = runtime.set_kernel_tuning_parameters(
            **{field: value for field, value in key.items() if field != "device"},
            values=schema["defaults"],
            path=cache,
            num_threads=1,
            load=False,
        )
        self.assertEqual(result["status"], "updated")
        kernel_reports.write_report(
            output / "metadata.json",
            {"execution": {"effective_threads": 1}, "measurement": {"repeat": 1}},
        )
        report = {
            "calibratable_keys": 1,
            "calibrated_profiles": [{**key, "values": schema["defaults"]}],
        }
        kernel_reports.write_report(output / "calibration.json", report)
        return report

    def run_phase(self, output, phase):
        """Executes the actual CLI in a fresh process with an isolated default cache."""
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "onnx_light.tools.kernel_reports",
                phase,
                "--output",
                str(output),
            ],
            env={
                **os.environ,
                "XDG_CACHE_HOME": str(output / "unused-default-cache"),
                "LOCALAPPDATA": str(output / "unused-default-cache"),
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def test_fresh_process_reload_and_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.prepare_profile(output)
            kernel_reports.write_report(output / "baseline.json", kernel_reports.baseline(1, 1))
            result = self.run_phase(output, "verify")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "calibration.json").read_text())
            self.assertEqual(report["reload_verification"]["published_profiles_resolved"], 1)
            comparison = json.loads((output / "comparison.json").read_text())
            self.assertTrue(comparison)
            self.assertTrue(all(row["same_machine_speedup"] > 0 for row in comparison))
            result = self.run_phase(output, "verify-incompatible")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_reload_rejects_wrong_values_and_abi(self):
        for field in ("values", "tuning_abi"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                report = self.prepare_profile(output)
                if field == "values":
                    report["calibrated_profiles"][0]["values"] = {"obsolete.parameter": 1}
                else:
                    report["calibrated_profiles"][0]["tuning_abi"] += 1
                kernel_reports.write_report(output / "calibration.json", report)
                result = self.run_phase(output, "verify")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "Schema mismatch" if field == "values" else "Incomplete reload", result.stderr
                )

    def test_collect_does_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_phase(Path(temporary), "collect")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)


if __name__ == "__main__":
    unittest.main()
