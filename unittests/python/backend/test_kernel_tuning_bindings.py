# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx import TensorProto

rt = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
from onnx_light import kernel_tuning  # noqa: E402


class TestKernelTuningBindings(ExtTestCase):
    def test_calibration_accepts_explicit_cpu_executor(self):
        policy = rt.CpuExecutionPolicy()
        policy.num_threads = 1
        policy.affinity_policy = rt.CpuAffinityPolicy.NONE

        report = rt.calibrate_kernel_tuning("not_registered", save=False, cpu_execution=policy)

        self.assertEqual(report["calibrated"], [])

    def test_lists_registered_parameters_and_defaults(self):
        report = kernel_tuning.kernel_tuning_parameters(
            kernel="Abs", element_type=int(TensorProto.FLOAT)
        )

        self.assertIn(report["cache_status"], {"loaded", "not_found"})
        self.assertEqual(len(report["kernels"]), 1)
        kernel = report["kernels"][0]
        self.assertEqual(kernel["library"], "onnx_light")
        self.assertEqual(kernel["kernel"], "Abs")
        self.assertEqual(kernel["implementation"], "portable")
        self.assertEqual(kernel["parameter_names"], ["parallel.minimum_elements"])
        self.assertGreater(kernel["defaults"]["parallel.minimum_elements"], 0)
        self.assertEqual(set(kernel["active_values"]), {"parallel.minimum_elements"})

    def test_updates_inspects_and_loads_local_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "kernel_tuning.cache")
            update = kernel_tuning.set_kernel_tuning_parameters(
                "Abs", int(TensorProto.FLOAT), {"parallel.minimum_elements": 12345}, path=path
            )
            self.assertEqual(update["status"], "updated")
            self.assertEqual(update["load"]["status"], "loaded")

            inspection = kernel_tuning.inspect_kernel_tuning_cache(path)
            self.assertEqual(inspection["status"], "loaded")
            self.assertEqual(len(inspection["profiles"]), 1)
            self.assertTrue(inspection["profiles"][0]["local"])
            self.assertEqual(
                inspection["profiles"][0]["values"]["parallel.minimum_elements"], 12345
            )

            report = rt.kernel_tuning_parameters(
                kernel="Abs", element_type=int(TensorProto.FLOAT), path=path
            )
            kernel = report["kernels"][0]
            self.assertEqual(kernel["cached_values"]["parallel.minimum_elements"], 12345)
            self.assertEqual(kernel["active_values"]["parallel.minimum_elements"], 12345)
            self.assertEqual(kernel["active_source"], "published_profile")

    def test_rejects_unknown_or_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "kernel_tuning.cache")
            with self.assertRaises(ValueError):
                rt.set_kernel_tuning_parameters(
                    "Abs", int(TensorProto.FLOAT), {"unknown": 1}, path=path
                )
            with self.assertRaises(ValueError):
                rt.set_kernel_tuning_parameters(
                    "Abs", int(TensorProto.FLOAT), {"parallel.minimum_elements": 0}, path=path
                )
            self.assertFalse(Path(path).exists())

    def test_load_reports_missing_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "missing.cache")
            report = rt.load_kernel_tuning_cache(path=path)
            self.assertEqual(report["status"], "not_found")
            self.assertEqual(report["path"], path)

    def test_default_cache_is_loaded_on_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "onnx-light" / "kernel_tuning.cache"
            rt.set_kernel_tuning_parameters(
                "Abs",
                int(TensorProto.FLOAT),
                {"parallel.minimum_elements": 23456},
                path=str(cache),
                load=False,
            )
            code = """
from onnx_light.onnx import TensorProto
from onnx_light import kernel_tuning
report = kernel_tuning.kernel_tuning_parameters(
    kernel="Abs", element_type=int(TensorProto.FLOAT)
)
kernel = report["kernels"][0]
assert kernel["active_values"]["parallel.minimum_elements"] == 23456, kernel
assert kernel["active_source"] == "published_profile", kernel
"""
            env = os.environ.copy()
            if sys.platform == "win32":
                env["LOCALAPPDATA"] = temporary
            else:
                env["XDG_CACHE_HOME"] = temporary
            subprocess.run([sys.executable, "-c", code], check=True, env=env)

    def test_proposes_missing_subset_and_cli_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "missing.cache")
            proposal = kernel_tuning.propose_kernel_tuning_updates(
                kernels=["Abs"], element_types=[int(TensorProto.FLOAT)], path=path
            )
            self.assertEqual(proposal["selected"], 1)
            self.assertEqual(proposal["covered"], 0)
            self.assertEqual(len(proposal["calibratable"]), 1)
            self.assertEqual(proposal["calibratable"][0]["kernel"], "Abs")

            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "onnx_light",
                    "tune-kernels",
                    "--kernel",
                    "Abs",
                    "--element-type",
                    "FLOAT",
                    "--cache",
                    path,
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cli = json.loads(process.stdout)
            self.assertEqual(cli["selected"], 1)
            self.assertEqual(len(cli["calibratable"]), 1)
            self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
