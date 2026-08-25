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
    def test_lists_registered_kernels(self):
        identifiers = rt.registered_kernels()
        self.assertEqual(identifiers, sorted(identifiers))
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn("ai.onnx:Abs", identifiers)
        self.assertIn("ai.onnx:Gemm", identifiers)

    def test_calibration_accepts_explicit_cpu_executor(self):
        policy = rt.CpuExecutionPolicy()
        policy.num_threads = 1
        policy.affinity_policy = rt.CpuAffinityPolicy.NONE

        report = rt.calibrate_kernel_tuning(
            "not_registered",
            save=False,
            cpu_execution=policy,
            profiling_capacity=4,
            profiling_hardware_counters=False,
        )

        self.assertEqual(report["calibrated"], [])
        self.assertEqual(report["candidate_diagnostics"], [])

    def test_calibration_exposes_bounded_parallel_diagnostics(self):
        report = rt.calibrate_kernel_tuning(
            "Abs",
            element_types=[int(TensorProto.FLOAT)],
            maximum_duration_ms=25,
            save=False,
            profiling_capacity=1,
            profiling_hardware_counters=False,
        )

        self.assertEqual(len(report["candidate_diagnostics"]), 1)
        diagnostic = report["candidate_diagnostics"][0]
        self.assertEqual(diagnostic["kernel"], "Abs")
        self.assertLessEqual(len(diagnostic["events"]), 1)
        self.assertGreaterEqual(diagnostic["dropped_events"], 0)
        if diagnostic["events"]:
            event = diagnostic["events"][0]
            self.assertIn("cpu_utilization", event)
            self.assertEqual(event["counter_status"], "disabled")
            self.assertIsNone(event["ipc"])
            self.assertIsNone(event["llc_miss_rate"])

    def test_calibrates_gemm_parallel_minimum_tasks(self):
        report = rt.calibrate_kernel_tuning(
            "Gemm",
            element_types=[int(TensorProto.FLOAT)],
            maximum_duration_ms=1000,
            save=False,
            profiling_capacity=1,
            profiling_hardware_counters=False,
        )

        self.assertEqual(len(report["candidate_diagnostics"]), 1)
        diagnostic = report["candidate_diagnostics"][0]
        self.assertEqual(diagnostic["kernel"], "Gemm")

    def test_lists_registered_parameters_and_defaults(self):
        report = kernel_tuning.kernel_tuning_parameters(
            kernel="Abs", element_type=int(TensorProto.FLOAT)
        )

        self.assertIn(report["cache_status"], {"loaded", "not_found"})
        self.assertEqual(len(report["kernels"]), 1)
        kernel = report["kernels"][0]
        self.assertEqual(kernel["library"], "onnx_light")
        self.assertEqual(kernel["kernel"], "Abs")
        self.assertEqual(kernel["device"], -1)
        self.assertEqual(kernel["device_name"], "CPU")
        self.assertEqual(kernel["implementation"], "portable")
        self.assertEqual(kernel["parameter_names"], ["parallel.minimum_elements"])
        self.assertGreater(kernel["defaults"]["parallel.minimum_elements"], 0)
        self.assertEqual(set(kernel["active_values"]), {"parallel.minimum_elements"})

        no_gpu_schema = kernel_tuning.kernel_tuning_parameters(
            kernel="Abs", element_type=int(TensorProto.FLOAT), device=0
        )
        self.assertEqual(no_gpu_schema["kernels"], [])

        no_library_schema = kernel_tuning.kernel_tuning_parameters(
            kernel="Abs", element_type=int(TensorProto.FLOAT), library="unknown_library"
        )
        self.assertEqual(no_library_schema["kernels"], [])

        all_schemas = kernel_tuning.kernel_tuning_parameters(
            kernel="Abs", library=None, device=None
        )
        self.assertGreater(len(all_schemas["kernels"]), 0)

        with self.assertRaisesRegex(ValueError, "Unknown kernel tuning device"):
            kernel_tuning.kernel_tuning_parameters(device=8192)

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
