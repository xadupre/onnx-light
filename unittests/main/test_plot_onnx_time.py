import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from onnx_light import __file__ as onnxl_file
from onnx_light.ext_test_case import ExtTestCase

ROOT = os.path.realpath(os.path.abspath(os.path.join(onnxl_file, "..", "..")))
SCRIPT = os.path.join(ROOT, "docs", "examples", "core", "plot_onnx_time.py")


class TestPlotOnnxTime(ExtTestCase):
    def _run_plot_onnx_time(self, tmp: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["UNITTEST_GOING"] = "1"
        env["MPLBACKEND"] = "Agg"
        pythonpath_entries = [tmp, ROOT]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        proc = subprocess.run(
            [sys.executable, SCRIPT, "--scenario", "load", "--scenario", "save"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"plot_onnx_time.py failed with code {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return proc

    def test_plot_onnx_time_runs_ir_py_benchmarks_when_module_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "onnx_ir.py"), "w", encoding="utf-8") as f:
                f.write(textwrap.dedent("""
                        import onnx


                        def load(path, format=None):
                            return onnx.load(path)


                        def save(
                            model,
                            path,
                            format=None,
                            external_data=None,
                            size_threshold_bytes=256,
                            callback=None,
                        ):
                            if external_data is None:
                                onnx.save(model, path)
                                return
                            onnx.save_model(
                                model,
                                path,
                                save_as_external_data=True,
                                all_tensors_to_one_file=True,
                                location=external_data,
                                size_threshold=size_threshold_bytes,
                            )
                        """))

            proc = self._run_plot_onnx_time(tmp)

            self.assertIn("load/1filex1/ir-py", proc.stdout)
            self.assertIn("load/2filex1/ir-py", proc.stdout)
            self.assertIn("save/1filex1/ir-py", proc.stdout)
            self.assertIn("save/2filex1/ir-py", proc.stdout)
            self.assertExists(os.path.join(tmp, "plot_onnx_time.png"))

    def test_plot_onnx_time_skips_ir_py_benchmarks_when_module_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "onnx_ir.py"), "w", encoding="utf-8") as f:
                f.write("raise ImportError('onnx_ir intentionally unavailable for test')\n")

            proc = self._run_plot_onnx_time(tmp)

            self.assertIn("skipping ir-py single-file load benchmark", proc.stdout)
            self.assertIn("skipping ir-py external-data load benchmark", proc.stdout)
            self.assertIn("skipping ir-py save benchmarks", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
