import unittest
import os
import sys
import importlib.util
import subprocess
import time
from onnx_light import __file__ as onnxl_file
from onnx_light.ext_test_case import (
    ExtTestCase,
    has_onnxruntime,
    is_windows,
    ignore_errors,
    import_or_skip,
)

# The documentation examples exercise the operator-kernel runtime and backend,
# which are absent from the reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF). Skip
# the whole module in that case.
import_or_skip("onnx_light.onnx_py._onnxpykernels")

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(onnxl_file, "..", "..")))


def has_onnx():
    try:
        import onnx  # noqa: F401

        return True
    except:  # noqa: E722
        return False


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(module_name, module_file_path)
        )
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(ExtTestCase):
    def run_test(self, fold: str, name: str, verbose=0) -> int:
        ppath = os.environ.get("PYTHONPATH", "")
        if not ppath:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if is_windows() else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        try:
            mod = import_source(fold, os.path.splitext(name)[0])
            assert mod is not None
        except FileNotFoundError:
            # try another way
            cmds = [sys.executable, "-u", os.path.join(fold, name)]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = p.communicate()
            _out, err = res
            st = err.decode("ascii", errors="ignore")
            if st and "Traceback" in st:
                if '"dot" not found in path.' in st:
                    # dot not installed, this part
                    # is tested in onnx framework
                    raise unittest.SkipTest(f"failed: {name!r} due to missing dot.")
                if (
                    "We couldn't connect to 'https://huggingface.co'" in st
                    or "Cannot access content at: https://huggingface.co/" in st
                ):
                    raise unittest.SkipTest(f"Connectivity issues due to\n{err}")
                raise AssertionError(  # noqa: B904
                    "Example '{}' (cmd: {} - exec_prefix='{}') "
                    "failed due to\n{}"
                    "".format(name, cmds, sys.exec_prefix, st)
                )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    def test_check_unittest_going_is_true(self):
        self.assertIn("UNITTEST_GOING", os.environ)
        self.assertEqual(os.environ["UNITTEST_GOING"], "1")

    @classmethod
    def add_test_methods(cls):
        this = os.path.abspath(os.path.dirname(__file__))
        root_fold = os.path.normpath(os.path.join(this, "..", "..", "docs", "examples"))
        # Collect (fold, name) pairs from all subdirectories
        found = []
        for subdir in ("runtime", "core", "proto", "patterns"):
            fold = os.path.join(root_fold, subdir)
            if os.path.isdir(fold):
                for name in os.listdir(fold):
                    if name.endswith(".py") and name.startswith("plot_"):
                        found.append((fold, name))
        _has_dot = int(os.environ.get("UNITTEST_DOT", "0"))
        for fold, name in found:
            reason = None

            if (
                not reason
                and not has_onnx()
                and name in {"plot_save_external_data_time.py", "plot_onnx_time.py"}
            ):
                reason = "onnx is missing"

            if (
                not reason
                and not has_onnxruntime()
                and name
                in {
                    "plot_save_ort_flatbuffers.py",
                    "plot_qwen3_init_benchmark.py",
                    "plot_backend_benchmark_vs_onnxruntime.py",
                }
            ):
                reason = "onnxruntime is missing"

            if reason:

                @unittest.skip(reason)
                def _test_(self, fold=fold, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            else:

                @ignore_errors(OSError)  # connectivity issues
                def _test_(self, fold=fold, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            short_name = os.path.split(os.path.splitext(name)[0])[-1]
            setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
