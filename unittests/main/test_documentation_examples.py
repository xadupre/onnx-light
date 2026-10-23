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
    def test_tensor_statistics_results(self):
        import runpy
        import matplotlib.pyplot as plt
        import numpy
        from onnx_light.onnx_core.quantization import dequantize_tensor_proto
        import onnx_light.onnx.numpy_helper as onh

        example = os.path.join(ROOT, "docs", "examples", "runtime", "plot_tensor_statistics.py")
        namespace = runpy.run_path(example)
        self.addCleanup(plt.close, namespace["fig"])
        tensors = namespace["tensors"]
        statistics = namespace["statistics"]
        self.assertEqual(list(statistics), ["float32", "int32", "INT8", "INT4", "INT3", "NF4"])
        for name, stats in statistics.items():
            with self.subTest(name=name):
                value = tensors[name]
                if name in ("INT8", "INT4", "INT3", "NF4"):
                    expected = onh.to_array(dequantize_tensor_proto(value))
                    self.assertEqual(stats.size_bytes, len(value.raw_data))
                else:
                    expected = value
                    self.assertEqual(stats.size_bytes, expected.nbytes)
                self.assertEqual(stats.shape, [8, 16])
                self.assertEqual(stats.count, expected.size)
                self.assertEqual(stats.min, expected.min())
                self.assertEqual(stats.max, expected.max())
                self.assertAlmostEqual(stats.mean, expected.astype(numpy.float64).mean())
                self.assertAlmostEqual(stats.median, numpy.median(expected))
                numpy.testing.assert_allclose(
                    stats.quantiles, numpy.quantile(expected, [0.25, 0.75])
                )
                self.assertEqual(
                    stats.distinct_count,
                    None if name == "float32" else len(numpy.unique(expected)),
                )
                counts, edges = numpy.histogram(expected, bins=16)
                numpy.testing.assert_array_equal(stats.distribution, counts)
                numpy.testing.assert_allclose(stats.bin_edges, edges)
                self.assertEqual(stats.rectangle_shape, [4, 8])
                numpy.testing.assert_allclose(
                    numpy.array(stats.rectangle_map).reshape(4, 8),
                    expected.reshape(4, 2, 8, 2).mean(axis=(1, 3)),
                    atol=1e-7,
                )

    def test_parallel_for_profiling_tuning_results(self):
        example = os.path.join(
            ROOT, "docs", "examples", "tuning", "plot_parallel_for_profiling.py"
        )
        result = subprocess.run(
            [sys.executable, example],
            env={**os.environ, "PYTHONPATH": ROOT, "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        self.assertIn("events=", result.stdout)
        self.assertRegex(result.stdout, r"parallel.minimum_elements: baseline=\d+, selected=\d+")
        self.assertIn("speedup vs baseline", result.stdout)
        self.assertRegex(result.stdout, r"\d+\s+\|\s+[\d.]+\s+\|\s+1.000x")
        self.assertRegex(result.stdout, r"\|\s+True")
        self.assertIn("calibrated profile:", result.stdout)
        self.assertIn("'parallel.minimum_elements':", result.stdout)
        self.assertIn("calibration diagnostics:", result.stdout)

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
                and name
                in {
                    "plot_onnx_cold_start.py",
                    "plot_save_external_data_time.py",
                    "plot_onnx_time.py",
                }
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
