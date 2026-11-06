import importlib.util
import subprocess
import sys
import textwrap
import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_core import shape_inference as si

HAS_KERNELS = importlib.util.find_spec("onnx_light.onnx_py._onnxpykernels") is not None


class TestPeakMemoryBindings(ExtTestCase):
    """Python tests for the peak-memory bindings exposed by
    ``onnx_light.onnx_core.shape_inference``."""

    def test_estimators_are_loaded_from_the_optional_kernel_library(self):
        script = textwrap.dedent("""
            import importlib.util
            import sys
            from onnx_light.onnx_py._onnxpycore import shape_inference as native

            shapes = [native.SymShape(s) for s in ([2, 4, 8, 16], [2, 4, 32, 16], [2, 4, 32, 16])]
            assert "onnx_light.onnx_py._onnxpykernels" not in sys.modules
            assert "ai.onnx:Attention" not in native.peak_memory_dispatch_table_keys()
            estimate = native.compute_peak_memory("", "Attention", native.Device.kCPU, shapes)
            assert estimate.worst_case.value() == 0

            spec = importlib.util.find_spec("onnx_light.onnx_py._onnxpykernels")
            has_kernels = spec is not None
            from onnx_light.onnx_core import shape_inference as facade
            estimate = facade.compute_peak_memory("", "Attention", facade.Device.kCPU, shapes)
            assert estimate.worst_case.value() == (82944 if has_kernels else 0)
            assert ("onnx_light.onnx_py._onnxpykernels" in sys.modules) == has_kernels
            """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_device_enum(self):
        self.assertEqual(int(si.Device.kUndefined), -2)
        self.assertEqual(int(si.Device.kCPU), -1)
        self.assertEqual(int(si.Device.kGPU0), 0)

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_attention_registered(self):
        self.assertIn("ai.onnx:Attention", si.peak_memory_dispatch_table_keys())

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_compute_peak_memory_attention_static(self):
        q = si.SymShape([2, 4, 8, 16])
        k = si.SymShape([2, 4, 32, 16])
        v = si.SymShape([2, 4, 32, 16])
        # Shape-only calls conservatively include all optional output staging.
        estimate = si.compute_peak_memory("ai.onnx", "Attention", si.Device.kCPU, [q, k, v])
        self.assertIsInstance(estimate, si.PeakMemoryEstimate)
        self.assertIsInstance(estimate.worst_case, si.SymDim)
        self.assertEqual(estimate.worst_case.value(), 82944)
        self.assertIsNone(estimate.average_case)

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_compute_peak_memory_preserves_symbolic_dimensions(self):
        q = si.SymShape(["N", 4, 8, 16])
        k = si.SymShape(["N", 4, 32, 16])
        v = si.SymShape(["N", 4, 32, 16])
        estimate = si.compute_peak_memory("ai.onnx", "Attention", si.Device.kCPU, [q, k, v])
        self.assertEqual(estimate.worst_case.value(), "41472*N")
        self.assertIsNone(estimate.average_case)

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_compute_peak_memory_empty_domain(self):
        q = si.SymShape([1, 2, 4, 8])
        k = si.SymShape([1, 2, 16, 8])
        v = si.SymShape([1, 2, 16, 8])
        self.assertEqual(
            si.compute_peak_memory("", "Attention", si.Device.kCPU, [q, k, v]).worst_case.value(),
            5248,
        )

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_compute_peak_memory_missing_value_shape_is_unknown(self):
        q = si.SymShape([1, 2, 4, 8])
        k = si.SymShape([1, 2, 16, 8])
        estimate = si.compute_peak_memory("", "Attention", si.Device.kCPU, [q, k])
        self.assertIsInstance(estimate.worst_case.value(), str)
        self.assertIn("peak_memory_max_", estimate.worst_case.value())
        self.assertIsNone(estimate.average_case)

    def test_compute_peak_memory_unregistered_defaults_to_zero(self):
        q = si.SymShape([2, 4, 8, 16])
        estimate = si.compute_peak_memory("ai.onnx", "NoSuchOp", si.Device.kCPU, [q])
        self.assertEqual(estimate.worst_case.value(), 0)
        self.assertEqual(estimate.average_case.value(), 0)

    def test_node_peak_memory_key_constant(self):
        self.assertEqual(si.NODE_PEAK_MEMORY_KEY, "onnx_light.peak_memory")

    def test_write_peak_memory_no_registered_op(self):
        """Uses zero for operators without an estimator."""
        add = oh.make_node("Add", inputs=["X", "X"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([add], "g", [x], [y])

        ctx = si.ShapesContext()
        ctx.set("X", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 3]))
        ctx.set("Y", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 3]))

        si.write_peak_memory_to_metadata(ctx, graph)

        meta = {kv.key: kv.value for kv in graph.node[0].metadata_props}
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_MAX_KEY], "0")
        self.assertNotIn(si.NODE_PEAK_MEMORY_KEY, meta)
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_AVERAGE_KEY], "0")

    def test_write_peak_memory_with_device_argument(self):
        """write_peak_memory_to_metadata accepts an explicit Device argument."""
        add = oh.make_node("Add", inputs=["X", "X"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([add], "g", [x], [y])

        ctx = si.ShapesContext()
        si.write_peak_memory_to_metadata(ctx, graph, si.Device.kCPU)
        meta = {kv.key: kv.value for kv in graph.node[0].metadata_props}
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_MAX_KEY], "0")
        self.assertNotIn(si.NODE_PEAK_MEMORY_KEY, meta)

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_write_peak_memory_attention(self):
        """Attention nodes with static shapes get a peak-memory metadata entry."""
        attention = oh.make_node(
            "Attention", inputs=["Q", "K", "V"], outputs=["out"], domain="ai.onnx"
        )
        q_vi = oh.make_tensor_value_info("Q", onnxl.TensorProto.FLOAT, [2, 4, 8, 16])
        k_vi = oh.make_tensor_value_info("K", onnxl.TensorProto.FLOAT, [2, 4, 32, 16])
        v_vi = oh.make_tensor_value_info("V", onnxl.TensorProto.FLOAT, [2, 4, 32, 16])
        out_vi = oh.make_tensor_value_info("out", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([attention], "g", [q_vi, k_vi, v_vi], [out_vi])

        ctx = si.ShapesContext()
        ctx.set("Q", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 4, 8, 16]))
        ctx.set("K", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 4, 32, 16]))
        ctx.set("V", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 4, 32, 16]))

        si.write_peak_memory_to_metadata(ctx, graph, si.Device.kCPU)

        meta = {kv.key: kv.value for kv in graph.node[0].metadata_props}
        self.assertIn(si.NODE_PEAK_MEMORY_KEY, meta)
        # Only Y is declared: no optional cache or qk staging.
        expected = 41984
        self.assertEqual(int(meta[si.NODE_PEAK_MEMORY_KEY]), expected)
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_MAX_KEY], str(expected))
        self.assertNotIn(si.NODE_PEAK_MEMORY_AVERAGE_KEY, meta)

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_write_peak_memory_refreshes_symbolic_estimates_and_concrete_hint(self):
        """Preserves symbolic bounds and removes stale hints and averages."""
        attention = oh.make_node(
            "Attention", inputs=["Q", "K", "V"], outputs=["out"], domain="ai.onnx"
        )
        q_vi = oh.make_tensor_value_info("Q", onnxl.TensorProto.FLOAT, ["N", 4, 8, 16])
        k_vi = oh.make_tensor_value_info("K", onnxl.TensorProto.FLOAT, [2, 4, 32, 16])
        v_vi = oh.make_tensor_value_info("V", onnxl.TensorProto.FLOAT, [2, 4, 32, 16])
        out_vi = oh.make_tensor_value_info("out", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([attention], "g", [q_vi, k_vi, v_vi], [out_vi])

        ctx = si.ShapesContext()
        ctx.set("Q", si.SymTensor(onnxl.TensorProto.FLOAT, ["N", 4, 8, 16]))
        ctx.set("K", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 4, 32, 16]))
        ctx.set("V", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 4, 32, 16]))

        ctx.set("Q", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 4, 8, 16]))
        si.write_peak_memory_to_metadata(ctx, graph)
        self.assertIn(si.NODE_PEAK_MEMORY_KEY, {kv.key for kv in graph.node[0].metadata_props})
        ctx.set("Q", si.SymTensor(onnxl.TensorProto.FLOAT, ["N", 4, 8, 16]))
        si.write_peak_memory_to_metadata(ctx, graph)
        meta = {kv.key: kv.value for kv in graph.node[0].metadata_props}
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_MAX_KEY], "4608*N+32768")
        self.assertNotIn(si.NODE_PEAK_MEMORY_AVERAGE_KEY, meta)
        self.assertNotIn(si.NODE_PEAK_MEMORY_KEY, meta)
        graph.node[0].op_type = "UnregisteredPeakMemory"
        si.write_peak_memory_to_metadata(ctx, graph)
        meta = {kv.key: kv.value for kv in graph.node[0].metadata_props}
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_MAX_KEY], "0")
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_AVERAGE_KEY], "0")
        self.assertNotIn(si.NODE_PEAK_MEMORY_KEY, meta)


if __name__ == "__main__":
    unittest.main()
