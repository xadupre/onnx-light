import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_core import shape_inference as si


class TestPeakMemoryBindings(ExtTestCase):
    """Python tests for the peak-memory bindings exposed by
    ``onnx_light.onnx_core.shape_inference``."""

    def test_device_enum(self):
        self.assertEqual(int(si.Device.kUndefined), -2)
        self.assertEqual(int(si.Device.kCPU), -1)
        self.assertEqual(int(si.Device.kGPU0), 0)

    def test_attention_registered(self):
        self.assertIn("ai.onnx:Attention", si.peak_memory_dispatch_table_keys())

    def test_compute_peak_memory_attention_static(self):
        q = si.SymShape([2, 4, 8, 16])
        k = si.SymShape([2, 4, 32, 16])
        v = si.SymShape([2, 4, 32, 16])
        # batch * q_num_heads * q_seq * kv_seq * 4 bytes.
        estimate = si.compute_peak_memory("ai.onnx", "Attention", si.Device.kCPU, [q, k, v])
        self.assertIsInstance(estimate, si.PeakMemoryEstimate)
        self.assertIsInstance(estimate.worst_case, si.SymDim)
        self.assertEqual(estimate.worst_case.value(), 2 * 4 * 8 * 32 * 4)
        self.assertEqual(estimate.average_case.value(), estimate.worst_case.value())

    def test_compute_peak_memory_preserves_symbolic_dimensions(self):
        q = si.SymShape(["N", 4, 8, 16])
        k = si.SymShape([2, 4, 32, 16])
        estimate = si.compute_peak_memory("ai.onnx", "Attention", si.Device.kCPU, [q, k])
        self.assertEqual(estimate.worst_case.value(), "4096*N")
        self.assertEqual(estimate.average_case.value(), estimate.worst_case.value())

    def test_compute_peak_memory_empty_domain(self):
        q = si.SymShape([1, 2, 4, 8])
        k = si.SymShape([1, 2, 16, 8])
        self.assertEqual(
            si.compute_peak_memory("", "Attention", si.Device.kCPU, [q, k]).worst_case.value(),
            1 * 2 * 4 * 16 * 4,
        )

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
        # batch=2, heads=4, q_seq=8, kv_seq=32, float32=4 bytes
        expected = 2 * 4 * 8 * 32 * 4
        self.assertEqual(int(meta[si.NODE_PEAK_MEMORY_KEY]), expected)
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_MAX_KEY], str(expected))
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_AVERAGE_KEY], str(expected))

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
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_MAX_KEY], "4096*N")
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_AVERAGE_KEY], "4096*N")
        self.assertNotIn(si.NODE_PEAK_MEMORY_KEY, meta)
        graph.node[0].op_type = "UnregisteredPeakMemory"
        si.write_peak_memory_to_metadata(ctx, graph)
        meta = {kv.key: kv.value for kv in graph.node[0].metadata_props}
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_MAX_KEY], "0")
        self.assertEqual(meta[si.NODE_PEAK_MEMORY_AVERAGE_KEY], "0")
        self.assertNotIn(si.NODE_PEAK_MEMORY_KEY, meta)


if __name__ == "__main__":
    unittest.main()
