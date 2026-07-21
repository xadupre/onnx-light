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
        self.assertEqual(
            si.compute_peak_memory("ai.onnx", "Attention", si.Device.kCPU, [q, k, v]),
            2 * 4 * 8 * 32 * 4,
        )

    def test_compute_peak_memory_symbolic_returns_zero(self):
        q = si.SymShape(["N", 4, 8, 16])
        k = si.SymShape([2, 4, 32, 16])
        self.assertEqual(
            si.compute_peak_memory("ai.onnx", "Attention", si.Device.kCPU, [q, k]), 0
        )

    def test_compute_peak_memory_empty_domain(self):
        q = si.SymShape([1, 2, 4, 8])
        k = si.SymShape([1, 2, 16, 8])
        self.assertEqual(
            si.compute_peak_memory("", "Attention", si.Device.kCPU, [q, k]), 1 * 2 * 4 * 16 * 4
        )

    def test_compute_peak_memory_unregistered_returns_zero(self):
        q = si.SymShape([2, 4, 8, 16])
        self.assertEqual(si.compute_peak_memory("ai.onnx", "NoSuchOp", si.Device.kCPU, [q]), 0)

    def test_node_peak_memory_key_constant(self):
        self.assertEqual(si.NODE_PEAK_MEMORY_KEY, "onnx_light.peak_memory")

    def test_write_peak_memory_no_registered_op(self):
        """Nodes with no registered peak-memory function produce no metadata entry."""
        add = oh.make_node("Add", inputs=["X", "X"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([add], "g", [x], [y])

        ctx = si.ShapesContext()
        ctx.set("X", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 3]))
        ctx.set("Y", si.SymTensor(onnxl.TensorProto.FLOAT, [2, 3]))

        si.write_peak_memory_to_metadata(ctx, graph)

        self.assertEqual(len(graph.node[0].metadata_props), 0)

    def test_write_peak_memory_with_device_argument(self):
        """write_peak_memory_to_metadata accepts an explicit Device argument."""
        add = oh.make_node("Add", inputs=["X", "X"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([add], "g", [x], [y])

        ctx = si.ShapesContext()
        si.write_peak_memory_to_metadata(ctx, graph, si.Device.kCPU)
        self.assertEqual(len(graph.node[0].metadata_props), 0)

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

    def test_write_peak_memory_symbolic_produces_no_entry(self):
        """Attention nodes with symbolic shapes produce no peak-memory entry."""
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

        si.write_peak_memory_to_metadata(ctx, graph)

        # Symbolic batch dimension → peak memory returns 0 → no entry.
        self.assertEqual(len(graph.node[0].metadata_props), 0)


if __name__ == "__main__":
    unittest.main()
