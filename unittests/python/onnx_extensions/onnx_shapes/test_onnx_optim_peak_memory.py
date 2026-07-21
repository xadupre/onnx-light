import unittest

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


if __name__ == "__main__":
    unittest.main()
