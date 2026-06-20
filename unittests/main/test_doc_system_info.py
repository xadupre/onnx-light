"""Tests for the system-information helpers in onnx_light.doc."""

import unittest
from unittest import mock

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

doc_module = import_or_skip("onnx_light.doc")


class TestGetCpuTopology(ExtTestCase):
    """Tests for :func:`onnx_light.doc.get_cpu_topology`."""

    def test_returns_expected_keys(self):
        topology = doc_module.get_cpu_topology()
        self.assertIsInstance(topology, dict)
        self.assertEqual(set(topology), {"logical", "physical_cores", "sockets"})
        for value in topology.values():
            self.assertTrue(value is None or isinstance(value, int))

    def test_parses_proc_cpuinfo_multiple_sockets(self):
        cpuinfo = (
            "processor\t: 0\n"
            "physical id\t: 0\n"
            "cpu cores\t: 2\n"
            "\n"
            "processor\t: 1\n"
            "physical id\t: 0\n"
            "cpu cores\t: 2\n"
            "\n"
            "processor\t: 2\n"
            "physical id\t: 1\n"
            "cpu cores\t: 2\n"
            "\n"
            "processor\t: 3\n"
            "physical id\t: 1\n"
            "cpu cores\t: 2\n"
            "\n"
        )
        with mock.patch("builtins.open", mock.mock_open(read_data=cpuinfo)):
            topology = doc_module.get_cpu_topology()
        self.assertEqual(topology["logical"], 4)
        self.assertEqual(topology["sockets"], 2)
        self.assertEqual(topology["physical_cores"], 4)

    def test_parses_proc_cpuinfo_without_trailing_blank_line(self):
        cpuinfo = "processor\t: 0\nphysical id\t: 0\ncpu cores\t: 4\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=cpuinfo)):
            topology = doc_module.get_cpu_topology()
        self.assertEqual(topology["logical"], 1)
        self.assertEqual(topology["sockets"], 1)
        self.assertEqual(topology["physical_cores"], 4)

    def test_ignores_non_integer_cpu_cores(self):
        cpuinfo = "processor\t: 0\nphysical id\t: 0\ncpu cores\t: not-a-number\n\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=cpuinfo)):
            topology = doc_module.get_cpu_topology()
        self.assertEqual(topology["logical"], 1)
        # The socket is never recorded because ``cpu cores`` could not be parsed.
        self.assertIsNone(topology["sockets"])
        self.assertIsNone(topology["physical_cores"])

    def test_falls_back_to_cpu_count_when_proc_unavailable(self):
        with (
            mock.patch("builtins.open", side_effect=OSError("no /proc")),
            mock.patch("onnx_light.doc.os.cpu_count", return_value=8),
        ):
            topology = doc_module.get_cpu_topology()
        self.assertEqual(topology["logical"], 8)
        self.assertIsNone(topology["physical_cores"])
        self.assertIsNone(topology["sockets"])


class TestGetTotalMemoryGb(ExtTestCase):
    """Tests for :func:`onnx_light.doc.get_total_memory_gb`."""

    def test_returns_positive_value(self):
        memory = doc_module.get_total_memory_gb()
        self.assertTrue(memory is None or memory > 0)

    def test_uses_sysconf(self):
        def fake_sysconf(name):
            return {"SC_PHYS_PAGES": 1024**2, "SC_PAGE_SIZE": 1024}[name]

        with mock.patch("onnx_light.doc.os.sysconf", side_effect=fake_sysconf):
            memory = doc_module.get_total_memory_gb()
        self.assertAlmostEqual(memory, 1.0)

    def test_falls_back_to_meminfo(self):
        meminfo = "MemTotal:        2097152 kB\nMemFree:         1048576 kB\n"
        with (
            mock.patch("onnx_light.doc.os.sysconf", side_effect=OSError("unsupported")),
            mock.patch("builtins.open", mock.mock_open(read_data=meminfo)),
        ):
            memory = doc_module.get_total_memory_gb()
        self.assertAlmostEqual(memory, 2.0)

    def test_returns_none_when_unavailable(self):
        with (
            mock.patch("onnx_light.doc.os.sysconf", side_effect=ValueError("unsupported")),
            mock.patch("builtins.open", side_effect=OSError("no /proc")),
        ):
            self.assertIsNone(doc_module.get_total_memory_gb())


if __name__ == "__main__":
    unittest.main(verbosity=2)
