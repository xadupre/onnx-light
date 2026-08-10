import importlib.util
import unittest
from pathlib import Path


def _load_reporter():
    script = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "scripts"
        / "report_proto_binary_size.py"
    )
    spec = importlib.util.spec_from_file_location("report_proto_binary_size", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReportProtoBinarySize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reporter = _load_reporter()

    def test_installed_size_budget_accepts_limit(self):
        measurements = [{"source": "lib.so", "installed_size": 1_048_576}]
        self.reporter._enforce_installed_size_budget(measurements, 1_048_576)

    def test_installed_size_budget_rejects_oversized_library(self):
        measurements = [{"source": "lib.so", "installed_size": 1_048_577}]
        with self.assertRaisesRegex(
            RuntimeError, r"maximum 1,048,576 bytes.*lib\.so: 1,048,577 bytes"
        ):
            self.reporter._enforce_installed_size_budget(measurements, 1_048_576)


if __name__ == "__main__":
    unittest.main(verbosity=2)
