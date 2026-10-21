import importlib.util
import subprocess
import sys
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

    def test_optional_size_budget_accepts_limit(self):
        measurements = [{"source": "lib.so", "text_size": 1024}]
        self.reporter._enforce_optional_size_budget(measurements, "text_size", ".text size", 1024)

    def test_size_budget_margin_boundary(self):
        for key in ("installed_size", "text_size"):
            with self.subTest(key=key):
                measurements = [{"source": "lib.so", key: 106}]
                if key == "installed_size":
                    enforce = self.reporter._enforce_installed_size_budget
                    arguments = ()
                else:
                    enforce = self.reporter._enforce_optional_size_budget
                    arguments = (key, ".text size")
                enforce(measurements, *arguments, 101, margin_percent=5)
                measurements[0][key] = 107
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"maximum 106.*lib\.so: 107.*Reference: 101 bytes; margin: 5%.*"
                    r"New proto fields, native code, or compiler/build changes",
                ):
                    enforce(measurements, *arguments, 101, margin_percent=5)

    def test_negative_size_budget_margin_is_rejected(self):
        result = subprocess.run(
            [
                sys.executable,
                self.reporter.__file__,
                "unused.so",
                "--size-budget-margin-percent",
                "-1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--size-budget-margin-percent must be non-negative", result.stderr)

    def test_text_size_budget_boundary(self):
        measurements = [{"source": "lib.so", "text_size": 1_174_506}]
        self.reporter._enforce_optional_size_budget(
            measurements, "text_size", ".text size", 1_174_506
        )
        measurements[0]["text_size"] += 1
        with self.assertRaisesRegex(
            RuntimeError, r"\.text size budget exceeded.*maximum 1,174,506.*lib\.so: 1,174,507"
        ):
            self.reporter._enforce_optional_size_budget(
                measurements, "text_size", ".text size", 1_174_506
            )

    def test_optional_size_budget_rejects_unavailable_measurement(self):
        measurements = [{"source": "lib.so", "dynamic_symbols": None}]
        with self.assertRaisesRegex(
            RuntimeError, r"defined dynamic-symbol count unavailable for: lib\.so"
        ):
            self.reporter._enforce_optional_size_budget(
                measurements, "dynamic_symbols", "defined dynamic-symbol count", 760
            )

    def test_optional_size_budget_rejects_oversized_measurement(self):
        measurements = [{"source": "lib.so", "dynamic_symbols": 761}]
        with self.assertRaisesRegex(RuntimeError, r"maximum 760.*lib\.so: 761"):
            self.reporter._enforce_optional_size_budget(
                measurements, "dynamic_symbols", "defined dynamic-symbol count", 760
            )

    def test_allowed_dependencies_accepts_subset(self):
        measurements = [{"source": "lib.so", "dependencies": ["libc.so.6"]}]
        self.reporter._enforce_allowed_dependencies(measurements, {"libc.so.6", "libstdc++.so.6"})

    def test_allowed_dependencies_rejects_addition(self):
        measurements = [{"source": "lib.so", "dependencies": ["libc.so.6", "libdecoder.so.1"]}]
        with self.assertRaisesRegex(
            RuntimeError, r"added shared dependencies: lib\.so: libdecoder\.so\.1"
        ):
            self.reporter._enforce_allowed_dependencies(measurements, {"libc.so.6"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
