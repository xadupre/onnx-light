import ast
import pathlib
import unittest

from onnx_light.ext_test_case import ExtTestCase


def _plot_threads_example_has_xlim_left_zero() -> bool:
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "proto" / "plot_threads_load_save.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "set_xlim":
            continue
        for keyword in node.keywords:
            if keyword.arg == "left" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value == 0:
                    return True
    return False


class TestPlotThreadsLoadSave(ExtTestCase):
    def test_example_sets_xlim_left_to_zero(self):
        """Verifies that the thread plot example pins the x-axis lower bound to zero."""
        self.assertTrue(_plot_threads_example_has_xlim_left_zero())


if __name__ == "__main__":
    unittest.main(verbosity=2)
