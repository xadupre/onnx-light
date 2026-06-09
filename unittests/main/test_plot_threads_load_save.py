import ast
import pathlib
import unittest


def _has_xlim_left_zero() -> bool:
    """Checks whether the example sets the x-axis minimum to zero."""
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_threads_load_save.py"
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


class TestPlotThreadsLoadSave(unittest.TestCase):
    def test_example_sets_xlim_left_to_zero(self):
        self.assertTrue(_has_xlim_left_zero())


if __name__ == "__main__":
    unittest.main(verbosity=2)
