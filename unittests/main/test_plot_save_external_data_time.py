import ast
import pathlib
import unittest


def _get_profile_call_keywords(result_name: str) -> dict[str, ast.AST]:
    """Returns keyword AST nodes for the ``profile_call`` named by *result_name*."""
    root = pathlib.Path(__file__).resolve().parents[1]
    source_path = root / "docs" / "examples" / "core" / "plot_save_external_data_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "profile_call":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value == result_name:
            return {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
    raise AssertionError(f"Unable to find profile_call for {result_name!r}")


def _get_profile_call_callable(result_name: str) -> ast.AST:
    """Returns callable AST node for the ``profile_call`` named by *result_name*."""
    root = pathlib.Path(__file__).resolve().parents[1]
    source_path = root / "docs" / "examples" / "core" / "plot_save_external_data_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "profile_call":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value == result_name:
            return node.args[1]
    raise AssertionError(f"Unable to find profile_call for {result_name!r}")


class TestPlotSaveExternalDataTime(unittest.TestCase):
    def test_external_save_benchmarks_are_single_shot(self):
        for result_name in (
            "save/2filex1/onnx",
            "save/2filex1/onnxlight",
            "save/2filex4/onnxlight",
        ):
            with self.subTest(result_name=result_name):
                keywords = _get_profile_call_keywords(result_name)
                self.assertEqual(1, keywords["repeat"].value)

    def test_external_save_benchmarks_use_flush_helpers(self):
        for result_name, helper_name in (
            ("save/2filex1/onnx", "_save_onnx_external_with_flush"),
            ("save/2filex1/onnxlight", "_save_onnxlight_external_with_flush"),
        ):
            with self.subTest(result_name=result_name):
                fn = _get_profile_call_callable(result_name)
                self.assertIsInstance(fn, ast.Name)
                self.assertEqual(helper_name, fn.id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
