import ast
import pathlib
import unittest


def _example_mentions_nonzero_shape_events() -> bool:
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "optimization" / "plot_shape_inference.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    has_case_name = False
    has_events_enabled = False
    has_compute_shape_model_call = False

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and node.value == "test_cc_shape_inference_nonzero_chain_named"
        ):
            has_case_name = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "events_enabled":
                    if isinstance(node.value, ast.Constant) and node.value.value is True:
                        has_events_enabled = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "compute_shape_model":
                has_compute_shape_model_call = True

    return has_case_name and has_events_enabled and has_compute_shape_model_call


class TestPlotShapeInferenceEvents(unittest.TestCase):
    def test_example_contains_nonzero_shape_inference_events_section(self):
        self.assertTrue(_example_mentions_nonzero_shape_events())


if __name__ == "__main__":
    unittest.main(verbosity=2)
