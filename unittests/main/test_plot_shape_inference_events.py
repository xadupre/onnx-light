import ast
import pathlib
import unittest

NONZERO_CHAIN_TEST_CASE_NAME = "test_cc_shape_inference_nonzero_chain_named"


def _example_has_nonzero_chain_event_logging() -> bool:
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "compute" / "plot_shape_inference.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    has_case_name = False
    has_events_enabled = False
    has_compute_shape_model_call = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == NONZERO_CHAIN_TEST_CASE_NAME:
            has_case_name = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "events_enabled":
                    if isinstance(node.value, ast.Constant) and node.value.value is True:
                        has_events_enabled = True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "compute_shape_model":
                has_compute_shape_model_call = True
            elif isinstance(func, ast.Attribute) and func.attr == "compute_shape_model":
                has_compute_shape_model_call = True

    return has_case_name and has_events_enabled and has_compute_shape_model_call


class TestPlotShapeInferenceEvents(unittest.TestCase):
    def test_nonzero_events_section_present(self):
        self.assertTrue(_example_has_nonzero_chain_event_logging())


if __name__ == "__main__":
    unittest.main(verbosity=2)
