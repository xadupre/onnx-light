import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.checker as checker
from onnx_light.backend.test.case import make_test_class


def check_model(model: onnxl.ModelProto, *inputs):
    checker.check_model(model)
    assert len(inputs) == len(model.graph.input)
    return None


TestCheckerBackend = make_test_class(check_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
