import unittest

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl
import onnx_light.onnx.checker as checker

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
make_test_class = import_or_skip("onnx_light.onnx_lib.backend.test.case", "make_test_class")


def check_model(model: onnxl.ModelProto, *inputs):
    checker.check_model(model)
    # Map-typed graph inputs are expanded to a (keys, values) tensor pair by
    # the backend harness, so count them as 2 rather than 1.
    expected = sum(2 if gi.type.has_map_type() else 1 for gi in model.graph.input)
    assert len(inputs) == expected
    return None


TestCheckerBackend = make_test_class(check_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
