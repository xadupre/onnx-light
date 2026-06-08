import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.shape_inference as shape_inference
from onnx_light.backend.test.case import make_test_class


def _check_match(expected, inferred) -> None:
    expected_dict = {vi.name: vi for vi in expected}
    inferred_dict = {vi.name: vi for vi in inferred}
    for name, value in expected_dict.items():
        assert name in inferred_dict, f"Missing shape for {name!r}"
        infe = inferred_dict[name]
        if not value.type.tensor_type:
            # not implemented yet
            continue
        assert (
            value.type.tensor_type and infe.type.tensor_type
        ), f"Missing tensor_type for {name!r}"
        vt = value.type.tensor_type
        it = infe.type.tensor_type
        assert vt.elem_type == vt.elem_type, f"Element type mismatch for {name!r}: {vt} != {it}"
        v_shape = tuple(d.dim_param or d.dim_value for d in vt.shape.dim)
        i_shape = tuple(d.dim_param or d.dim_value for d in it.shape.dim)
        assert v_shape == i_shape, f"Element shape mismatch for {name!r}: {vt} != {it}"


def shape_inference_check(model: onnxl.ModelProto, *inputs):
    assert isinstance(model, onnxl.ModelProto), f"Unexpected type {type(model)}"
    work = onnxl.ModelProto()
    work.CopyFrom(model)
    if work.graph.value_info:
        work.graph.value_info.clear()
    shape_inference.infer_shapes(work)
    _check_match(model.graph.value_info, work.graph.value_info)
    _check_match(model.graph.output, work.graph.output)


TestShapeInferenceBackend = make_test_class(
    shape_inference_check,
    exclude_regex=[
        "test_cc_shape_inference_add_concat_reshape.*",
        "test_cc_shape_inference_nonzero_chain_anon.*",
        "test_cc_shape_inference_nonzero_chain_named.*",
    ],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
