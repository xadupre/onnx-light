import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx_optim.shape_inference as shape_inference
from onnx_light.backend.test.case import make_test_class


def _inputs(inputs):
    def _v(v):
        t = tuple(d.dim_param or d.dim_value for d in v.shape.dim)
        return f"{v.elem_type}: {t}"

    return ", ".join([_v(i.type.tensor_type) for i in inputs])


def _check_match(inputs, expected, inferred, rank_only=False) -> None:
    expected_dict = {vi.name: vi for vi in expected}
    inferred_dict = {vi.name: vi for vi in inferred}
    for name, value in expected_dict.items():
        assert name in inferred_dict, f"Missing shape for {name!r}, inputs: {_inputs(inputs)}"
        infe = inferred_dict[name]
        if not value.type.tensor_type:
            # not implemented yet
            continue
        assert (
            value.type.tensor_type and infe.type.tensor_type
        ), f"Missing tensor_type for {name!r}, inputs: {_inputs(inputs)}"
        vt = value.type.tensor_type
        it = infe.type.tensor_type
        assert (
            vt.elem_type == vt.elem_type
        ), f"Element type mismatch for {name!r}: {vt} != {it}, inputs: {_inputs(inputs)}"
        v_shape = tuple(d.dim_param or d.dim_value for d in vt.shape.dim)
        i_shape = tuple(d.dim_param or d.dim_value for d in it.shape.dim)
        if rank_only:
            assert len(v_shape) == len(
                i_shape
            ), f"Element shape mismatch for {name!r}: {vt} != {it}, inputs: {_inputs(inputs)}"
            continue
        assert (
            v_shape == i_shape
        ), f"Element shape mismatch for {name!r}: {vt} != {it}, inputs: {_inputs(inputs)}"


def shape_inference_check(model: onnxl.ModelProto, *inputs):
    assert isinstance(model, onnxl.ModelProto), f"Unexpected type {type(model)}"
    work = onnxl.ModelProto()
    work.CopyFrom(model)
    if work.graph.value_info:
        work.graph.value_info.clear()
    shape_inference.infer_shapes_model(work)
    _check_match(model.graph.input, model.graph.value_info, work.graph.value_info)
    _check_match(model.graph.input, model.graph.output, work.graph.output, rank_only=True)


TestOptimShapeInferenceBackend = make_test_class(
    shape_inference_check,
    exclude_regex=[
        "test_cc_shape_inference_add_concat_reshape.*",
        "test_cc_shape_inference_nonzero_chain_anon.*",
        "test_cc_shape_inference_nonzero_chain_named.*",
        "test_cc_attention_3d.*",
        "test_cc_cast_map_.*",
        "test_cc_dict_vectorizer_.*",
        "test_cc_loop11_carried_state.*",
        "test_cc_optional_get_element_optional_sequence.*",
        "test_cc_sequence_map_add_2_sequences.*",
        "test_cc_sequence_map_identity_2_sequences.*",
        "test_cc_squeeze_all_singleton.*",
        "test_if_seq.*",
        "test_scan_sum.*",
        "test_cc_loop13_seq.*",
    ],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
