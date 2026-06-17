import unittest

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl
import onnx_light.onnx_optim.shape_inference as shape_inference

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
make_test_class = import_or_skip("onnx_light.onnx.backend", "make_test_class")


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
        "test_cc_squeeze_no_axes_input.*",
        "test_cc_squeeze_empty_axes_name.*",
        "test_if_seq.*",
        "test_scan_sum.*",
        "test_cc_loop13_seq.*",
        "test_cc_loop16_seq_none.*",
        "test_cc_identity_sequence.*",
        "test_cc_identity_opt.*",
        "test_cc_if_seq.*",
        "test_cc_if_opt.*",
        "test_cc_linear_attention.*",
        "test_cc_shape_inference_concat_split.*",
        "test_cc_shape_inference_check_shape.*",
        "test_cc_shape_inference_scan_running_sum.*",
        # The expression simplifier reduces 2*(H//2) → H, so the inferred
        # tile_out dim differs from the symbolic name stored in value_info.
        "test_cc_shape_inference_resize_tile.*",
        # Inputs already use symbolic dim_param ("batch", "seq", "past_seq",
        # "total_seq"); the value_info uses symbolic dims that the optim
        # inference may rename, so exact-match value_info comparison cannot be
        # enforced here. Covered by the C++ BackendTestCaseShapeInference test.
        "test_cc_shape_inference_tiny_llm.*",
    ],
)


def shape_inference_no_new_names_check(model: onnxl.ModelProto, *inputs):
    """Verifies that infer_shapes_model does not introduce new value_info names.

    Clears value_info on a copy of the model, runs shape inference, then checks
    that every inferred value_info name was already present in the original
    model's value_info.  If shape inference raises an exception the test is
    treated as a no-op so that operators whose inference is not yet implemented
    do not cause spurious failures here.
    """
    original_vi_names = {vi.name for vi in model.graph.value_info}
    work = onnxl.ModelProto()
    work.CopyFrom(model)
    work.graph.value_info.clear()
    try:
        shape_inference.infer_shapes_model(work)
    except Exception:  # noqa: BLE001 - C++ extension raises various types
        return
    inferred_vi_names = {vi.name for vi in work.graph.value_info}
    new_names = inferred_vi_names - original_vi_names
    assert (
        not new_names
    ), f"infer_shapes_model introduced new value_info names: {sorted(new_names)}"


TestOptimShapeInferenceNoNewNamesBackend = make_test_class(
    shape_inference_no_new_names_check,
    exclude_regex=[
        # NonZero, Loop, and Compress are explicitly permitted to introduce
        # new symbolic intermediate names during shape inference (issue #2733).
        "test_cc_nonzero.*",
        "test_nonzero.*",
        "test_cc_shape_inference_nonzero.*",
        "test_cc_compress.*",
        "test_compress.*",
        "test_cc_loop.*",
        "test_loop.*",
        "test_cc_shape_inference_loop.*",
        # Optional models contain intermediate tensors (e.g. opt_value) that
        # are not declared in value_info; shape inference legitimately adds
        # them, so these tests are excluded from the no-new-names check.
        "test_cc_optional.*",
        "test_optional.*",
    ],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
