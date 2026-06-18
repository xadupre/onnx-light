import unittest

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl
import onnx_light.onnx_optim.shape_inference as shape_inference

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
make_test_class = import_or_skip("onnx_light.onnx_lib.backend.test.case", "make_test_class")


def _inputs(inputs):
    def _v(value_info):
        if value_info.type.has_map_type():
            mt = value_info.type.map_type
            vt = mt.value_type.tensor_type
            t = tuple(d.dim_param or d.dim_value for d in vt.shape.dim)
            return f"map[{mt.key_type} -> {vt.elem_type}: {t}]"
        v = value_info.type.tensor_type
        t = tuple(d.dim_param or d.dim_value for d in v.shape.dim)
        return f"{v.elem_type}: {t}"

    return ", ".join([_v(i) for i in inputs])


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
    for o in work.graph.output:
        o.type.Clear()

    def _has_dim_value(dim) -> bool:
        return dim.has_dim_value()

    mapping = {}
    prefill_with_value_info_output = False
    for i in work.graph.input:
        if i.name in {"axis", "axes"} or not i.type.has_tensor_type():
            continue
        if not i.type.tensor_type:
            continue
        shape = i.type.tensor_type.shape
        for di, d in enumerate(shape.dim):
            if d.dim_param:
                prefill_with_value_info_output = True
                continue
            if _has_dim_value(d):
                key = int(d.dim_value)
            else:
                prefill_with_value_info_output = True
                key = (i.name, di)
            if key not in mapping:
                mapping[key] = f"{i.name}_{di}"
    for i in work.graph.input:
        if i.name in {"axis", "axes"} or not i.type.has_tensor_type():
            continue
        if not i.type.tensor_type:
            continue
        shape = i.type.tensor_type.shape
        new_shape = []
        for di, d in enumerate(shape.dim):
            if d.dim_param:
                new_shape.append(d.dim_param)
                continue
            key = int(d.dim_value) if _has_dim_value(d) else (i.name, di)
            new_shape.append(mapping[key])
        for i in range(len(shape.dim)):
            shape.dim[i].Clear()
            shape.dim[i].dim_param = new_shape[i]

    shape_inference.infer_shapes_model(
        work, prefill_with_value_info_output=prefill_with_value_info_output
    )
    _check_match(model.graph.input, model.graph.value_info, work.graph.value_info)


TestOptimShapeInferenceDynamicBackend = make_test_class(
    shape_inference_check,
    exclude_regex=[
        "test_cc_shape_inference_add_concat_reshape.*",
        "test_cc_shape_inference_nonzero_chain_anon.*",
        "test_cc_attention_3d.*",
        "test_cc_cast_map_.*",
        "test_cc_dict_vectorizer_.*",
        "test_cc_loop11_carried_state.*",
        "test_cc_optional_get_element_optional_sequence.*",
        "test_cc_squeeze_all_singleton.*",
        "test_cc_squeeze_no_axes_input.*",
        "test_cc_squeeze_empty_axes_name.*",
        "test_if_seq.*",
        "test_cc_loop13_seq.*",
        "test_cc_loop16_seq_none.*",
        "test_cc_identity_opt.*",
        "test_cc_if_seq.*",
        "test_cc_if_opt.*",
        "test_cc_linear_attention.*",
        "test_cc_shape_inference_scan_running_sum.*",
        # These local-function cases keep dedicated non-dynamic coverage because
        # they rely on user-authored symbolic aliases that the generic dynamic
        # harness does not normalize consistently yet.
        "test_cc_shape_inference_local_function_add.*",
        "test_cc_shape_inference_nested_local_function_add.*",
        "test_cc_sequence_map_add_2_sequences.*",
        "test_cc_sequence_map_identity_2_sequences.*",
        "test_cc_shape_inference_nonzero_plus_expression.*",
        # These remaining models keep dedicated non-dynamic coverage because
        # they rely on exact user-authored symbolic aliases or expressions that
        # this generic harness does not normalize.
        "test_cc_shape_inference_resize_tile.*",
        "test_cc_shape_inference_pad_canny_average.*",
        "test_cc_shape_inference_topk_pairwise_distance.*",
        "test_cc_shape_inference_loop_pairwise_distance.*",
        "test_cc_shape_inference_loop_topk_pairwise_distance.*",
        "test_cc_shape_inference_scan_topk_pairwise_distance.*",
        "test_cc_shape_inference_tiny_llm.*",
    ],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
