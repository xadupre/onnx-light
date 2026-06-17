import unittest

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl
import onnx_light.onnx_optim.shape_inference as shape_inference

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
make_test_class = import_or_skip("onnx_light.onnx_lib.backend.test.case", "make_test_class")


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
    for o in work.graph.output:
        o.type.Clear()
    mapping = {}
    for i in work.graph.input:
        if i.name in {"axis", "axes"}:
            continue
        shape = i.type.tensor_type.shape
        for di, d in enumerate(shape.dim):
            if d.dim_value not in mapping:
                assert not d.dim_param, f"Unexpected input dimension in {i}"
                mapping[int(d.dim_value)] = f"{i.name}_{di}"
    for i in work.graph.input:
        if i.name in {"axis", "axes"}:
            continue
        shape = i.type.tensor_type.shape
        new_shape = [mapping[d.dim_value] for d in shape.dim]
        for i in range(len(shape.dim)):
            shape.dim[i].Clear()
            shape.dim[i].dim_param = new_shape[i]

    shape_inference.infer_shapes_model(work)
    _check_match(model.graph.input, model.graph.value_info, work.graph.value_info)


TestOptimShapeInferenceDynamicBackend = make_test_class(
    shape_inference_check,
    exclude_regex=[
        "attention_3d",
        "cast_map_int64_float_dense",
        "cast_map_int64_float_sparse",
        "cast_map_int64_string_dense",
        "dict_vectorizer_int64_float",
        "dict_vectorizer_string_int64",
        "identity_opt",
        "identity_sequence",
        "if_opt",
        "if_seq",
        "loop11_carried_state",
        "loop13_seq",
        "loop16_seq_none",
        "optional_get_element_optional_sequence",
        "sequence_map_add_2_sequences",
        "sequence_map_identity_2_sequences",
        "shape_inference_add_concat_reshape",
        "shape_inference_check_shape",
        "shape_inference_concat_split_even",
        "shape_inference_concat_split_odd",
        "shape_inference_local_function_add",
        "shape_inference_loop_pairwise_distance",
        "shape_inference_nested_local_function_add",
        "shape_inference_nonzero_chain_named",
        "shape_inference_nonzero_plus_expression",
        "shape_inference_pad_canny_average",
        "shape_inference_reshape_reshape",
        "shape_inference_resize_tile",
        "shape_inference_scan_running_sum",
        "shape_inference_value_as_shape",
        "scan_sum",
    ],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
