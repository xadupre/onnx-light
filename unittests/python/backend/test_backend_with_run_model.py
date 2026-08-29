# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Backend tests that exercise the runtime's model execution path
(``ExecutionPlan`` + ``RuntimeSession``, exposed through
``onnx_light.onnx_py._onnxpykernels.runtime``) against every backend test
case whose top-level graph contains a single node of an op registered in
``KernelDispatchTable``.

This is the Python counterpart of
``unittests/cc/onnx_extensions/backend_test/test_backend_run_model.cc``: both walk the
same C++-generated backend test registry and validate that running the model
reproduces the expected outputs without discrepancies.
"""

from __future__ import annotations

import unittest

import numpy as np

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl
import onnx_light.onnx.numpy_helper as onh

# The kernels runtime and backend test registries are only available in the
# full build; skip this module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
_backend_case = import_or_skip("onnx_light.onnx_lib.backend.test.case")
make_test_class = _backend_case.make_test_class
collect_test_case = _backend_case.collect_test_case
rt = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


def _default_opset_version(model: onnxl.ModelProto) -> int:
    """Returns the version of the default ai.onnx opset for ``model``."""
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return 18


def run_model_backend(model: onnxl.ModelProto, *inputs: np.ndarray) -> list[np.ndarray]:
    """Executes ``model`` by registering its local functions and driving
    ``model.graph``'s :class:`ExecutionPlan` through a :class:`RuntimeSession`.

    Mirrors the signature expected by :func:`make_test_class` (the same as
    :func:`onnxruntime_backend` in ``test_backend_with_onnxruntime.py``):
    ``rt(model, *inputs)`` and returns the model's outputs as numpy arrays
    in graph-output order.
    """
    ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(_default_opset_version(model))))

    # Wire positional ``inputs`` to the model's declared graph inputs by name.
    graph_inputs = [vi.name for vi in model.graph.input]
    if len(inputs) != len(graph_inputs):
        raise ValueError(
            f"Expected {len(graph_inputs)} positional inputs for graph inputs "
            f"{graph_inputs}, got {len(inputs)}."
        )
    for name, arr in zip(graph_inputs, inputs):
        tp = onh.from_array(np.ascontiguousarray(arr), name=name)
        ctx.set(name, rt.tensor_from_proto(tp))

    rt.register_model_functions(model, ctx)
    for init in model.graph.initializer:
        if not ctx.has(init.name):
            ctx.set(init.name, rt.tensor_from_proto(init), "initializer")
    plan = rt.ExecutionPlan(model.graph)
    rt.RuntimeSession(plan).run(ctx)

    outputs: list[np.ndarray] = []
    for vi in model.graph.output:
        t = ctx.get(vi.name)
        dtype_map = {
            int(onnxl.TensorProto.FLOAT): np.float32,
            int(onnxl.TensorProto.DOUBLE): np.float64,
            int(onnxl.TensorProto.INT32): np.int32,
            int(onnxl.TensorProto.INT64): np.int64,
            int(onnxl.TensorProto.UINT8): np.uint8,
            int(onnxl.TensorProto.INT8): np.int8,
            int(onnxl.TensorProto.UINT16): np.uint16,
            int(onnxl.TensorProto.INT16): np.int16,
            int(onnxl.TensorProto.UINT32): np.uint32,
            int(onnxl.TensorProto.UINT64): np.uint64,
            int(onnxl.TensorProto.BOOL): np.bool_,
            int(onnxl.TensorProto.FLOAT16): np.float16,
        }
        dtype = dtype_map.get(int(t.data_type))
        if dtype is None:
            raise NotImplementedError(
                f"run_model_backend cannot convert C++ Tensor data_type={t.data_type} to numpy."
            )
        arr = np.frombuffer(t.raw_data(), dtype=dtype)
        outputs.append(arr.reshape(tuple(int(d) for d in t.shape)))
    return outputs


def _single_node_op_type(tc) -> str | None:
    """Returns the ``op_type`` of the top-level node when ``tc.model.graph``
    contains exactly one node, otherwise ``None``.
    """
    if tc.model is None:
        return None
    nodes = list(tc.model.graph.node)
    if len(nodes) != 1:
        return None
    return nodes[0].op_type


TestRunModelBackend = make_test_class(
    run_model_backend,
    exclude_regex=[
        ".*FLOAT4.*",
        ".*FLOAT8.*",
        ".*float4.*",
        ".*float8.*",
        ".*bfloat16.*",
        "BFLOAT16.*",
        ".*STRING.*",
        ".*int4.*",
        ".*int2.*",
        ".*INT4.*",
        ".*INT2.*",
        ".*e4m3.*",
        ".*e5m2.*",
        ".*sequence_map_identity.*",
        ".*string_normalizer.*",
        ".*strnormalizer.*",
        ".*string_concat.*",
        ".*string_split.*",
        ".*split_to_sequence.*",
        #
        "blackmanwindow",
        "cast_BOOL_to_STRING",
        "cast_map_int64_float_dense",
        "cast_map_int64_float_sparse",
        "cast_map_int64_string_dense",
        "category_mapper_int_to_string",
        "constant_value_string",
        "dict_vectorizer_int64_float",
        "dict_vectorizer_string_int64",
        "einsum_scalar",
        "hammingwindow",
        "hannwindow",
        "identity_op",
        "identity_sequence",
        "if_seq",
        "if_opt",
        "image_decoder_decode_jpeg_bgr",
        "image_decoder_decode_jpeg_grayscale",
        "image_decoder_decode_jpeg_rgb",
        "loop16_seq_none",
        "melweightmatrix",
        "optional_get_element_optional_sequence",
        "optional_get_element_sequence",
        "scan9_input_reverse",
        "scan9_output_reverse",
        "scan9_scalar",
        "sequence_at_neg",
        "sequence_at_pos0",
        "sequence_at_pos2",
        "sequence_construct",
        "sequence_erase_default",
        "sequence_erase_neg",
        "sequence_erase_pos1",
        "sequence_insert_at_back",
        "sequence_insert_at_front",
        "sequence_insert_default",
        "sequence_insert_neg",
        "sequence_insert_pos1",
        "sequence_map_add_1_sequence_1_tensor",
        "sequence_map_add_2_sequences",
        "sequence_map_extract_shapes",
        # run_model_backend cannot convert STRING tensors to numpy; the
        # ``.*STRING.*`` regex above is uppercase-only and misses this case.
        "test_cc_where_string",
        "test_cc_label_encoder_int64_to_string",
        "test_cc_label_encoder_string_to_string_tensor_attributes",
        # The loop pairwise-distance model uses Manhattan distance (L1) but
        # the expected outputs are Euclidean (L2): numerical mismatch.
        "test_cc_shape_inference_loop_pairwise_distance.*",
        # TopK k input exceeds the axis length for the scan/loop topk variants.
        "test_cc_shape_inference_loop_topk_pairwise_distance.*",
        "test_cc_shape_inference_scan_topk_pairwise_distance.*",
    ],
)


class TestBackendTestOracle(unittest.TestCase):
    def test_expected_outputs_ignore_global_kernel_override(self):
        rt.register_custom_kernel("", "Abs", lambda node, ctx: None)
        try:
            case = _backend_case.get_test_case("test_cc_abs")
        finally:
            rt.unregister_custom_kernel("", "Abs")

        if case is None:
            self.fail("Missing test_cc_abs backend test case.")
        self.assertEqual(case.expected_output_oracle, "onnx-light::ai.onnx::Abs")
        np.testing.assert_array_equal(
            case.data_sets[0][1][0],
            np.array([[1.0, 0.0, 1.5], [2.25, 3.5, 4.75]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
