import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_optim import shape_inference as si


class TestInPlaceReuse(ExtTestCase):
    """Python tests for the ``compute_inplace_reuse`` binding exposed by
    ``onnx_light.onnx_py._onnxpyoptim.shape_inference``."""

    @staticmethod
    def _reuse_pairs(reuse):
        """Normalises the nested ``InPlaceReuse`` result into a list of
        lists of ``(output_index, input_index)`` tuples."""
        return [[(r.output_index, r.input_index) for r in node] for node in reuse]

    def test_inplace_reuse_struct(self):
        r = si.InPlaceReuse()
        r.output_index = 1
        r.input_index = 2
        self.assertEqual(r.output_index, 1)
        self.assertEqual(r.input_index, 2)
        # The default kind is kEqual (same-sized buffer).
        self.assertEqual(r.kind, si.InPlaceReuseKind.kEqual)
        self.assertEqual(repr(r), "InPlaceReuse(output_index=1, input_index=2, kind=kEqual)")
        r.kind = si.InPlaceReuseKind.kGreater
        self.assertEqual(r.kind, si.InPlaceReuseKind.kGreater)
        self.assertEqual(repr(r), "InPlaceReuse(output_index=1, input_index=2, kind=kGreater)")

    def _build_model(self, nodes, inputs, outputs):
        graph = oh.make_graph(nodes, "g", inputs, outputs)
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8
        return model

    def test_abs_chain_reuses_intermediates(self):
        nodes = [
            oh.make_node("Abs", ["X"], ["A"]),
            oh.make_node("Abs", ["A"], ["B"]),
            oh.make_node("Abs", ["B"], ["Y"]),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        reuse = self._reuse_pairs(si.compute_inplace_reuse(ctx, model.graph))

        # Node 0 reads the declared graph input X (must not be overwritten).
        self.assertEqual(reuse, [[], [(0, 0)], [(0, 0)]])
        # The reused buffers are the same size as the outputs.
        raw = si.compute_inplace_reuse(ctx, model.graph)
        self.assertEqual(raw[1][0].kind, si.InPlaceReuseKind.kEqual)
        self.assertEqual(raw[2][0].kind, si.InPlaceReuseKind.kEqual)

    def test_larger_input_buffer_reported_as_greater(self):
        # A is INT64 (8 bytes/elem); Y is INT32 (4 bytes/elem). A's buffer is
        # strictly larger than Y, so Y may still reuse it in place.
        nodes = [
            oh.make_node("Cast", ["X"], ["A"], to=onnxl.TensorProto.INT64),
            oh.make_node("Cast", ["A"], ["Y"], to=onnxl.TensorProto.INT32),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.INT32, [4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        raw = si.compute_inplace_reuse(ctx, model.graph)
        reuse = self._reuse_pairs(raw)

        self.assertEqual(reuse, [[], [(0, 0)]])
        self.assertEqual(raw[1][0].kind, si.InPlaceReuseKind.kGreater)

    def test_value_read_twice_reused_only_at_last_use(self):
        nodes = [
            oh.make_node("Abs", ["X"], ["A"]),
            oh.make_node("Abs", ["A"], ["B"]),
            oh.make_node("Add", ["A", "B"], ["Y"]),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        reuse = self._reuse_pairs(si.compute_inplace_reuse(ctx, model.graph))

        # A is alive at node 2, so node 1 cannot reuse it; node 2 reuses A.
        self.assertEqual(reuse, [[], [], [(0, 0)]])

    def test_shape_mismatch_yields_no_reuse(self):
        nodes = [
            oh.make_node("Abs", ["X"], ["A"]),
            oh.make_node("Transpose", ["A"], ["B"]),
            oh.make_node("Transpose", ["B"], ["Y"]),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [4, 3])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        reuse = self._reuse_pairs(si.compute_inplace_reuse(ctx, model.graph))

        # Both transposes change the shape from [3,4] to [4,3] (and back).
        self.assertEqual(reuse, [[], [], []])

    def test_graph_output_input_is_not_reused(self):
        nodes = [oh.make_node("Abs", ["X"], ["A"]), oh.make_node("Abs", ["A"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 2])
        a = oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, [2, 2])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 2])
        model = self._build_model(nodes, [x], [a, y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        reuse = self._reuse_pairs(si.compute_inplace_reuse(ctx, model.graph))

        # A is a declared graph output; node 1 must not overwrite it.
        self.assertEqual(reuse, [[], []])

    @staticmethod
    def _node_metadata(node):
        return {m.key: m.value for m in node.metadata_props}

    def test_write_inplace_reuse_to_metadata(self):
        nodes = [
            oh.make_node("Abs", ["X"], ["A"]),
            oh.make_node("Abs", ["A"], ["B"]),
            oh.make_node("Abs", ["B"], ["Y"]),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        si.write_inplace_reuse_to_metadata(ctx, model.graph)

        # Node 0 reads the declared graph input X, so it has no reuse and no
        # metadata is written for it.
        self.assertEqual(self._node_metadata(model.graph.node[0]), {})
        self.assertEqual(
            self._node_metadata(model.graph.node[1]), {"onnx_light.inplace_reuse": "0:0:equal"}
        )
        self.assertEqual(
            self._node_metadata(model.graph.node[2]), {"onnx_light.inplace_reuse": "0:0:equal"}
        )

    def test_write_inplace_reuse_to_metadata_greater(self):
        nodes = [
            oh.make_node("Cast", ["X"], ["A"], to=onnxl.TensorProto.INT64),
            oh.make_node("Cast", ["A"], ["Y"], to=onnxl.TensorProto.INT32),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.INT32, [4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        si.write_inplace_reuse_to_metadata(ctx, model.graph)

        self.assertEqual(self._node_metadata(model.graph.node[0]), {})
        self.assertEqual(
            self._node_metadata(model.graph.node[1]), {"onnx_light.inplace_reuse": "0:0:greater"}
        )

    def test_write_inplace_reuse_to_metadata_updates_existing_key(self):
        nodes = [oh.make_node("Abs", ["X"], ["A"]), oh.make_node("Abs", ["A"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])
        # Pre-existing entry under the same key should be replaced in place.
        model.graph.node[1].add_metadata("onnx_light.inplace_reuse", "stale")

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        si.write_inplace_reuse_to_metadata(ctx, model.graph)

        self.assertEqual(
            self._node_metadata(model.graph.node[1]), {"onnx_light.inplace_reuse": "0:0:equal"}
        )

    def test_allow_input_overwrite_reuses_graph_input(self):
        nodes = [oh.make_node("Abs", ["X"], ["A"]), oh.make_node("Abs", ["A"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        # By default the graph input X must not be overwritten.
        reuse = self._reuse_pairs(si.compute_inplace_reuse(ctx, model.graph))
        self.assertEqual(reuse, [[], [(0, 0)]])

        # Opt-in: node 0 is the only use of X, so it may reuse it in place.
        reuse_ovw = self._reuse_pairs(
            si.compute_inplace_reuse(ctx, model.graph, allow_input_overwrite=True)
        )
        self.assertEqual(reuse_ovw, [[(0, 0)], [(0, 0)]])

    def test_allow_input_overwrite_keeps_input_that_is_also_output(self):
        nodes = [oh.make_node("Abs", ["X"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 2])
        x_out = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 2])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 2])
        model = self._build_model(nodes, [x], [x_out, y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        # X is also a declared output, so it stays protected even with the option.
        reuse = self._reuse_pairs(
            si.compute_inplace_reuse(ctx, model.graph, allow_input_overwrite=True)
        )
        self.assertEqual(reuse, [[]])

    def test_inplace_context_stores_result(self):
        nodes = [
            oh.make_node("Abs", ["X"], ["A"]),
            oh.make_node("Abs", ["A"], ["B"]),
            oh.make_node("Abs", ["B"], ["Y"]),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        inplace = si.InplaceContext()
        self.assertEqual(len(inplace), 0)
        inplace.compute_inplace_reuse_graph(model.graph, ctx)

        self.assertEqual(len(inplace), 3)
        # The stored result matches the free-function wrapper.
        self.assertEqual(self._reuse_pairs(inplace.reuse), [[], [(0, 0)], [(0, 0)]])
        self.assertEqual(self._reuse_pairs([inplace.node_reuse(1)]), [[(0, 0)]])
        self.assertEqual(inplace.node_reuse(1)[0].kind, si.InPlaceReuseKind.kEqual)
        # node_reuse rejects an out-of-bounds index.
        with self.assertRaises(IndexError):
            inplace.node_reuse(3)
        # clear empties the stored result.
        inplace.clear()
        self.assertEqual(len(inplace), 0)

    def test_inplace_context_allow_input_overwrite(self):
        nodes = [oh.make_node("Abs", ["X"], ["A"]), oh.make_node("Abs", ["A"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        inplace = si.InplaceContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx, allow_input_overwrite=True)
        self.assertEqual(self._reuse_pairs(inplace.reuse), [[(0, 0)], [(0, 0)]])

    def test_inplace_context_write_to_metadata(self):
        nodes = [
            oh.make_node("Abs", ["X"], ["A"]),
            oh.make_node("Abs", ["A"], ["B"]),
            oh.make_node("Abs", ["B"], ["Y"]),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        inplace = si.InplaceContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx)
        inplace.write_to_metadata(model.graph)

        self.assertEqual(self._node_metadata(model.graph.node[0]), {})
        self.assertEqual(
            self._node_metadata(model.graph.node[1]), {"onnx_light.inplace_reuse": "0:0:equal"}
        )
        self.assertEqual(
            self._node_metadata(model.graph.node[2]), {"onnx_light.inplace_reuse": "0:0:equal"}
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
