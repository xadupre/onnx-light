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

    def test_transpose_same_byte_size_reported_as_greater(self):
        nodes = [
            oh.make_node("Abs", ["X"], ["A"]),
            oh.make_node("Transpose", ["A"], ["B"]),
            oh.make_node("Transpose", ["B"], ["Y"]),
        ]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        raw = si.compute_inplace_reuse(ctx, model.graph)
        reuse = self._reuse_pairs(raw)

        # Both transposes change layout but keep the same byte size.
        self.assertEqual(reuse, [[], [(0, 0)], [(0, 0)]])
        self.assertEqual(raw[1][0].kind, si.InPlaceReuseKind.kGreater)
        self.assertEqual(raw[2][0].kind, si.InPlaceReuseKind.kGreater)

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
            self._node_metadata(model.graph.node[1]),
            {"onnx_light.inplace_reuse": "0:0:equal", "onnx_light.release_after": "A"},
        )
        self.assertEqual(
            self._node_metadata(model.graph.node[2]),
            {"onnx_light.inplace_reuse": "0:0:equal", "onnx_light.release_after": "B"},
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
            self._node_metadata(model.graph.node[1]),
            {"onnx_light.inplace_reuse": "0:0:greater", "onnx_light.release_after": "A"},
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
            self._node_metadata(model.graph.node[1]),
            {"onnx_light.inplace_reuse": "0:0:equal", "onnx_light.release_after": "A"},
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

        inplace = si.ComputeContext()
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

        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx, allow_input_overwrite=True)
        self.assertEqual(self._reuse_pairs(inplace.reuse), [[(0, 0)], [(0, 0)]])

    def test_inplace_context_memory_tracks_sources_and_reuse(self):
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.INT32, [4])
        shape_init = oh.make_tensor("S", onnxl.TensorProto.INT64, [1], [4])
        graph = oh.make_graph(
            [
                oh.make_node("P", ["X", "S"], ["A"]),
                oh.make_node("Q", ["A"], ["B"]),
                oh.make_node("R", ["B"], ["Y"]),
            ],
            "g",
            [x],
            [y],
            [shape_init],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        ctx = si.ShapesContext()
        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, [4]))
        ctx.set("S", si.OptimTensor(onnxl.TensorProto.INT64, [1]))
        ctx.set("A", si.OptimTensor(onnxl.TensorProto.FLOAT, [4]))
        ctx.set("B", si.OptimTensor(onnxl.TensorProto.INT64, [4]))
        ctx.set("Y", si.OptimTensor(onnxl.TensorProto.INT32, [4]))

        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx, value_tags={"S": "shape"})

        self.assertEqual(self._reuse_pairs(inplace.reuse), [[], [], [(0, 0)]])
        self.assertEqual(len(inplace.memory), 3)

        mem0 = inplace.node_memory(0)
        self.assertEqual(
            list(mem0.keys()),
            [
                "already_allocated_bytes",
                "initializers",
                "inputs",
                "intermediates",
                "output_allocation_bytes",
                "outputs",
                "total_bytes",
            ],
        )
        self.assertEqual(mem0["total_bytes"], 40)
        self.assertEqual(mem0["inputs"], {"": 16})
        self.assertEqual(mem0["already_allocated_bytes"], 24)
        self.assertEqual(mem0["output_allocation_bytes"], 16)
        self.assertEqual(mem0["inputs"], {"": 16})
        self.assertEqual(mem0["initializers"], {"shape": 8})
        self.assertEqual(mem0["intermediates"], {})
        self.assertEqual(mem0["outputs"], {"": 16})

        mem1 = inplace.node_memory(1)
        self.assertEqual(mem1["total_bytes"], 72)
        self.assertEqual(mem1["already_allocated_bytes"], 40)
        self.assertEqual(mem1["output_allocation_bytes"], 32)
        self.assertEqual(mem1["inputs"], {"": 16})
        self.assertEqual(mem1["initializers"], {"shape": 8})
        self.assertEqual(mem1["intermediates"], {"": 16})
        self.assertEqual(mem1["outputs"], {"": 32})

        mem2 = inplace.node_memory(2)
        self.assertEqual(mem2["total_bytes"], 56)
        self.assertEqual(mem2["already_allocated_bytes"], 56)
        self.assertEqual(mem2["output_allocation_bytes"], 0)
        self.assertEqual(mem2["inputs"], {"": 16})
        self.assertEqual(mem2["initializers"], {"shape": 8})
        self.assertEqual(mem2["intermediates"], {"": 32})
        self.assertEqual(mem2["outputs"], {})

        with self.assertRaises(IndexError):
            inplace.node_memory(3)

    def test_inplace_context_memory_keeps_symbolic_shapes(self):
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N"])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.INT32, ["N"])
        shape_init = oh.make_tensor("S", onnxl.TensorProto.INT64, [1], [4])
        graph = oh.make_graph(
            [oh.make_node("P", ["X", "S"], ["A"]), oh.make_node("Q", ["A"], ["Y"])],
            "g",
            [x],
            [y],
            [shape_init],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        ctx = si.ShapesContext()
        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, ["N"]))
        ctx.set("S", si.OptimTensor(onnxl.TensorProto.INT64, [1]))
        ctx.set("A", si.OptimTensor(onnxl.TensorProto.FLOAT, ["N"]))
        ctx.set("Y", si.OptimTensor(onnxl.TensorProto.INT32, ["N"]))

        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx, value_tags={"S": "shape"})

        mem0 = inplace.node_memory(0)
        self.assertEqual(mem0["outputs"], {"": "4*N"})
        self.assertEqual(mem0["total_bytes"], "8*N+8")
        self.assertEqual(mem0["already_allocated_bytes"], "4*N+8")
        self.assertEqual(mem0["output_allocation_bytes"], "4*N")
        self.assertEqual(mem0["inputs"], {"": "4*N"})
        self.assertEqual(mem0["initializers"], {"shape": 8})
        self.assertEqual(mem0["intermediates"], {})
        self.assertEqual(mem0["outputs"], {"": "4*N"})

        mem1 = inplace.node_memory(1)
        self.assertEqual(mem1["total_bytes"], "12*N+8")
        self.assertEqual(mem1["already_allocated_bytes"], "8*N+8")
        self.assertEqual(mem1["output_allocation_bytes"], "4*N")
        self.assertEqual(mem1["inputs"], {"": "4*N"})
        self.assertEqual(mem1["initializers"], {"shape": 8})
        self.assertEqual(mem1["intermediates"], {"": "4*N"})
        self.assertEqual(mem1["outputs"], {"": "4*N"})

    def test_inplace_context_memory_simplifies_repeated_symbolic_sums(self):
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N"])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, ["N"])
        nodes = [oh.make_node("P", ["X"], [f"A{i}"]) for i in range(5)]
        nodes.append(oh.make_node("Q", [f"A{i}" for i in range(5)], ["Y"]))
        graph = oh.make_graph(nodes, "g", [x], [y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        ctx = si.ShapesContext()
        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, ["N"]))
        for i in range(5):
            ctx.set(f"A{i}", si.OptimTensor(onnxl.TensorProto.FLOAT, ["N"]))
        ctx.set("Y", si.OptimTensor(onnxl.TensorProto.FLOAT, ["N"]))

        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx)

        mem4 = inplace.node_memory(4)
        self.assertEqual(mem4["already_allocated_bytes"], "20*N")
        self.assertEqual(mem4["output_allocation_bytes"], "4*N")
        self.assertEqual(mem4["total_bytes"], "24*N")
        self.assertEqual(mem4["inputs"], {"": "4*N"})
        self.assertEqual(mem4["intermediates"], {"": "16*N"})
        self.assertEqual(mem4["outputs"], {"": "4*N"})

        mem5 = inplace.node_memory(5)
        self.assertEqual(mem5["already_allocated_bytes"], "24*N")
        self.assertEqual(mem5["output_allocation_bytes"], 0)
        self.assertEqual(mem5["total_bytes"], "24*N")
        self.assertEqual(mem5["inputs"], {"": "4*N"})
        self.assertEqual(mem5["intermediates"], {"": "20*N"})
        self.assertEqual(mem5["outputs"], {})

    def test_if_subgraph_local_shadowing_excluded_from_captures(self):
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        cond = oh.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, [])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])

        then_branch = oh.make_graph(
            [oh.make_node("Abs", ["X"], ["A"]), oh.make_node("Identity", ["A"], ["T"])],
            "then",
            [],
            [oh.make_tensor_value_info("T", onnxl.TensorProto.FLOAT, [3, 4])],
        )
        else_branch = oh.make_graph(
            [oh.make_node("Abs", ["X"], ["A"]), oh.make_node("Identity", ["A"], ["E"])],
            "else",
            [],
            [oh.make_tensor_value_info("E", onnxl.TensorProto.FLOAT, [3, 4])],
        )

        model = self._build_model(
            [
                oh.make_node("Abs", ["X"], ["A"]),
                oh.make_node(
                    "If", ["cond"], ["B"], then_branch=then_branch, else_branch=else_branch
                ),
                oh.make_node("Abs", ["B"], ["Y"]),
            ],
            [x, cond],
            [y],
        )

        ctx = si.ShapesContext()
        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, [3, 4]))
        ctx.set("A", si.OptimTensor(onnxl.TensorProto.FLOAT, [3, 4]))
        ctx.set("B", si.OptimTensor(onnxl.TensorProto.FLOAT, [3, 4]))
        ctx.set("Y", si.OptimTensor(onnxl.TensorProto.FLOAT, [3, 4]))

        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx)

        mem1 = inplace.node_memory(1)
        self.assertEqual(mem1["already_allocated_bytes"], 48)
        self.assertEqual(mem1["inputs"], {"": 48})
        self.assertEqual(mem1["intermediates"], {})
        self.assertEqual(mem1["output_allocation_bytes"], 48)
        self.assertEqual(mem1["outputs"], {"": 48})
        self.assertEqual(mem1["total_bytes"], 96)

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

        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx)
        inplace.write_to_metadata(model.graph)

        self.assertEqual(self._node_metadata(model.graph.node[0]), {})
        self.assertEqual(
            self._node_metadata(model.graph.node[1]),
            {"onnx_light.inplace_reuse": "0:0:equal", "onnx_light.release_after": "A"},
        )
        self.assertEqual(
            self._node_metadata(model.graph.node[2]),
            {"onnx_light.inplace_reuse": "0:0:equal", "onnx_light.release_after": "B"},
        )

    def test_shape_tag_release_info_populates_shape_tagged(self):
        # Shape(X)->S, Reshape(X, S)->Y: S is shape-tagged and released after Reshape.
        nodes = [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Reshape", ["X", "S"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        value_tags = {"S": "shape"}
        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx, value_tags=value_tags)

        self.assertEqual(len(inplace), 2)
        # S is released after the Reshape node (index 1).
        self.assertEqual(inplace.release_after_shape_tagged[0], [])
        self.assertEqual(inplace.release_after_shape_tagged[1], ["S"])
        # Per-node accessor mirrors the list.
        self.assertEqual(inplace.node_release_after_shape_tagged(1), ["S"])
        with self.assertRaises(IndexError):
            inplace.node_release_after_shape_tagged(2)

    def test_compute_value_and_node_tags_method(self):
        nodes = [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Reshape", ["X", "S"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        model = self._build_model(nodes, [x], [y])

        compute = si.ComputeContext()
        value_tags, node_tags = compute.compute_value_and_node_tags(model.graph)

        self.assertEqual(value_tags["S"], "shape")
        self.assertEqual(node_tags[0], "shape")
        self.assertEqual(len(node_tags), 2)
        self.assertEqual(compute.value_tags, value_tags)
        self.assertEqual(compute.node_tags, node_tags)
        self.assertEqual(compute.node_tag(0), "shape")
        self.assertEqual(compute.node_tag(1), node_tags[1])
        with self.assertRaises(IndexError):
            compute.node_tag(2)

    def test_infer_value_and_node_tags_binding_removed(self):
        self.assertFalse(hasattr(si._C, "infer_value_and_node_tags"))
        self.assertIs(si.infer_value_and_node_tags, si.compute_value_and_node_tags)

    def test_shape_tag_release_info_uses_stored_tags(self):
        nodes = [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Reshape", ["X", "S"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        inplace = si.ComputeContext()
        inplace.compute_value_and_node_tags(model.graph)
        inplace.compute_inplace_reuse_graph(model.graph, ctx)

        self.assertEqual(inplace.release_after_shape_tagged[0], [])
        self.assertEqual(inplace.release_after_shape_tagged[1], ["S"])

    def test_shape_tag_release_info_writes_metadata(self):
        nodes = [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Reshape", ["X", "S"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        value_tags = {"S": "shape"}
        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx, value_tags=value_tags)
        inplace.write_to_metadata(model.graph)

        # Node 0 (Shape) has no release metadata.
        self.assertEqual(self._node_metadata(model.graph.node[0]), {})
        meta1 = self._node_metadata(model.graph.node[1])
        # kReleaseAfterMetadataKey lists S.
        self.assertEqual(meta1.get("onnx_light.release_after"), "S")
        # kReleaseAfterShapeTagMetadataKey also lists S (it is shape-tagged).
        self.assertEqual(meta1.get("onnx_light.release_after_shape_tag"), "S")

    def test_shape_tag_no_value_tags_leaves_shape_tagged_empty(self):
        nodes = [oh.make_node("Abs", ["X"], ["A"]), oh.make_node("Abs", ["A"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx)
        inplace.write_to_metadata(model.graph)

        # Without value_tags the shape-tagged vector is itself empty.
        self.assertEqual(inplace.release_after_shape_tagged, [])
        # kReleaseAfterShapeTagMetadataKey must not appear.
        for node in model.graph.node:
            self.assertNotIn("onnx_light.release_after_shape_tag", self._node_metadata(node))

    def test_write_inplace_reuse_to_metadata_with_value_tags(self):
        nodes = [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Reshape", ["X", "S"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        value_tags = {"S": "shape"}
        si.write_inplace_reuse_to_metadata(ctx, model.graph, value_tags=value_tags)

        meta1 = self._node_metadata(model.graph.node[1])
        self.assertEqual(meta1.get("onnx_light.release_after"), "S")
        self.assertEqual(meta1.get("onnx_light.release_after_shape_tag"), "S")

    def test_shape_tag_clear_resets_shape_tagged(self):
        nodes = [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Reshape", ["X", "S"], ["Y"])]
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        model = self._build_model(nodes, [x], [y])

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)

        inplace = si.ComputeContext()
        inplace.compute_inplace_reuse_graph(model.graph, ctx, value_tags={"S": "shape"})
        self.assertEqual(len(inplace), 2)
        inplace.clear()
        self.assertEqual(len(inplace), 0)
        self.assertEqual(inplace.release_after_shape_tagged, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
