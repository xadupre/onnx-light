import tempfile
import unittest
from pathlib import Path

import onnx_light.onnx as onnx
import onnx_light.onnx.helper as oh
from onnx_light.onnx.checker import check_model


class TestGraphLayout(unittest.TestCase):
    @staticmethod
    def model():
        return oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["X"], ["A"]), oh.make_node("Relu", ["A"], ["Y"])],
                "layout",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )

    def test_opt_in_and_round_trip(self):
        model = self.model()
        original = model.SerializeToString()
        self.assertEqual(onnx.load(original).SerializeToString(), original)
        loaded = onnx.load(original, layout=True)
        placement = onnx.GraphLayout(loaded.graph)
        placement.load()
        self.assertEqual(len(placement.order), 2)
        first, second = placement.order
        self.assertGreater(placement.get(second).x, placement.get(first).x)
        self.assertTrue(placement.set_position(first, -500, -200))
        placement.persist()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            onnx.save(loaded, path)
            restored = onnx.load(path, layout=True)
        resumed = onnx.GraphLayout(restored.graph)
        resumed.load()
        check_model(restored)
        self.assertEqual(resumed.order, placement.order)
        self.assertEqual(resumed.get(first).x, -500)
        self.assertEqual(resumed.get(first).y, -200)
        self.assertEqual(
            [list(n.input) for n in restored.graph.node],
            [list(n.input) for n in model.graph.node],
        )
        self.assertEqual(
            [list(n.output) for n in restored.graph.node],
            [list(n.output) for n in model.graph.node],
        )
        del restored.graph.metadata_props[:]
        for node in restored.graph.node:
            del node.metadata_props[:]
        self.assertEqual(restored.SerializeToString(), original)

    def test_streaming_and_local_edits(self):
        graph = onnx.GraphProto()
        placement = onnx.GraphLayout(graph)
        placement.load()
        consumer = placement.append(oh.make_node("Add", ["late", "late"], ["Y"]))
        independent = placement.append(oh.make_node("Identity", ["X"], ["Z"]))
        before = placement.get(independent)
        producer = placement.insert(0, oh.make_node("Identity", ["X"], ["late"]))
        self.assertGreater(placement.get(consumer).x, placement.get(producer).x)
        self.assertEqual(len(placement.edges(consumer)), 2)
        self.assertEqual(
            [(e.source_port, e.target_port) for e in placement.edges(consumer)], [(0, 0), (0, 1)]
        )
        self.assertEqual(
            (placement.get(independent).x, placement.get(independent).y), (before.x, before.y)
        )
        self.assertTrue(placement.set_size(producer, 400, 140))
        self.assertGreaterEqual(
            placement.get(consumer).x, placement.get(producer).x + placement.get(producer).width
        )
        self.assertTrue(placement.rewire_input(consumer, 0, "X"))
        self.assertTrue(placement.replace(producer, oh.make_node("Relu", ["X"], ["late"])))
        self.assertTrue(placement.erase(producer))
        self.assertEqual(
            (placement.get(independent).x, placement.get(independent).y), (before.x, before.y)
        )

    def test_nested_loading_and_empty_graph(self):
        model = self.model()
        container = model.graph.node[0]
        single = container.attribute.add()
        single.name = "branch"
        single.type = onnx.AttributeProto.GRAPH
        single.g.CopyFrom(self.model().graph)
        multiple = container.attribute.add()
        multiple.name = "branches"
        multiple.type = onnx.AttributeProto.GRAPHS
        multiple.graphs.add().CopyFrom(self.model().graph)
        multiple.graphs.add().name = "empty"
        loaded = onnx.load(model.SerializeToString(), layout=True)
        placement = onnx.GraphLayout(loaded.graph)
        placement.load()
        children = placement.children(placement.order[0])
        self.assertEqual(len(children), 3)
        self.assertEqual([len(child.order) for child in children], [2, 2, 0])
        for child in children:
            self.assertEqual(child.diagnostics, [])
        placement.replace(placement.order[0], oh.make_node("Identity", ["X"], ["A"]))
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            _ = children[0].order
        empty = onnx.load(oh.make_model(onnx.GraphProto()).SerializeToString(), layout=True)
        self.assertTrue(
            any(p.key == "onnx_light.layout.version" for p in empty.graph.metadata_props)
        )

    def test_graph_callback_and_lifetime(self):
        visits = []
        options = onnx.ParseOptions()
        options.graph_callback = lambda graph: visits.append(graph.name)
        onnx.enable_graph_layout(options)
        loaded = onnx.ModelProto()
        loaded.ParseFromString(self.model().SerializeToString(), options)
        self.assertEqual(visits, ["layout"])
        placement = onnx.GraphLayout(loaded.graph)
        del loaded
        placement.load()
        self.assertEqual(len(placement.order), 2)
        snapshot = placement.get(placement.order[0])
        placement.erase(placement.order[0])
        self.assertGreater(snapshot.width, 0)
        with self.assertRaises(KeyError):
            placement.get(123456789)

    def test_layout_callback_can_be_retrieved_and_chained(self):
        visits = []
        options = onnx.ParseOptions()
        self.assertIsNone(options.graph_callback)

        def callback(graph):
            visits.append(graph.name)

        options.graph_callback = callback
        self.assertIs(options.graph_callback, callback)
        onnx.enable_graph_layout(options)
        layout_callback = options.graph_callback
        self.assertTrue(callable(layout_callback))
        options.graph_callback = None

        def chained(graph):
            layout_callback(graph)
            visits.append("after layout")

        options.graph_callback = chained
        loaded = onnx.ModelProto()
        loaded.ParseFromString(self.model().SerializeToString(), options)
        self.assertEqual(visits, ["layout", "after layout"])
        self.assertTrue(
            any(p.key == "onnx_light.layout.version" for p in loaded.graph.metadata_props)
        )

    def test_textproto(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.textproto"
            onnx.save(self.model(), path)
            loaded = onnx.load(path, layout=True)
        placement = onnx.GraphLayout(loaded.graph)
        placement.load()
        self.assertEqual(len(placement.order), 2)

    def test_textproto_reports_nested_layout_diagnostics(self):
        model = self.model()
        branch = model.graph.node[0].attribute.add()
        branch.name = "branch"
        branch.type = onnx.AttributeProto.GRAPH
        branch.g.CopyFrom(self.model().graph)
        version = branch.g.metadata_props.add()
        version.key = "onnx_light.layout.version"
        version.value = "999"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.textproto"
            onnx.save(model, path)
            with self.assertWarnsRegex(RuntimeWarning, "Unsupported layout metadata version"):
                onnx.load(path, layout=True)

    def test_metadata_diagnostics_and_unrelated_keys(self):
        model = self.model()
        entry = model.graph.metadata_props.add()
        entry.key = "application.note"
        entry.value = "keep"
        version = model.graph.metadata_props.add()
        version.key = "onnx_light.layout.version"
        version.value = "999"
        with self.assertWarns(RuntimeWarning):
            loaded = onnx.load(model.SerializeToString(), layout=True)
        self.assertEqual(
            [p.value for p in loaded.graph.metadata_props if p.key == "application.note"],
            ["keep"],
        )
        placement = onnx.GraphLayout(loaded.graph)
        placement.load()
        node_id = placement.order[0]
        self.assertFalse(placement.set_position(node_id, float("nan"), 0))
        self.assertFalse(placement.set_position(node_id, 0, float("inf")))
        self.assertFalse(placement.set_size(node_id, -1, 10))

    def test_partial_identity_metadata(self):
        model = onnx.load(self.model().SerializeToString(), layout=True)
        placement = onnx.GraphLayout(model.graph)
        placement.load()
        survivor = placement.order[1]
        second = placement.get(survivor)
        metadata = model.graph.node[0].metadata_props
        metadata[:] = [entry for entry in metadata if entry.key != "onnx_light.layout.id"]
        restored = onnx.load(model.SerializeToString(), layout=True)
        resumed = onnx.GraphLayout(restored.graph)
        resumed.load()
        self.assertEqual(resumed.order[1], survivor)
        self.assertEqual((resumed.get(survivor).x, resumed.get(survivor).y), (second.x, second.y))


if __name__ == "__main__":
    unittest.main()
