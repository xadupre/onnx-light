import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_proto._text_format import parse_from_textproto, serialize_to_textproto

_HAS_ONNX = importlib.util.find_spec("onnx") is not None


def _rich_model() -> onnxl.ModelProto:
    """Builds a model exercising many textproto field kinds."""
    x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1, 3, "N"])
    y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1, 3, "N"])
    init = oh.make_tensor("B", onnxl.TensorProto.FLOAT, [3], [1.5, -2.0, 3.25])
    add = oh.make_node("Add", ["X", "B"], ["T"], name="add0", doc_string="an add")
    # A node carrying a few attribute kinds, including non-utf8 bytes.
    cst = oh.make_node("Constant", [], ["C"], name="cst")
    a_f = cst.attribute.add()
    a_f.name = "alpha"
    a_f.type = onnxl.AttributeProto.FLOAT
    a_f.f = 0.5
    a_i = cst.attribute.add()
    a_i.name = "beta"
    a_i.type = onnxl.AttributeProto.INTS
    a_i.ints.append(1)
    a_i.ints.append(-7)
    a_s = cst.attribute.add()
    a_s.name = "raw"
    a_s.type = onnxl.AttributeProto.STRING
    a_s.s = b'hi\xff\n"quote"\\back'
    out = oh.make_node("Mul", ["T", "C"], ["Y"], name="mul0")
    graph = oh.make_graph([add, cst, out], "g", [x], [y], [init])
    graph.metadata_props.add().key = "author"
    graph.metadata_props[0].value = "onnx-light"
    model = oh.make_model(
        graph, producer_name="onnx-light-test", opset_imports=[oh.make_opsetid("", 18)]
    )
    model.ir_version = 9
    model.doc_string = "documentation"
    return model


class TestTextFormat(ExtTestCase):
    """Tests for the pure-Python textproto parser and serializer."""

    def test_rejects_retired_rewriting_renames(self):
        legacy = 'value_renames { key: "old" value: "new" }'
        for proto, text in (
            (onnxl.LocalRewritingProto(), legacy),
            (onnxl.RewritingProto(), f"rewrites {{ {legacy} }}"),
        ):
            with (
                self.subTest(proto=type(proto).__name__),
                self.assertRaisesRegex(ValueError, "value_renames is retired"),
            ):
                parse_from_textproto(text, proto)

    def test_serialize_returns_text(self):
        """Tests that serializing a model returns readable textproto."""
        model = _rich_model()
        text = serialize_to_textproto(model)
        self.assertIsInstance(text, str)
        self.assertIn('producer_name: "onnx-light-test"', text)
        self.assertIn("graph {", text)
        self.assertIn('op_type: "Add"', text)

    def test_round_trip_rich_model(self):
        """Tests that a rich model round-trips through serialize/parse exactly."""
        model = _rich_model()
        text = serialize_to_textproto(model)
        parsed = parse_from_textproto(text, onnxl.ModelProto())
        self.assertEqual(model.SerializeToString(), parsed.SerializeToString())

    def test_round_trip_function_replacements(self):
        plan = onnxl.RewritingProto()
        function = plan.added_functions.add()
        function.domain = "custom"
        function.name = "F:pair"
        function.overload = "v%1"
        plan.replaced_function_ids.append("custom:F%3Apair:v%251")
        text = serialize_to_textproto(plan)
        parsed = parse_from_textproto(text, onnxl.RewritingProto())
        self.assertEqual(plan.SerializeToString(), parsed.SerializeToString())

    def test_round_trip_pipeline(self):
        """Tests models and generation settings in a pipeline textproto."""
        pipeline = onnxl.PipelineProto(
            name="text-generation",
            prefill=_rich_model(),
            max_new_tokens=64,
            temperature=0.75,
            top_k=20,
            top_p=0.9,
            input_ids_name="input_ids",
            logits_name="logits",
            eos_token_ids=[0, 2, 2],
        )
        metadata = pipeline.metadata_props.add()
        metadata.key = "family"
        metadata.value = "decoder-only"

        text = serialize_to_textproto(pipeline)
        self.assertIn('name: "text-generation"', text)
        self.assertIn("prefill {", text)
        self.assertIn("max_new_tokens: 64", text)
        parsed = parse_from_textproto(text, onnxl.PipelineProto())
        self.assertIn("eos_token_ids: 0", text)
        self.assertTrue(parsed.HasField("eos_token_ids"))
        self.assertEqual(list(parsed.eos_token_ids), [0, 2, 2])
        restored = onnxl.PipelineProto()
        restored.ParseFromString(pipeline.SerializeToString())
        self.assertEqual(list(restored.eos_token_ids), [0, 2, 2])
        self.assertEqual(pipeline.SerializeToString(), parsed.SerializeToString())

    def test_round_trip_non_utf8_bytes(self):
        """Tests that non-UTF-8 attribute bytes survive a round-trip."""
        model = _rich_model()
        parsed = parse_from_textproto(serialize_to_textproto(model), onnxl.ModelProto())
        attr = parsed.graph.node[1].attribute[2]
        self.assertEqual(attr.s, b'hi\xff\n"quote"\\back')

    def test_round_trip_subgraph_attribute(self):
        """Tests that a node with a subgraph attribute round-trips exactly."""
        body = oh.make_graph(
            [oh.make_node("Identity", ["x"], ["y"])],
            "body",
            [oh.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, None)],
            [oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, None)],
        )
        node = oh.make_node("Loop", ["M", "cond", "x"], ["y"])
        attr = node.attribute.add()
        attr.name = "body"
        attr.type = onnxl.AttributeProto.GRAPH
        attr.g.CopyFrom(body)
        graph = oh.make_graph([node], "g", [], [])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        text = serialize_to_textproto(model)
        parsed = parse_from_textproto(text, onnxl.ModelProto())
        self.assertEqual(model.SerializeToString(), parsed.SerializeToString())

    def test_round_trip_tensor_raw_data(self):
        """Tests that a tensor stored with raw_data round-trips exactly."""
        tensor = onnxl.TensorProto()
        tensor.name = "w"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.dims.append(2)
        tensor.raw_data = b"\x00\x01\x02\x03\x04\x05\x06\x07"
        text = serialize_to_textproto(tensor)
        self.assertIn("data_type: 1", text)
        parsed = parse_from_textproto(text, onnxl.TensorProto())
        self.assertEqual(tensor.SerializeToString(), parsed.SerializeToString())

    def test_data_location_enum_name(self):
        """Tests that the true enum field data_location is written using its name."""
        tensor = onnxl.TensorProto()
        tensor.name = "w"
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key = "location"
        entry.value = "weights.bin"
        text = serialize_to_textproto(tensor)
        self.assertIn("data_location: EXTERNAL", text)
        parsed = parse_from_textproto(text, onnxl.TensorProto())
        self.assertEqual(tensor.SerializeToString(), parsed.SerializeToString())

    def test_attribute_type_enum_name(self):
        """Tests that the true enum field AttributeProto.type is written using its name."""
        attr = onnxl.AttributeProto()
        attr.name = "a"
        attr.type = onnxl.AttributeProto.FLOAT
        attr.f = 1.0
        text = serialize_to_textproto(attr)
        self.assertIn("type: FLOAT", text)

    def test_parse_accepts_enum_integer(self):
        """Tests that the parser accepts integer values for true enum fields."""
        text = 'name: "w"\ndata_location: 1\n'
        parsed = parse_from_textproto(text, onnxl.TensorProto())
        self.assertEqual(int(parsed.data_location), 1)

    def test_parse_skips_unknown_fields(self):
        """Tests that unknown scalar and message fields are skipped during parsing."""
        text = (
            'producer_name: "p"\n'
            "unknown_scalar: 123\n"
            "unknown_message { nested: 1 deeper { x: 2 } }\n"
            "ir_version: 9\n"
        )
        parsed = parse_from_textproto(text, onnxl.ModelProto())
        self.assertEqual(parsed.producer_name, "p")
        self.assertEqual(parsed.ir_version, 9)

    def test_parse_angle_bracket_message(self):
        """Tests that the parser accepts angle-bracket message delimiters."""
        text = 'graph <name: "g">\n'
        parsed = parse_from_textproto(text, onnxl.ModelProto())
        self.assertEqual(parsed.graph.name, "g")

    def test_serialize_unsupported_type_raises(self):
        """Tests that serializing an unsupported object raises TypeError."""
        with self.assertRaises(TypeError):
            serialize_to_textproto(object())


class TestTextFormatIO(ExtTestCase):
    """Tests for load/save with the textproto format."""

    def test_save_load_infer_from_extension(self):
        """Tests that save/load infer textproto from the .textproto extension."""
        model = _rich_model()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.textproto")
            onnxl.save(model, path)
            loaded = onnxl.load(path)
            self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_save_load_explicit_format(self):
        """Tests that save/load honor an explicit textproto format on any extension."""
        model = _rich_model()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.onnx")
            onnxl.save(model, path, format="textproto")
            loaded = onnxl.load(path, format="textproto")
            self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_load_textproto_from_bytes(self):
        """Tests that load parses textproto provided as a bytes object."""
        model = _rich_model()
        text_bytes = serialize_to_textproto(model).encode("utf-8")
        loaded = onnxl.load(text_bytes, format="textproto")
        self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_save_textproto_rejects_external_data(self):
        """Tests that saving textproto with external data raises ValueError."""
        model = _rich_model()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.textproto")
            with self.assertRaises(ValueError):
                onnxl.save(model, path, save_as_external_data=True)

    def test_load_unsupported_format_raises(self):
        """Tests that load raises ValueError for an unsupported format."""
        with self.assertRaises(ValueError) as ctx:
            onnxl.load(b"", format="json")
        self.assertIn("Unsupported format", str(ctx.exception))


class TestPipelineIO(ExtTestCase):
    def test_save_load_pipeline_extensions(self):
        pipeline = onnxl.PipelineProto(
            name="generation", prefill=_rich_model(), max_new_tokens=16
        )
        with tempfile.TemporaryDirectory() as folder:
            for extension in (".pipe-onnx", ".pipeline"):
                for path_type in (str, Path):
                    for num_threads in (1, -1):
                        with self.subTest(
                            extension=extension, path_type=path_type, num_threads=num_threads
                        ):
                            path = path_type(os.path.join(folder, f"generation{extension}"))
                            onnxl.save(pipeline, path, num_threads=num_threads)
                            self.assertEqual(
                                Path(path).read_bytes(), pipeline.SerializeToString()
                            )
                            loaded = onnxl.load(path, num_threads=num_threads)
                            self.assertIsInstance(loaded, onnxl.PipelineProto)
                            self.assertEqual(
                                pipeline.SerializeToString(), loaded.SerializeToString()
                            )
                            self.assertFalse(loaded.HasField("decode"))

    def test_save_pipeline_stream(self):
        pipeline = onnxl.PipelineProto(name="generation", prefill=_rich_model())
        for format in ("protobuf", "textproto"):
            with self.subTest(format=format):
                stream = io.BytesIO()
                onnxl.save(pipeline, stream, format=format)
                loaded = onnxl.PipelineProto()
                if format == "protobuf":
                    loaded.ParseFromString(stream.getvalue())
                else:
                    parse_from_textproto(stream.getvalue().decode("utf-8"), loaded)
                self.assertEqual(pipeline.SerializeToString(), loaded.SerializeToString())

    def test_pipeline_explicit_textproto(self):
        pipeline = onnxl.PipelineProto(
            name="generation",
            embedding=_rich_model(),
            prefill=_rich_model(),
            decode=_rich_model(),
        )
        with tempfile.TemporaryDirectory() as folder:
            for extension in (".pipe-onnx", ".pipeline"):
                with self.subTest(extension=extension):
                    path = Path(folder) / f"generation{extension}"
                    onnxl.save(pipeline, path, format="textproto")
                    self.assertIn('name: "generation"', path.read_text(encoding="utf-8"))
                    loaded = onnxl.load(path, format="textproto")
                    self.assertIsInstance(loaded, onnxl.PipelineProto)
                    self.assertEqual(pipeline.SerializeToString(), loaded.SerializeToString())

    def test_other_extensions_still_load_models(self):
        model = _rich_model()
        with tempfile.TemporaryDirectory() as folder:
            for extension in (".onnx", ".pb", ".pipeline.onnx", ""):
                with self.subTest(extension=extension):
                    path = Path(folder) / f"model{extension}"
                    onnxl.save(model, path)
                    loaded = onnxl.load(path)
                    self.assertIsInstance(loaded, onnxl.ModelProto)
                    self.assertEqual(model.SerializeToString(), loaded.SerializeToString())
        self.assertIsInstance(onnxl.load(model.SerializeToString()), onnxl.ModelProto)

    def test_pipeline_rejects_external_output(self):
        pipeline = onnxl.PipelineProto(name="generation", prefill=_rich_model())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "generation.pipeline"
            weights = Path(folder) / "generation.data"
            path.write_bytes(b"existing pipeline")
            weights.write_bytes(b"existing weights")
            with self.assertRaisesRegex(RuntimeError, "does not support external-weight"):
                onnxl.save(pipeline, path, save_as_external_data=True, location=str(weights))
            self.assertEqual(path.read_bytes(), b"existing pipeline")
            self.assertEqual(weights.read_bytes(), b"existing weights")

    def test_pipeline_rejects_unsupported_format(self):
        pipeline = onnxl.PipelineProto(name="generation", prefill=_rich_model())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "generation.pipe-onnx"
            with self.assertRaisesRegex(ValueError, "Unsupported format"):
                onnxl.save(pipeline, path, format="json")
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(ValueError, "Unsupported format"):
                onnxl.load(path, format="json")


@unittest.skipUnless(_HAS_ONNX, "onnx is not installed")
class TestTextFormatOnnxInterop(ExtTestCase):
    """Tests interoperability with the onnx/protobuf text format."""

    def test_onnxlight_text_parsed_by_onnx(self):
        """Tests that text produced by onnx-light is parseable by onnx/protobuf."""
        import onnx
        from google.protobuf import text_format

        model = _rich_model()
        text = serialize_to_textproto(model)
        onnx_model = onnx.ModelProto()
        text_format.Parse(text, onnx_model)
        from_binary = onnx.load_model_from_string(model.SerializeToString())
        self.assertEqual(onnx_model.graph.name, from_binary.graph.name)
        self.assertEqual(onnx_model.producer_name, from_binary.producer_name)
        self.assertEqual(len(onnx_model.graph.node), len(from_binary.graph.node))

    def test_onnx_text_parsed_by_onnxlight(self):
        """Tests that text produced by onnx/protobuf is parseable by onnx-light."""
        import onnx
        from google.protobuf import text_format

        model = _rich_model()
        from_binary = onnx.load_model_from_string(model.SerializeToString())
        onnx_text = text_format.MessageToString(from_binary)
        parsed = parse_from_textproto(onnx_text, onnxl.ModelProto())
        self.assertEqual(model.SerializeToString(), parsed.SerializeToString())


if __name__ == "__main__":
    unittest.main(verbosity=2)
