# source: https://github.com/onnx/onnx/blob/main/onnx/test/helper_test.py
import os
import random
import tempfile
import unittest
from typing import Any
import numpy as np
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.checker as checker
import onnx_light.onnx.helper as oh


class TestOnnxLightHelper(ExtTestCase):
    def test_make_operatorsetid(self):
        op = oh.make_operatorsetid("", 19)
        self.assertEqual(op.domain, "")
        self.assertEqual(op.version, 19)
        op = oh.make_operatorsetid("ai.onnx.ml", 5)
        self.assertEqual(op.domain, "ai.onnx.ml")
        self.assertEqual(op.version, 5)
        s = str(op)
        self.assertIn('domain: "ai.onnx.ml",', s)

    def test_make_tensor_type_proto(self) -> None:
        proto = oh.make_tensor_type_proto(elem_type=2, shape=None)
        self.assertEqual(proto.tensor_type.elem_type, 2)
        self.assertNotEmpty(proto.tensor_type.shape)
        self.assertEmpty(proto.sequence_type)
        s = str(proto)
        self.assertIn("elem_type: 2,", s)

    def test_make_optional_value_info(self) -> None:
        tensor_type_proto = oh.make_tensor_type_proto(elem_type=2, shape=[5])
        tensor_val_into = oh.make_value_info(name="test", type_proto=tensor_type_proto)
        optional_type_proto = oh.make_optional_type_proto(tensor_type_proto)
        optional_val_info = oh.make_value_info(name="test", type_proto=optional_type_proto)

        self.assertEqual(optional_val_info.name, "test")
        self.assertTrue(optional_val_info.type.optional_type)
        self.assertEqual(optional_val_info.type.optional_type.elem_type, tensor_val_into.type)

        # Test Sequence
        sequence_type_proto = oh.make_sequence_type_proto(tensor_type_proto)
        optional_type_proto = oh.make_optional_type_proto(sequence_type_proto)
        optional_val_info = oh.make_value_info(name="test", type_proto=optional_type_proto)

        self.assertEqual(optional_val_info.name, "test")
        self.assertTrue(optional_val_info.type.optional_type)
        sequence_value_info = oh.make_value_info(name="test", type_proto=tensor_type_proto)
        self.assertEqual(
            optional_val_info.type.optional_type.elem_type.sequence_type.elem_type,
            sequence_value_info.type,
        )

    def test_make_sequence_value_info3(self) -> None:
        tensor_type_proto = oh.make_tensor_type_proto(elem_type=2, shape=None)
        sequence_type_proto = oh.make_sequence_type_proto(tensor_type_proto)
        sequence_val_info = oh.make_value_info(name="test", type_proto=sequence_type_proto)
        sequence_val_info_prim = oh.make_tensor_sequence_value_info(
            name="test", elem_type=2, shape=None
        )

        self.assertEqual(sequence_val_info, sequence_val_info_prim)

    def test_attr_float(self) -> None:
        # float
        attr = oh.make_attribute("float", 1.0)
        self.assertEqual(attr.name, "float")
        self.assertEqual(attr.f, 1.0)
        checker.check_attribute(attr)
        # float with scientific
        attr = oh.make_attribute("float", 1e10)
        self.assertEqual(attr.name, "float")
        self.assertEqual(attr.f, 1e10)
        checker.check_attribute(attr)

    def test_attr_int(self) -> None:
        # integer
        attr = oh.make_attribute("int", 3)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 3)
        checker.check_attribute(attr)
        # long integer
        attr = oh.make_attribute("int", 5)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 5)
        checker.check_attribute(attr)
        # octinteger
        attr = oh.make_attribute("int", 0o1701)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 0o1701)
        checker.check_attribute(attr)
        # hexinteger
        attr = oh.make_attribute("int", 0x1701)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 0x1701)
        checker.check_attribute(attr)

    def test_attr_doc_string(self) -> None:
        attr = oh.make_attribute("a", "value")
        self.assertEqual(attr.name, "a")
        self.assertEqual(attr.doc_string, "")
        attr = oh.make_attribute("a", "value", "doc")
        self.assertEqual(attr.name, "a")
        self.assertEqual(attr.doc_string, "doc")

    def test_make_attribute_ref(self) -> None:
        attr = oh.make_attribute_ref(
            "alpha", onnxl.AttributeProto.FLOAT, ref_attr_name="parent_alpha"
        )
        self.assertEqual(attr.name, "alpha")
        self.assertEqual(attr.type, onnxl.AttributeProto.FLOAT)
        self.assertEqual(attr.ref_attr_name, "parent_alpha")
        # A reference attribute carries no data; it is resolved from the parent
        # function's attribute at instantiation time.
        with self.assertRaises(ValueError):
            oh.get_attribute_value(attr)

    def test_make_attribute_ref_doc_string(self) -> None:
        attr = oh.make_attribute_ref(
            "alpha", onnxl.AttributeProto.FLOAT, doc_string="doc", ref_attr_name="parent_alpha"
        )
        self.assertEqual(attr.ref_attr_name, "parent_alpha")
        self.assertEqual(attr.doc_string, "doc")

    def test_make_attribute_ref_doc_string_positional(self) -> None:
        attr = oh.make_attribute_ref("alpha", onnxl.AttributeProto.FLOAT, "doc")
        self.assertEqual(attr.ref_attr_name, "alpha")
        self.assertEqual(attr.doc_string, "doc")

    def test_make_attribute_ref_requires_ref_attr_name(self) -> None:
        with self.assertRaises(ValueError):
            oh.make_attribute_ref("alpha", onnxl.AttributeProto.FLOAT, ref_attr_name="")

    def test_attr_string(self) -> None:
        # bytes
        attr = oh.make_attribute("str", b"test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        checker.check_attribute(attr)
        # unspecified
        attr = oh.make_attribute("str", "test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        checker.check_attribute(attr)
        # unicode
        attr = oh.make_attribute("str", "test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        checker.check_attribute(attr)

    def test_attr_repeated_float(self) -> None:
        attr = oh.make_attribute("floats", [1.0, 2.0])
        self.assertEqual(attr.name, "floats")
        self.assertEqual(list(attr.floats), [1.0, 2.0])
        checker.check_attribute(attr)

    def test_attr_repeated_int(self) -> None:
        attr = oh.make_attribute("ints", [1, 2])
        self.assertEqual(attr.name, "ints")
        self.assertEqual(list(attr.ints), [1, 2])
        checker.check_attribute(attr)
        self.assertEqual(repr(attr.ints), "[1, 2]")

    def test_attr_repeated_mixed_floats_and_ints(self) -> None:
        attr = oh.make_attribute("mixed", [1, 2, 3.0, 4.5])
        self.assertEqual(attr.name, "mixed")
        self.assertEqual(list(attr.floats), [1.0, 2.0, 3.0, 4.5])
        checker.check_attribute(attr)

    def test_attr_repeated_str(self) -> None:
        attr = oh.make_attribute("strings", ["str1", "str2"])
        self.assertEqual(attr.name, "strings")
        self.assertEqual(list(attr.strings), [b"str1", b"str2"])
        checker.check_attribute(attr)
        self.assertEqual(repr(attr.strings), "['str1', 'str2']")

    def test_make_tensor_external_data_dict(self) -> None:
        tensor = oh.make_tensor(
            name="weights",
            data_type=onnxl.TensorProto.FLOAT,
            dims=(2, 3),
            external_data={"location": "weights.bin", "offset": "0", "length": "24"},
        )
        self.assertEqual(tensor.name, "weights")
        self.assertEqual(list(tensor.dims), [2, 3])
        self.assertEqual(int(tensor.data_location), int(onnxl.TensorProto.EXTERNAL))
        entries = {e.key: e.value for e in tensor.external_data}
        self.assertEqual(entries, {"location": "weights.bin", "offset": "0", "length": "24"})
        self.assertEqual(len(tensor.float_data), 0)
        self.assertEqual(len(tensor.raw_data), 0)

    def test_make_tensor_external_data_sequence(self) -> None:
        tensor = oh.make_tensor(
            name="w",
            data_type=onnxl.TensorProto.FLOAT,
            dims=(2,),
            external_data=[("location", "f.bin"), ("offset", "8")],
        )
        self.assertEqual(int(tensor.data_location), int(onnxl.TensorProto.EXTERNAL))
        self.assertEqual(
            [(e.key, e.value) for e in tensor.external_data],
            [("location", "f.bin"), ("offset", "8")],
        )

    def test_make_tensor_external_data_errors(self) -> None:
        self.assertRaises(
            ValueError,
            oh.make_tensor,
            "b",
            onnxl.TensorProto.FLOAT,
            [1],
            vals=[1.0],
            external_data={"location": "x"},
        )
        self.assertRaises(ValueError, oh.make_tensor, "c", onnxl.TensorProto.FLOAT, [1])
        self.assertRaises(
            ValueError,
            oh.make_tensor,
            "d",
            onnxl.TensorProto.FLOAT,
            [1],
            raw=True,
            external_data={"location": "x"},
        )

    def test_attr_repeated_tensor_proto(self) -> None:
        tensors = [
            oh.make_tensor(
                name="a", data_type=onnxl.TensorProto.FLOAT, dims=(1,), vals=np.ones(1)
            ),
            oh.make_tensor(
                name="b", data_type=onnxl.TensorProto.FLOAT, dims=(1,), vals=np.ones(1)
            ),
        ]
        attr = oh.make_attribute("tensors", tensors)
        attr_tensors = list(attr.tensors)
        self.assertIsInstance(attr_tensors, list)
        self.assertIsInstance(tensors, list)
        self.assertEqual(attr.name, "tensors")
        self.assertEqual(tensors, attr_tensors)
        checker.check_attribute(attr)

    def test_attr_sparse_tensor_proto(self) -> None:
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = oh.make_tensor(
            name="sparse_values",
            data_type=onnxl.TensorProto.FLOAT,
            dims=[len(sparse_values)],
            vals=np.array(sparse_values).astype(np.float32),
            raw=False,
        )

        linear_indices = [2, 3, 5]
        indices_tensor = oh.make_tensor(
            name="indices",
            data_type=onnxl.TensorProto.INT64,
            dims=[len(linear_indices)],
            vals=np.array(linear_indices).astype(np.int64),
            raw=False,
        )
        sparse_tensor = oh.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)

        attr = oh.make_attribute("sparse_attr", sparse_tensor)
        self.assertEqual(attr.name, "sparse_attr")
        checker.check_sparse_tensor(oh.get_attribute_value(attr))
        checker.check_attribute(attr)

    def test_attr_sparse_tensor_repeated_protos(self) -> None:
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = oh.make_tensor(
            name="sparse_values",
            data_type=onnxl.TensorProto.FLOAT,
            dims=[len(sparse_values)],
            vals=np.array(sparse_values).astype(np.float32),
            raw=False,
        )

        linear_indices = [2, 3, 5]
        indices_tensor = oh.make_tensor(
            name="indices",
            data_type=onnxl.TensorProto.INT64,
            dims=[len(linear_indices)],
            vals=np.array(linear_indices).astype(np.int64),
            raw=False,
        )
        sparse_tensor = oh.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)

        repeated_sparse = [sparse_tensor, sparse_tensor]
        attr = oh.make_attribute("sparse_attrs", repeated_sparse)
        self.assertEqual(attr.name, "sparse_attrs")
        checker.check_attribute(attr)
        for s in oh.get_attribute_value(attr):
            checker.check_sparse_tensor(s)

    @unittest.skipIf(True, "not yet implemented")
    def test_attr_repeated_graph_proto(self) -> None:
        graphs = [onnxl.GraphProto(), onnxl.GraphProto()]
        graphs[0].name = "a"
        graphs[1].name = "b"
        attr = oh.make_attribute("graphs", graphs)
        self.assertEqual(attr.name, "graphs")
        self.assertEqual(list(attr.graphs), graphs)
        checker.check_attribute(attr)

    def test_attr_empty_list(self) -> None:
        attr = oh.make_attribute("empty", [], attr_type=onnxl.AttributeProto.STRINGS)
        self.assertEqual(int(attr.type), onnxl.AttributeProto.STRINGS)
        self.assertEqual(len(attr.strings), 0)
        self.assertRaises(ValueError, oh.make_attribute, "empty", [])

    def test_attr_mismatch(self) -> None:
        with self.assertRaisesRegex(TypeError, "Inferred attribute type 'FLOAT'"):
            oh.make_attribute("test", 6.4, attr_type=onnxl.AttributeProto.STRING)

    def test_is_attr_legal(self) -> None:
        # no name, no field
        attr = onnxl.AttributeProto()
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)
        # name, but no field
        attr = onnxl.AttributeProto()
        attr.name = "test"
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)
        # name, with two fields
        attr = onnxl.AttributeProto()
        attr.name = "test"
        attr.f = 1.0
        attr.i = 2
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)

    def test_is_attr_legal_verbose(self) -> None:
        def _set(
            attr: onnxl.AttributeProto,
            type_: onnxl.AttributeProto.AttributeType,
            var: str,
            value: Any,
        ) -> None:
            setattr(attr, var, value)
            attr.type = type_

        def _extend(
            attr: onnxl.AttributeProto,
            type_: onnxl.AttributeProto.AttributeType,
            var: list[Any],
            value: Any,
        ) -> None:
            var.extend(value)
            attr.type = type_

        SET_ATTR = [
            (lambda attr: _set(attr, onnxl.AttributeProto.FLOAT, "f", 1.0)),
            (lambda attr: _set(attr, onnxl.AttributeProto.INT, "i", 1)),
            (lambda attr: _set(attr, onnxl.AttributeProto.STRING, "s", b"str")),
            (lambda attr: _extend(attr, onnxl.AttributeProto.FLOATS, attr.floats, [1.0, 2.0])),
            (lambda attr: _extend(attr, onnxl.AttributeProto.INTS, attr.ints, [1, 2])),
            (
                lambda attr: _extend(
                    attr, onnxl.AttributeProto.STRINGS, attr.strings, [b"a", b"b"]
                )
            ),
        ]
        # Randomly set one field, and the result should be legal.
        for _i in range(100):
            attr = onnxl.AttributeProto()
            attr.name = "test"
            random.choice(SET_ATTR)(attr)
            checker.check_attribute(attr)
        # Randomly set two fields, and then ensure helper function catches it.
        for _i in range(100):
            attr = onnxl.AttributeProto()
            attr.name = "test"
            for func in random.sample(SET_ATTR, 2):
                func(attr)
            self.assertRaises(checker.ValidationError, checker.check_attribute, attr)

    def test_node_no_arg(self) -> None:
        node_def = oh.make_node("Relu", ["X"], ["Y"], name="test")
        self.assertEqual(node_def.op_type, "Relu")
        self.assertEqual(node_def.name, "test")
        self.assertEqual(list(node_def.input), ["X"])
        self.assertEqual(list(node_def.output), ["Y"])

    def test_node_with_arg(self) -> None:
        node_def = oh.make_node("Relu", ["X"], ["Y"], arg_value=1)
        self.assertEqual(node_def.op_type, "Relu")
        self.assertEqual(list(node_def.input), ["X"])
        self.assertEqual(list(node_def.output), ["Y"])
        self.assertEqual(len(node_def.attribute), 1)
        self.assertEqual(node_def.attribute[0], oh.make_attribute("arg_value", 1))

    def test_node_domain(self) -> None:
        node_def = oh.make_node(
            "Relu", ["X"], ["Y"], name="test", doc_string="doc", domain="test.domain"
        )
        self.assertEqual(node_def.domain, "test.domain")

    def test_graph(self) -> None:
        node_def1 = oh.make_node("Relu", ["X"], ["Y"])
        node_def2 = oh.make_node("Add", ["X", "Y"], ["Z"])
        value_info = [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1, 2])]
        graph = oh.make_graph(
            [node_def1, node_def2],
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1, 2])],
            [oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, [1, 2])],
            doc_string=None,
            value_info=value_info,
        )
        self.assertEqual(graph.name, "test")
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0], node_def1)
        self.assertEqual(graph.node[1], node_def2)
        self.assertEqual(graph.doc_string, "")
        self.assertEqual(graph.value_info[0], value_info[0])

    def test_graph_docstring(self) -> None:
        graph = oh.make_graph([], "my graph", [], [], None, "my docs")
        self.assertEqual(graph.name, "my graph")
        self.assertEqual(graph.doc_string, "my docs")

    def test_model(self) -> None:
        node_def = oh.make_node("Relu", ["X"], ["Y"])
        graph_def = oh.make_graph(
            [node_def],
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1, 2])],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1, 2])],
        )
        self.assertRaises((AttributeError, TypeError), oh.make_model, graph_def, xxx=1)
        model_def = oh.make_model(graph_def, producer_name="test")
        self.assertEqual(model_def.producer_name, "test")

    def test_model_docstring(self) -> None:
        graph = oh.make_graph([], "my graph", [], [])
        model_def = oh.make_model(graph, doc_string="test")
        # models may have their own documentation, but don't have a name
        # their name is the domain-qualified name of the underlying graph.
        self.assertFalse(hasattr(model_def, "name"))
        self.assertEqual(model_def.doc_string, "test")

    def test_model_metadata_props(self) -> None:
        graph = oh.make_graph([], "my graph", [], [])
        model_def = oh.make_model(graph, doc_string="test")
        oh.set_model_props(model_def, {"Title": "my graph", "Keywords": "test;graph"})
        checker.check_model(model_def)
        oh.set_model_props(model_def, {"Title": "my graph", "Keywords": "test;graph"})
        checker.check_model(model_def)  # helper replaces, so no dupe

        dupe = model_def.metadata_props.add()
        dupe.key = "Title"
        dupe.value = "Other"
        self.assertRaises(checker.ValidationError, checker.check_model, model_def)

    def test_make_optional(self) -> None:
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        values_tensor = oh.make_tensor(
            name="test", data_type=onnxl.TensorProto.FLOAT, dims=(5,), vals=values
        )
        optional = oh.make_optional(
            name="test", elem_type=onnxl.OptionalProto.TENSOR, value=values_tensor
        )
        self.assertEqual(optional.name, "test")
        self.assertEqual(optional.elem_type, onnxl.OptionalProto.TENSOR)
        self.assertEqual(optional.tensor_value, values_tensor)

        # Test Sequence
        values_sequence = oh.make_sequence(
            name="test",
            elem_type=onnxl.SequenceProto.TENSOR,
            values=[values_tensor, values_tensor],
        )
        optional = oh.make_optional(
            name="test", elem_type=onnxl.OptionalProto.SEQUENCE, value=values_sequence
        )
        self.assertEqual(optional.name, "test")
        self.assertEqual(optional.elem_type, onnxl.OptionalProto.SEQUENCE)
        self.assertEqual(optional.sequence_value, values_sequence)

        # Test None
        optional_none = oh.make_optional(
            name="test", elem_type=onnxl.OptionalProto.UNDEFINED, value=None
        )
        self.assertEqual(optional_none.name, "test")
        self.assertEqual(optional_none.elem_type, onnxl.OptionalProto.UNDEFINED)
        self.assertFalse(optional_none.HasField("tensor_value"))

    def test_make_optional_value_info2(self) -> None:
        tensor_type_proto = oh.make_tensor_type_proto(elem_type=2, shape=[5])
        tensor_val_into = oh.make_value_info(name="test", type_proto=tensor_type_proto)
        optional_type_proto = oh.make_optional_type_proto(tensor_type_proto)
        optional_val_info = oh.make_value_info(name="test", type_proto=optional_type_proto)

        self.assertEqual(optional_val_info.name, "test")
        self.assertTrue(optional_val_info.type.optional_type)
        self.assertEqual(optional_val_info.type.optional_type.elem_type, tensor_val_into.type)

        # Test Sequence
        sequence_type_proto = oh.make_sequence_type_proto(tensor_type_proto)
        optional_type_proto = oh.make_optional_type_proto(sequence_type_proto)
        optional_val_info = oh.make_value_info(name="test", type_proto=optional_type_proto)

        self.assertEqual(optional_val_info.name, "test")
        self.assertTrue(optional_val_info.type.optional_type)
        sequence_value_info = oh.make_value_info(name="test", type_proto=tensor_type_proto)
        self.assertEqual(
            optional_val_info.type.optional_type.elem_type.sequence_type.elem_type,
            sequence_value_info.type,
        )

    def test_make_sequence_value_info(self) -> None:
        tensor_type_proto = oh.make_tensor_type_proto(elem_type=2, shape=None)
        sequence_type_proto = oh.make_sequence_type_proto(tensor_type_proto)
        sequence_val_info = oh.make_value_info(name="test", type_proto=sequence_type_proto)
        sequence_val_info_prim = oh.make_tensor_sequence_value_info(
            name="test", elem_type=2, shape=None
        )

        self.assertEqual(sequence_val_info, sequence_val_info_prim)

    def test_make_map(self):
        m = oh.make_map(
            "map",
            onnxl.TensorProto.INT8,
            [0],
            oh.make_sequence(
                "seq",
                onnxl.SequenceProto.DataType.TENSOR,
                [
                    oh.make_tensor(
                        name="test",
                        data_type=onnxl.TensorProto.FLOAT,
                        dims=(5,),
                        vals=[1.1, 2.2, 3.3, 4.4, 5.5],
                    )
                ],
            ),
        )
        self.assertIsInstance(m, onnxl.MapProto)


class TestAlignExternalDataStreaming(ExtTestCase):
    @staticmethod
    def _make_model_with_external(path: str) -> list[np.ndarray]:
        """Builds and saves a model with three differently-sized FLOAT initializers
        as external data; returns the per-initializer numpy payloads."""
        payloads = [
            np.full((7,), 1.5, dtype=np.float32),  # 28 bytes (unaligned)
            np.full((3,), -2.0, dtype=np.float32),  # 12 bytes
            np.full((17,), 3.0, dtype=np.float32),  # 68 bytes
        ]
        inits = [
            oh.make_tensor(
                name=f"w{i}",
                data_type=onnxl.TensorProto.FLOAT,
                dims=arr.shape,
                vals=arr.tobytes(),
                raw=True,
            )
            for i, arr in enumerate(payloads)
        ]
        graph = oh.make_graph([], "g", [], [], initializer=inits)
        model = oh.make_model(graph, producer_name="test")
        onnxl.save(model, path, save_as_external_data=True, size_threshold=0)
        return payloads

    def test_align_external_data_streaming_python_binding(self) -> None:
        from onnx_light.onnx import align_external_data_streaming

        with tempfile.TemporaryDirectory() as tdir:
            src_onnx = os.path.join(tdir, "src.onnx")
            dst_onnx = os.path.join(tdir, "dst.onnx")
            dst_weights = os.path.join(tdir, "dst.data")
            payloads = self._make_model_with_external(src_onnx)
            # Default location used by onnxl.save is "<src_onnx>.data"
            self.assertTrue(os.path.exists(src_onnx + ".data"))

            alignment = 64
            # Tiny chunk_size forces multiple iterations of the streaming copy loop.
            total = align_external_data_streaming(
                src_onnx_path=src_onnx,
                dst_onnx_path=dst_onnx,
                dst_weights_path=dst_weights,
                alignment=alignment,
                chunk_size=7,
            )
            self.assertIsInstance(total, int)
            self.assertGreater(total, 0)
            self.assertEqual(os.path.getsize(dst_weights), total)

            # 1) Inspect rewritten metadata (offsets aligned, location rewritten).
            metadata = onnxl.load(dst_onnx, load_external_data=False)
            inits = list(metadata.graph.initializer)
            self.assertEqual(len(inits), len(payloads))
            for init, arr in zip(inits, payloads):
                self.assertEqual(int(init.data_location), int(onnxl.TensorProto.EXTERNAL))
                entries = {e.key: e.value for e in init.external_data}
                self.assertEqual(str(entries["location"]), os.path.basename(dst_weights))
                offset = int(str(entries["offset"]))
                length = int(str(entries["length"]))
                self.assertEqual(offset % alignment, 0)
                self.assertEqual(length, arr.nbytes)

            # 2) Verify byte-for-byte equality by reading the destination weights file
            #    at each tensor's recorded offset/length.
            with open(dst_weights, "rb") as fh:
                weights_bytes = fh.read()
            for init, arr in zip(inits, payloads):
                entries = {e.key: e.value for e in init.external_data}
                offset = int(str(entries["offset"]))
                length = int(str(entries["length"]))
                got = np.frombuffer(weights_bytes[offset : offset + length], dtype=np.float32)
                np.testing.assert_array_equal(got, arr)

    def test_align_external_data_streaming_default_alignment_arg(self) -> None:
        from onnx_light.onnx import align_external_data_streaming

        with tempfile.TemporaryDirectory() as tdir:
            src_onnx = os.path.join(tdir, "src.onnx")
            dst_onnx = os.path.join(tdir, "dst.onnx")
            dst_weights = os.path.join(tdir, "dst.data")
            self._make_model_with_external(src_onnx)
            # Use default alignment (4096) and default chunk_size.
            total = align_external_data_streaming(src_onnx, dst_onnx, dst_weights)
            self.assertGreater(total, 0)
            metadata = onnxl.load(dst_onnx, load_external_data=False)
            for init in metadata.graph.initializer:
                entries = {e.key: e.value for e in init.external_data}
                self.assertEqual(int(str(entries["offset"])) % 4096, 0)


class TestSaveModelWithSharedExternalData(ExtTestCase):
    @staticmethod
    def _make_first_model(path: str) -> list[np.ndarray]:
        """Saves a first model with two external initializers and returns their payloads."""
        payloads = [
            np.full((7,), 1.5, dtype=np.float32),  # 28 bytes
            np.full((11,), -2.0, dtype=np.float32),  # 44 bytes
        ]
        inits = [
            oh.make_tensor(
                name=f"a{i}",
                data_type=onnxl.TensorProto.FLOAT,
                dims=arr.shape,
                vals=arr.tobytes(),
                raw=True,
            )
            for i, arr in enumerate(payloads)
        ]
        graph = oh.make_graph([], "g", [], [], initializer=inits)
        model = oh.make_model(graph, producer_name="first")
        onnxl.save(model, path, save_as_external_data=True, size_threshold=0)
        return payloads

    def test_save_model_reuses_first_model_weights(self) -> None:
        from onnx_light.onnx import save_model_with_shared_external_data
        from onnx_light.onnx_lib import SerializeOptions

        with tempfile.TemporaryDirectory() as tdir:
            src_onnx = os.path.join(tdir, "src.onnx")
            payloads_first = self._make_first_model(src_onnx)
            src_data = src_onnx + ".data"
            src_data_size = os.path.getsize(src_data)
            self.assertGreater(src_data_size, 0)

            # Build a second model that reuses every initializer from the first one
            # (loaded without external data — so external_data entries are preserved)
            # and adds two brand-new initializers with inline raw_data.
            first = onnxl.load(src_onnx, load_external_data=False)
            payloads_new = [
                np.full((5,), 7.0, dtype=np.float32),  # 20 bytes (new)
                np.full((3,), -3.5, dtype=np.float32),  # 12 bytes (new)
            ]
            new_inits = [
                oh.make_tensor(
                    name=f"n{i}",
                    data_type=onnxl.TensorProto.FLOAT,
                    dims=arr.shape,
                    vals=arr.tobytes(),
                    raw=True,
                )
                for i, arr in enumerate(payloads_new)
            ]
            second_inits = list(first.graph.initializer) + new_inits
            graph_b = oh.make_graph([], "g2", [], [], initializer=second_inits)
            second = oh.make_model(graph_b, producer_name="second")

            # Save the second model next to the first one so the reused initializers'
            # external_data.location (kept as-is) still resolves to src.onnx.data.
            dst_onnx = os.path.join(tdir, "dst.onnx")
            dst_weights = dst_onnx + ".data"
            alignment = 64
            opts = SerializeOptions()
            opts.alignment = alignment
            total = save_model_with_shared_external_data(
                model=second, dst_onnx_path=dst_onnx, options=opts
            )
            # Only the new initializers should land in dst.data (20 bytes, then 64-aligned
            # padding, then 12 bytes => 76 bytes total).
            self.assertIsInstance(total, int)
            self.assertEqual(total, 76)
            self.assertEqual(os.path.getsize(dst_weights), total)
            # The first model's data file must NOT have been modified or duplicated.
            self.assertEqual(os.path.getsize(src_data), src_data_size)

            # 1) Inspect metadata: reused initializers keep pointing at src.onnx.data
            #    (their location is preserved as-is), new ones point at dst.onnx.data
            #    with aligned offsets.
            meta = onnxl.load(dst_onnx, load_external_data=False)
            inits = list(meta.graph.initializer)
            self.assertEqual(len(inits), len(payloads_first) + len(payloads_new))
            for init in inits:
                self.assertEqual(int(init.data_location), int(onnxl.TensorProto.EXTERNAL))
                entries = {e.key: str(e.value) for e in init.external_data}
                location = str(entries["location"])
                offset = int(str(entries["offset"]))
                if str(init.name).startswith("a"):
                    self.assertEqual(location, os.path.basename(src_data))
                else:
                    self.assertEqual(location, os.path.basename(dst_weights))
                    self.assertEqual(offset % alignment, 0)

            # 2) Load the destination model with external data and verify bytes match.
            loaded = onnxl.load(dst_onnx, load_external_data=True)
            loaded_inits = list(loaded.graph.initializer)
            all_payloads = payloads_first + payloads_new
            for init, arr in zip(loaded_inits, all_payloads):
                got = np.frombuffer(init.raw_data, dtype=np.float32)
                np.testing.assert_array_equal(got, arr)

    def test_save_model_with_only_reused_weights(self) -> None:
        """No new initializer: secondary weights file is not created, return value is
        0, reused initializers' external_data entries are written out unchanged."""
        from onnx_light.onnx import save_model_with_shared_external_data

        with tempfile.TemporaryDirectory() as tdir:
            src_onnx = os.path.join(tdir, "src.onnx")
            payloads_first = self._make_first_model(src_onnx)
            first = onnxl.load(src_onnx, load_external_data=False)

            graph_b = oh.make_graph([], "g2", [], [], initializer=list(first.graph.initializer))
            second = oh.make_model(graph_b, producer_name="second")

            dst_onnx = os.path.join(tdir, "dst.onnx")
            dst_weights = dst_onnx + ".data"
            total = save_model_with_shared_external_data(second, dst_onnx)
            self.assertEqual(total, 0)
            # The secondary weights file is not created when there is nothing to write.
            self.assertFalse(os.path.exists(dst_weights))

            loaded = onnxl.load(dst_onnx, load_external_data=True)
            for init, arr in zip(loaded.graph.initializer, payloads_first):
                got = np.frombuffer(init.raw_data, dtype=np.float32)
                np.testing.assert_array_equal(got, arr)

    def test_save_model_preserves_existing_external_location_as_is(self) -> None:
        """Reused initializers' external_data entries (location, offset, length) are
        written out unchanged — no rewriting relative to the destination's directory."""
        from onnx_light.onnx import save_model_with_shared_external_data

        with tempfile.TemporaryDirectory() as tdir:
            src_onnx = os.path.join(tdir, "src.onnx")
            self._make_first_model(src_onnx)
            first = onnxl.load(src_onnx, load_external_data=False)

            # Capture the original external_data entries before saving.
            original_entries: list[dict[str, str]] = [
                {e.key: str(e.value) for e in init.external_data}
                for init in first.graph.initializer
            ]

            new_arr = np.full((2,), 9.0, dtype=np.float32)
            new_init = oh.make_tensor(
                name="n0",
                data_type=onnxl.TensorProto.FLOAT,
                dims=new_arr.shape,
                vals=new_arr.tobytes(),
                raw=True,
            )
            graph_b = oh.make_graph(
                [], "g2", [], [], initializer=[*first.graph.initializer, new_init]
            )
            second = oh.make_model(graph_b, producer_name="second")

            # Save the second model in a subdirectory: with the new API, the reused
            # initializers' location is kept exactly as it was on the first model,
            # so it still reads "src.onnx.data" (not a "../src.onnx.data" rewrite).
            sub = os.path.join(tdir, "sub")
            os.makedirs(sub)
            dst_onnx = os.path.join(sub, "dst.onnx")
            save_model_with_shared_external_data(second, dst_onnx)

            meta = onnxl.load(dst_onnx, load_external_data=False)
            reused = [init for init in meta.graph.initializer if str(init.name).startswith("a")]
            self.assertEqual(len(reused), len(original_entries))
            for init, original in zip(reused, original_entries):
                entries = {e.key: str(e.value) for e in init.external_data}
                # Every external_data entry of a reused initializer is preserved verbatim.
                self.assertEqual(entries, original)

            # The newly written initializer points at the auto-derived secondary file
            # placed next to dst.onnx, with a plain basename location.
            new_metas = [
                init for init in meta.graph.initializer if str(init.name).startswith("n")
            ]
            self.assertEqual(len(new_metas), 1)
            new_entries = {e.key: str(e.value) for e in new_metas[0].external_data}
            self.assertEqual(str(new_entries["location"]), os.path.basename(dst_onnx) + ".data")


if __name__ == "__main__":
    unittest.main(verbosity=2)
