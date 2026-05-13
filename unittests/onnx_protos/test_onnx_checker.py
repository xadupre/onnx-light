# source: https://github.com/onnx/onnx/blob/main/onnx/test/checker_test.py
import unittest
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.pychecker as pychecker


class TestChecker(unittest.TestCase):
    def make_sparse(
        self,
        shape: tuple[int, ...],
        values: list[int],
        indices_shape: tuple[int, ...],
        indices: list[int],
    ) -> onnxl.SparseTensorProto:
        sparse = onnxl.SparseTensorProto()
        sparse.dims.extend(shape)
        nnz = len(values)
        sparse.values.CopyFrom(oh.make_tensor("spval", onnxl.TensorProto.INT64, (nnz,), values))
        sparse.indices.CopyFrom(
            oh.make_tensor("spind", onnxl.TensorProto.INT64, indices_shape, indices)
        )
        return sparse

    def test_check_attribute(self) -> None:
        attr = onnxl.AttributeProto()
        attr.name = "test"
        attr.i = 2
        pychecker.check_attribute(attr)

    def test_check_attribute_fails_without_value(self) -> None:
        attr = onnxl.AttributeProto()
        attr.name = "test"
        with self.assertRaises(pychecker.ValidationError):
            pychecker.check_attribute(attr)

    def test_check_attribute_fails_with_two_values(self) -> None:
        attr = onnxl.AttributeProto()
        attr.name = "test"
        attr.i = 2
        attr.f = 1.0
        with self.assertRaises(pychecker.ValidationError):
            pychecker.check_attribute(attr)

    def test_check_sparse_tensor(self) -> None:
        sparse = self.make_sparse((2, 3), [1, 2], (2, 2), [0, 1, 1, 2])
        pychecker.check_sparse_tensor(sparse)

    def test_check_sparse_tensor_invalid_shape(self) -> None:
        sparse = self.make_sparse((6,), [1, 2], (2,), [0, 5])
        with self.assertRaises(pychecker.ValidationError):
            pychecker.check_sparse_tensor(sparse)

    def test_check_model_metadata_props(self) -> None:
        node = oh.make_node("Relu", ["X"], ["Y"])
        graph = oh.make_graph(
            [node],
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1, 2])],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1, 2])],
        )
        model_def = oh.make_model(graph, producer_name="test")
        oh.set_model_props(model_def, {"Title": "my graph", "Keywords": "test;graph"})
        pychecker.check_model(model_def)
        dupe = model_def.metadata_props.add()
        dupe.key = "Title"
        dupe.value = "Other"
        with self.assertRaises(pychecker.ValidationError):
            pychecker.check_model(model_def)


if __name__ == "__main__":
    unittest.main(verbosity=2)
