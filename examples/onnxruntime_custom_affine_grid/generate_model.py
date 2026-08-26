"""Generates the model used by the C++ ONNX Runtime custom-op example."""

import sys

import onnx
from onnx import TensorProto, helper


def main() -> None:
    """Generates a model containing the custom AffineGrid node."""
    output = sys.argv[1] if len(sys.argv) > 1 else "affine_grid_custom.onnx"
    node = helper.make_node(
        "AffineGrid", ["theta", "size"], ["grid"], domain="com.example", align_corners=0
    )
    graph = helper.make_graph(
        [node],
        "custom-affine-grid",
        [
            helper.make_tensor_value_info("theta", TensorProto.FLOAT, [1, 2, 3]),
            helper.make_tensor_value_info("size", TensorProto.INT64, [4]),
        ],
        [helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 256, 256, 2])],
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.example", 1)],
    )
    onnx.save(model, output)


if __name__ == "__main__":
    main()
