"""Generates the model used by the C++ ONNX Runtime custom-op example."""

import sys

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto


def main() -> None:
    """Generates a model containing the custom AffineGrid node."""
    output = sys.argv[1] if len(sys.argv) > 1 else "affine_grid_custom.onnx"
    node = oh.make_node(
        "AffineGrid", ["theta", "size"], ["grid"], domain="com.example", align_corners=0
    )
    graph = oh.make_graph(
        [node],
        "custom-affine-grid",
        [
            oh.make_tensor_value_info("theta", TensorProto.FLOAT, [1, 2, 3]),
            oh.make_tensor_value_info("size", TensorProto.INT64, [4]),
        ],
        [oh.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 256, 256, 2])],
    )
    model = oh.make_model(
        graph,
        ir_version=10,
        opset_imports=[oh.make_opsetid("", 20), oh.make_opsetid("com.example", 1)],
    )
    onnxl.save(model, output)


if __name__ == "__main__":
    main()
