from pathlib import Path

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh


def main() -> None:
    """Generates the deterministic external-data model fixture."""
    fixture_dir = Path(__file__).resolve().parent
    model_path = fixture_dir / "prepared_execution_external.onnx"
    data_path = fixture_dir / "prepared_execution_external.data"
    model_path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    weight = onh.from_array(np.array([1, 2, 3, 4], dtype=np.float32), name="W")
    graph = oh.make_graph(
        [oh.make_node("Identity", ["W"], ["Y"])],
        "prepared_execution_external",
        [],
        [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [4])],
        [weight],
    )
    model = oh.make_model(graph, producer_name="onnx-light")
    onnxl.save(
        model,
        model_path,
        save_as_external_data=True,
        location=data_path.name,
        size_threshold=0,
        num_threads=1,
    )


if __name__ == "__main__":
    main()
