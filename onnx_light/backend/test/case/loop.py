from __future__ import annotations

from typing import Any

import numpy as np

from .... import onnx
from .base import Base, expect


class Loop(Base):
    @staticmethod
    def export_loop_13() -> None:
        # Given a tensor x of values [x1, ..., xN],
        # Return a sequence of tensors of
        #   [[x1], [x1, x2], ..., [x1, ..., xN]]

        seq_in = onnx.helper.make_tensor_sequence_value_info(
            "seq_in", onnx.TensorProto.FLOAT, None
        )
        seq_out = onnx.helper.make_tensor_sequence_value_info(
            "seq_out", onnx.TensorProto.FLOAT, None
        )
        cond_in = onnx.helper.make_tensor_value_info(
            "cond_in", onnx.TensorProto.BOOL, []
        )
        cond_out = onnx.helper.make_tensor_value_info(
            "cond_out", onnx.TensorProto.BOOL, []
        )
        iter_count = onnx.helper.make_tensor_value_info(
            "iter_count", onnx.TensorProto.INT64, []
        )

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        x_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["x"],
            value=onnx.helper.make_tensor(
                name="const_tensor_x",
                data_type=onnx.TensorProto.FLOAT,
                dims=x.shape,
                vals=x.flatten().astype(float),
            ),
        )

        one_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one"],
            value=onnx.helper.make_tensor(
                name="const_tensor_one",
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[1],
            ),
        )

        zero_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["slice_start"],
            value=onnx.helper.make_tensor(
                name="const_tensor_zero",
                data_type=onnx.TensorProto.INT64,
                dims=(1,),
                vals=[0],
            ),
        )

        axes_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes"],
            value=onnx.helper.make_tensor(
                name="const_tensor_axes",
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[0],
            ),
        )

        add_node = onnx.helper.make_node(
            "Add", inputs=["iter_count", "one"], outputs=["end"]
        )

        end_unsqueeze_node = onnx.helper.make_node(
            "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
        )

        slice_node = onnx.helper.make_node(
            "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
        )

        insert_node = onnx.helper.make_node(
            "SequenceInsert", inputs=["seq_in", "slice_out"], outputs=["seq_out"]
        )

        identity_node = onnx.helper.make_node(
            "Identity", inputs=["cond_in"], outputs=["cond_out"]
        )

        loop_body = onnx.helper.make_graph(
            [
                identity_node,
                x_const_node,
                one_const_node,
                zero_const_node,
                add_node,
                axes_node,
                end_unsqueeze_node,
                slice_node,
                insert_node,
            ],
            "loop_body",
            [iter_count, cond_in, seq_in],
            [cond_out, seq_out],
        )

        node = onnx.helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "seq_empty"],
            outputs=["seq_res"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        seq_empty: list[Any] = []
        seq_res = [x[: int(i)] for i in x]
        cond = np.array(1).astype(bool)
        expect(
            node,
            inputs=[trip_count, cond, seq_empty],
            outputs=[seq_res],
            name="test_loop13_seq",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
            input_type_protos=[
                onnx.helper.make_tensor_type_proto(
                    onnx.TensorProto.INT64, trip_count.shape
                ),
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, cond.shape),
                onnx.helper.make_sequence_type_proto(
                    onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
                ),
            ],
        )
