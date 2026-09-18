"""Executes native ORT FlatBuffers round-trips of representative backend models.

ONNX Runtime is the independent executor of the reconstructed model. Comparing protobuf bytes
after an ORT round-trip would not be valid: ORT is a resolved runtime graph
format, not a lossless encoding of every ModelProto field.
"""

from __future__ import annotations

import unittest

import onnxruntime

import onnx_light.onnx as onnxl
from onnx_light.ext_test_case import import_or_skip

make_test_class = import_or_skip("onnx_light.onnx.backend", "make_test_class")


def ort_round_trip_check(model: onnxl.ModelProto, *inputs):
    """Executes a model round-tripped through the native ORT writer and reader."""
    options = onnxl.SerializeOptions()
    options.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
    parse_options = onnxl.ParseOptions()
    parse_options.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(model.SerializeToString(options), parse_options)
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        parsed.SerializeToString(),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    feeds = dict(zip((value.name for value in session.get_inputs()), inputs))
    return session.run(None, feeds)


TestBackendRoundTripOrtFormat = make_test_class(
    ort_round_trip_check,
    include_regex=[
        r"^test_(?:cc_)?(?:add|sub|mul|div|relu|sigmoid|tanh|matmul|gemm|concat|transpose|reshape)$"
    ],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
