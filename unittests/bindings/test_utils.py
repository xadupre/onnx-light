# Copyright (c) ONNX Project Contributors
# Adapted from https://github.com/onnx/onnx/blob/main/onnx/test/utils_test.py
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import utils


class TestUtilityFunctions(ExtTestCase):
    def test_extract_model(self) -> None:
        def create_tensor(name):
            return oh.make_tensor_value_info(name, onnxl.TensorProto.FLOAT, [1, 2])

        A0 = create_tensor("A0")
        A1 = create_tensor("A1")
        B0 = create_tensor("B0")
        B1 = create_tensor("B1")
        B2 = create_tensor("B2")
        C0 = create_tensor("C0")
        C1 = create_tensor("C1")
        D0 = create_tensor("D0")
        L0_0 = oh.make_node("Add", ["A0", "A1"], ["B0"])
        L0_1 = oh.make_node("Sub", ["A0", "A1"], ["B1"])
        L0_2 = oh.make_node("Mul", ["A0", "A1"], ["B2"])
        L1_0 = oh.make_node("Add", ["B0", "B1"], ["C0"])
        L1_1 = oh.make_node("Sub", ["B1", "B2"], ["C1"])
        L2_0 = oh.make_node("Mul", ["C0", "C1"], ["D0"])

        # onnx_light has no whole-model shape inference; supply value_info explicitly
        g0 = oh.make_graph(
            [L0_0, L0_1, L0_2, L1_0, L1_1, L2_0],
            "test",
            [A0, A1],
            [D0],
            value_info=[B0, B1, B2, C0, C1],
        )
        m0 = oh.make_model(g0, producer_name="test")
        tdir = tempfile.mkdtemp()
        p0 = os.path.join(tdir, "original.onnx")
        onnxl.save(m0, p0)

        p1 = os.path.join(tdir, "extracted.onnx")
        input_names = ["B0", "B1", "B2"]
        output_names = ["C0", "C1"]
        utils.extract_model(p0, p1, input_names, output_names)

        m1 = onnxl.load(p1)
        self.assertEqual(str(m1.producer_name), "onnx_light.utils.extract_model")
        self.assertEqual(m1.ir_version, m0.ir_version)
        self.assertEqual(len(m1.opset_import), len(m0.opset_import))
        self.assertEqual(len(m1.graph.node), 2)
        self.assertEqual(len(m1.graph.input), 3)
        self.assertEqual(len(m1.graph.output), 2)
        self.assertEqual(str(m1.graph.input[0].name), "B0")
        self.assertEqual(str(m1.graph.input[1].name), "B1")
        self.assertEqual(str(m1.graph.input[2].name), "B2")
        self.assertEqual(str(m1.graph.output[0].name), "C0")
        self.assertEqual(str(m1.graph.output[1].name), "C1")
        shutil.rmtree(tdir, ignore_errors=True)

    def test_tar_members_filter_rejects_sibling_prefix_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            base = os.path.join(tdir, "model")
            os.mkdir(base)
            tar_path = os.path.join(tdir, "payload.tar")

            with tarfile.open(tar_path, "w") as tar:
                payload = b"outside extraction root"
                info = tarfile.TarInfo("../model_evil/pwned.txt")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            with (
                tarfile.open(tar_path) as tar,
                self.assertRaisesRegex(RuntimeError, "directory traversal"),
            ):
                utils._tar_members_filter(tar, base)


if __name__ == "__main__":
    unittest.main()
