# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks native constant folding independently of evaluator import order."""

import importlib.util
import subprocess
import sys
import textwrap
import unittest

HAS_KERNELS = importlib.util.find_spec("onnx_light.onnx_py._onnxpykernels") is not None


class TestGraphBuilderKernelImport(unittest.TestCase):
    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_shape_concat_folds_without_evaluator_import(self):
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            class BlockEvaluator(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if (
                        fullname == "onnx" or fullname.startswith("onnx.")
                        or fullname.startswith("onnx_light._reference")
                        or fullname == "onnx_light.onnx.reference"
                    ):
                        raise AssertionError("Evaluator import forbidden: " + fullname)
                    return None

            sys.meta_path.insert(0, BlockEvaluator())
            import numpy
            from onnx_light.onnx_core.graph_builder import GraphBuilder
            from onnx_light.onnx import TensorProto, numpy_helper

            builder = GraphBuilder("fresh_process")
            builder.make_input("x", TensorProto.FLOAT, [6])
            builder.make_initializer(numpy_helper.from_array(numpy.array([2]), "a"))
            builder.make_initializer(numpy_helper.from_array(numpy.array([3]), "b"))
            builder.make_node("Concat", ["a", "b"], outputs=["shape"], attributes={"axis": 0})
            builder.make_node("Reshape", ["x", "shape"], outputs=["y"])
            builder.make_output("y")
            assert builder.constant_fold() == 1
            model = builder.to_onnx()
            assert not any(node.op_type == "Concat" for node in model.graph.node)
            assert any(
                tensor.name == "shape"
                and numpy.array_equal(numpy_helper.to_array(tensor), [2, 3])
                for tensor in model.graph.initializer
            )
            assert "onnx_light.onnx_py._onnxpykernels" in sys.modules
            assert not any(name.startswith("onnx_light._reference") for name in sys.modules)
            """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipIf(HAS_KERNELS, "Exercises a reduced build without compiled kernels.")
    def test_reduced_builder_import_and_missing_folding_kernel(self):
        script = textwrap.dedent("""
            import sys
            import unittest
            import numpy
            from onnx_light.onnx_core.graph_builder import GraphBuilder
            from onnx_light.onnx import TensorProto, numpy_helper

            builder = GraphBuilder("reduced")
            builder.make_input("x", TensorProto.FLOAT, [6])
            builder.make_initializer(numpy_helper.from_array(numpy.array([2]), "a"))
            builder.make_initializer(numpy_helper.from_array(numpy.array([3]), "b"))
            builder.make_node("Concat", ["a", "b"], outputs=["shape"], attributes={"axis": 0})
            builder.make_node("Reshape", ["x", "shape"], outputs=["y"])
            builder.make_output("y")
            assert len(builder.to_onnx().graph.node) == 2
            assert "onnx_light.onnx_py._onnxpykernels" not in sys.modules
            with unittest.TestCase().assertRaisesRegex((ValueError, RuntimeError), "Concat"):
                builder.constant_fold()
            assert not any(name.startswith("onnx_light._reference") for name in sys.modules)
            """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
