# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks that Python-owned native callbacks release their bindings at shutdown."""

import subprocess
import sys
import textwrap
import unittest


class TestNativeShutdown(unittest.TestCase):
    def test_shape_callback_cycle(self):
        script = textwrap.dedent("""
            import gc
            from onnx_light.onnx_py._onnxpycore import shape_inference
            from onnx_light.onnx_py._onnxpyprotoop import FunctionProto
            from onnx_light.onnx_py._onnxpypatterns import ConcatGatherPattern
            from onnx_light.onnx_core.graph_builder import GraphBuilder

            def register():
                context = shape_inference.ShapesContext()
                function = FunctionProto()
                builder = GraphBuilder("g")
                pattern = ConcatGatherPattern()

                def callback(ctx, node):
                    return context, function, builder, pattern

                context.set_custom_shape_inference_function("test", "Op", callback)

            register()
            gc.collect()
            """)
        for _ in range(3):
            with self.subTest(run=_):
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("nanobind: leaked", result.stderr)


if __name__ == "__main__":
    unittest.main()
