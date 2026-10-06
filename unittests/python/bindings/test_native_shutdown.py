# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks that Python-owned native callbacks release their bindings at shutdown."""

import importlib.util
import subprocess
import sys
import textwrap
import unittest


class TestNativeShutdown(unittest.TestCase):
    def assert_clean_exit(self, script):
        """Checks repeated process exits for native leak diagnostics."""
        for run in range(3):
            with self.subTest(run=run):
                result = subprocess.run(
                    [sys.executable, "-c", textwrap.dedent(script)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("nanobind: leaked", result.stderr)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx_light.onnx_py._onnxpygradient") is not None,
        "Requires the full native gradient extension.",
    )
    def test_gradient_callback_cycle(self):
        script = """
            import gc
            from onnx_light.onnx_py._onnxpygradient import (
                GradRegistry, register_gradient_function
            )

            def register():
                registry = GradRegistry()
                def callback(*args):
                    return registry
                register_gradient_function("test", "Op", callback, registry)

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    def test_pattern_graph_cycle(self):
        script = """
            import gc
            from onnx_light.onnx_py._onnxpycore import builder

            class PythonPattern(builder.PatternOptimization):
                def match(self, graph, node):
                    return self.no_match(node, "unused")
                def apply(self, graph, nodes):
                    return []

            def register():
                owner = builder.GraphBuilder("g")
                pattern = PythonPattern(name="Custom")
                graph = builder.GraphGraph(owner, [pattern])
                pattern.graph = graph

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx_light.onnx_py._onnxpykernels") is not None,
        "Requires the full native runtime extension.",
    )
    def test_runtime_callback_cycle(self):
        script = """
            import gc
            from onnx_light.onnx_py._onnxpykernels import runtime

            def register():
                context = runtime.RuntimeContext()
                def callback(node, ctx):
                    return context
                context.register_custom_kernel("test", "Op", callback)

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    def test_builder_schema_callback_cycle(self):
        script = """
            import gc
            from onnx_light.onnx_py._onnxpycore import builder

            def register():
                owner = None
                def lookup(op_type):
                    return [owner]
                owner = builder.GraphBuilder("g", lookup)
                owner.make_subgraph("body")
                owner.make_local_function("local", "test")

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    def test_callback_finalizer_registers_another_owner(self):
        script = """
            from onnx_light.onnx_py._onnxpyprotoop import ParseOptions

            class AddAnother:
                def __init__(self, owner):
                    self.owner = owner

                def __call__(self, node, graph):
                    pass

                def __del__(self):
                    next_owner = ParseOptions()
                    next_owner.node_callback = lambda node, graph: next_owner

            def register():
                owner = ParseOptions()
                owner.node_callback = AddAnother(owner)

            register()
            """
        self.assert_clean_exit(script)

    def test_builder_embedded_shape_callback(self):
        script = """
            import gc
            from onnx_light.onnx_py._onnxpycore import builder

            def register():
                owner = builder.GraphBuilder("g")
                owner.shapes.set_custom_shape_inference_function(
                    "test", "Op", lambda ctx, node: owner
                )

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    def test_callback_finalizer_reregisters_same_owner(self):
        script = """
            from onnx_light.onnx_py._onnxpyprotoop import ParseOptions

            class ReRegister:
                def __init__(self, owner):
                    self.owner = owner

                def __call__(self, node, graph):
                    pass

                def __del__(self):
                    self.owner.node_callback = lambda node, graph: self.owner

            def register():
                owner = ParseOptions()
                owner.node_callback = ReRegister(owner)

            register()
            """
        self.assert_clean_exit(script)

    def test_proto_callback_cycles(self):
        script = """
            import gc
            from onnx_light.onnx_py._onnxpyprotoop import (
                ParseOptions, RawDataCallback, SerializeOptions
            )

            def register():
                parse = ParseOptions()
                parse.node_callback = lambda node, graph: parse
                parse_raw = ParseOptions()
                parse_raw.raw_data_callback = lambda tensor, graph: parse_raw
                serialize = SerializeOptions()
                serialize.raw_data_callback = lambda tensor, graph, buffer, size_only: serialize
                serialize_node = SerializeOptions()
                serialize_node.node_callback = lambda node, graph: serialize_node
                raw = RawDataCallback()
                raw.on_tensor = lambda tensor: raw
                raw_ctor = RawDataCallback(lambda tensor: raw_ctor)

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    def test_value_tag_callback_cycle(self):
        script = """
            import gc
            from onnx_light.onnx_py._onnxpycore import shape_inference

            def register():
                context = shape_inference.ComputeContext()
                def callback(ctx, node, index):
                    return context
                context.set_custom_value_tag_function("test", "Op", callback)

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    def test_schema_callback_cycle(self):
        script = """
            import gc
            from onnx_light.onnx_lib import defs

            def register():
                schema = defs.OpSchema("Lifetime", "test", 1)
                schema.set_type_and_shape_inference_function(
                    lambda context, owner=schema: None
                )

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    def test_registered_schema_callback_cycle(self):
        script = """
            import gc
            from onnx_light.onnx_lib import defs

            def register():
                schema = defs.OpSchema("ShutdownCallback", "test.shutdown", 1)
                schema.set_type_and_shape_inference_function(
                    lambda context, owner=schema: None
                )
                defs.register_schema(schema)

            register()
            gc.collect()
            """
        self.assert_clean_exit(script)

    def test_shape_callback_cycle(self):
        script = """
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
            """
        self.assert_clean_exit(script)


if __name__ == "__main__":
    unittest.main()
