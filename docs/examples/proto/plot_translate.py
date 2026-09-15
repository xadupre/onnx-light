"""
.. _l-example-plot-translate:

translate: turn an ONNX model back into Python code
====================================================

:func:`~onnx_light.tools.translate` converts an existing ``ModelProto``
(or ``GraphProto``) into Python code that rebuilds an equivalent model.
Two *flavours* are available:

* ``api="onnx-compact"`` — a single nested
  :mod:`onnx_light.onnx.helper` expression
  (``oh.make_model(oh.make_graph([...], ...))``).
* ``api="builder"`` — an incremental script driving the
  :class:`~onnx_light.onnx_core.graph_builder.GraphBuilder`
  (``g.inp(...)``, ``g.init(...)``, ``g.op.<operator>(...)``,
  ``g.out(...)``, ``g.to_onnx(...)``).
* ``api="cpp"`` — a C++ function that rebuilds the model with
  :cpp:class:`~onnx_light::core::builder::GraphBuilder`.

:func:`~onnx_light.tools.translate_header` returns the matching import
header, so ``translate_header(api) + translate(model, api)`` is a fully
runnable Python snippet.  The example below builds a small model, prints
both flavours and then executes the generated code to rebuild the model.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/translate.png"

from __future__ import annotations

import numpy as np

import onnx_light.onnx as onnx
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.tools import translate, translate_header

# Built-in operator schemas are registered so the rebuilt models validate.
defs.register_onnx_operator_set_schema()


#####################################
# Build the model
# +++++++++++++++
#
# A tiny graph ``Y = Add(Mul(X, W), B)`` with two initializers so the
# translation exercises nodes, inputs/outputs and initializers.

model = oh.make_model(
    oh.make_graph(
        [oh.make_node("Mul", ["X", "W"], ["XW"]), oh.make_node("Add", ["XW", "B"], ["Y"])],
        "linear",
        [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["N", 3])],
        [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["N", 3])],
        [
            onh.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="W"),
            onh.from_array(np.array([0.5, 0.5, 0.5], dtype=np.float32), name="B"),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=8,
)


#####################################
# onnx-compact flavour
# ++++++++++++++++++++
#
# ``translate_header("onnx-compact")`` returns the imports and
# ``translate(model, api="onnx-compact")`` the nested ``oh.make_model``
# expression.

compact_code = translate_header("onnx-compact") + translate(model, api="onnx-compact")
print("=== onnx-compact ===")
print(compact_code)


#####################################
# builder flavour
# +++++++++++++++
#
# The ``builder`` flavour rebuilds the same model step by step with the compact
# :class:`~onnx_light.onnx_core.graph_builder.GraphBuilder` API
# (``g.inp``, ``g.init``, ``g.op`` and ``g.out``).

builder_code = translate_header("builder") + translate(model, api="builder")
print("\n=== builder ===")
print(builder_code)


#####################################
# C++ flavour
# +++++++++++
#
# The ``cpp`` flavour emits a standalone ``BuildModel`` function using the
# native :cpp:class:`~onnx_light::core::builder::GraphBuilder`.

cpp_code = translate_header("cpp") + translate(model, api="cpp")
print("\n=== cpp ===")
print(cpp_code)


#####################################
# Round-trip
# ++++++++++
#
# The generated code is plain Python: executing it rebuilds an equivalent
# model.  Here we run the ``builder`` snippet and check that the
# rebuilt graph has the same nodes as the original.

namespace: dict = {}
exec(builder_code, namespace)  # noqa: S102
rebuilt = namespace["model"]

original_ops = [node.op_type for node in model.graph.node]
rebuilt_ops = [node.op_type for node in rebuilt.graph.node]
print("\n=== round-trip ===")
print("original ops:", original_ops)
print("rebuilt ops :", rebuilt_ops)
assert original_ops == rebuilt_ops
