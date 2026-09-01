"""
.. _l-example-plot-custom-kernel:

Extend ReferenceEvaluator with a custom kernel
==============================================

:class:`~onnx_light.onnx.reference.ReferenceEvaluator` dispatches every
``NodeProto`` against the static C++ ``KernelDispatchTable``. Any operator
that is not built in — typically an operator from a user-defined domain,
or a temporary stand-in for an op that is not yet implemented — would
otherwise fail with ``unsupported op_type``.

The
:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.register_custom_kernel`
hook makes it possible to plug a Python callable into the runtime without
touching the C++ dispatch table. The callable is invoked as
``fn(node, *numpy_inputs)`` and must return either a single
:class:`numpy.ndarray` or a tuple/list of arrays for multi-output ops.
Registrations are stored on the evaluator's persistent ``RuntimeContext``.
Registering or unregistering
a kernel invalidates cached runtime sessions, and the next
:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.run` call recreates
them, so the same evaluator can be reused safely across runs.

This example:

* parses a small ONNX graph that calls ``my.domain.Scale`` — an operator
  that is *not* part of the built-in dispatch table,
* registers a numpy-friendly Python implementation through
  :meth:`~onnx_light.onnx.reference.ReferenceEvaluator.register_custom_kernel`,
* runs the model and prints the result.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/custom_kernel.png"

from __future__ import annotations

import numpy as np

from onnx_light.onnx_lib import parser
from onnx_light.onnx.reference import ReferenceEvaluator

#####################################
# Build a model that uses a custom op
# +++++++++++++++++++++++++++++++++++
#
# ``my.domain.Scale`` is a user-defined operator: it multiplies its
# single input by a ``factor`` attribute. The ONNX parser accepts the
# unknown op as long as the domain is declared in ``opset_import``.

model = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>'
    "agraph (float[3] x) => (float[3] y) {"
    "  y = my.domain.Scale<factor=3.0>(x)"
    "}"
)
print(model)

#####################################
# Implement the custom kernel
# +++++++++++++++++++++++++++
#
# The callable receives the :class:`NodeProto` (so it can read attributes)
# followed by one ``numpy.ndarray`` per declared input. It returns either
# a single array or a tuple/list of arrays — one per declared output.


def scale(node, x):
    factor = next(a for a in node.attribute if str(a.name) == "factor").f
    return x * float(factor)


#####################################
# Register and run
# ++++++++++++++++
#
# :meth:`register_custom_kernel` is a wrapper around the low-level
# :py:meth:`RuntimeContext.register_custom_kernel` binding. The
# registration is stored on the evaluator. Registering or unregistering
# invalidates cached runtime sessions, and the next
# :meth:`ReferenceEvaluator.run` recreates them, so the same evaluator can
# run multiple inputs without re-registering.

sess = ReferenceEvaluator(model)
sess.register_custom_kernel("my.domain", "Scale", scale)

x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
(y,) = sess.run(None, {"x": x})
print(f"y = {y}")

#####################################
# Overriding a built-in kernel
# ++++++++++++++++++++++++++++
#
# A custom registration under the default ONNX domain takes precedence
# over the entry that ``KernelDispatchTable`` would otherwise dispatch.
# This is convenient to instrument or replace a specific kernel without
# patching the C++ runtime. Below ``Abs`` is replaced by negation just
# to demonstrate the override mechanism.

override_model = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 18]>'
    "agraph (float[3] x) => (float[3] y) { y = Abs(x) }"
)


def fake_abs(node, x):
    return -x


sess2 = ReferenceEvaluator(override_model)
sess2.register_custom_kernel("", "Abs", fake_abs)
(y2,) = sess2.run(None, {"x": np.array([-1.0, -2.0, -3.0], dtype=np.float32)})
print(f"Abs replaced by negation: y = {y2}")

#####################################
# Registering globally instead of per session
# ++++++++++++++++++++++++++++++++++++++++++++
#
# The registrations above only affect the evaluator they are called on. A
# kernel can instead be registered *globally* so that every evaluator created
# afterwards picks it up, without registering it on each one. A per-session
# registration still overrides a global one for the same ``(domain, op_type)``.
# Register the global kernel before running the evaluators that should use it,
# since an evaluator caches its runtime sessions on first ``run``.

ReferenceEvaluator.register_custom_kernel_global("my.domain", "Scale", scale)
sess3 = ReferenceEvaluator(model)  # no per-session registration needed
(y3,) = sess3.run(None, {"x": x})
print(f"y (global kernel) = {y3}")
ReferenceEvaluator.unregister_custom_kernel_global("my.domain", "Scale")

#####################################
# See also
# ++++++++
#
# * :ref:`l-howto-use-custom-kernel` for how to register custom kernels,
#   including the matching low-level Python
#   (``RuntimeContext.register_custom_kernel``) and C++
#   (``RuntimeContext::RegisterCustomKernel``) entry points.
