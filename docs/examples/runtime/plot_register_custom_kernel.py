"""
.. _l-example-plot-register-custom-kernel:

Replace a built-in kernel with a Python one and prove it ran
============================================================

This is the Python counterpart of the standalone C++ example
``examples/register_custom_kernel`` (see
:ref:`l-cpp-register-custom-kernel-example`). The C++ example writes a brand-new
:cpp:class:`~onnx_light::core::runtime::KernelBase` subclass for the existing
``Abs`` operator, installs it into the shared dispatch table and checks — via a
run counter — that the custom kernel, not the built-in one, executed the node.

Here the same scenario is expressed in Python:

* a one-node ``y = Abs(x)`` model is parsed,
* a numpy-friendly implementation of ``Abs`` is registered through
  :meth:`~onnx_light.onnx.reference.ReferenceEvaluator.register_custom_kernel`
  under the default ONNX domain, overriding the built-in kernel,
* the model runs and the example asserts both that ``y == |x|`` and that the
  custom kernel — bumping a shared run counter on every call — was the one
  dispatched.

It also prints the custom kernel's ``"<library>:<device>:<domain>:<op_type>"``
identifier next to the official built-in one and checks the two differ.
"""

from __future__ import annotations

import numpy as np

from onnx_light.onnx_lib import parser
from onnx_light.onnx.reference import ReferenceEvaluator

#####################################
# Build a one-node ``Abs`` model
# ++++++++++++++++++++++++++++++
#
# ``Abs`` is a built-in operator, so without an override the built-in kernel
# would compute the result.

model = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 18]>'
    "agraph (float[4] x) => (float[4] y) { y = Abs(x) }"
)
print(model)

#####################################
# Implement the custom kernel
# +++++++++++++++++++++++++++
#
# The callable receives the :class:`NodeProto` followed by one
# ``numpy.ndarray`` per input and returns the element-wise absolute value. A
# module-level counter records every invocation so the example can prove the
# custom kernel — and not the built-in ``Abs`` — actually ran.

run_count = 0


# Built-in kernels expose a ``"<library>:<device>:<domain>:<op_type>"``
# identifier (see the C++ classes, e.g. the official ``Abs`` kernel is named
# ``"onnx_kernels:CPU:ai.onnx:Abs"``). The custom kernel below advertises its
# own name under a distinct ``example`` library prefix so it never collides
# with — and is clearly distinguishable from — the built-in one.
OFFICIAL_ABS_KERNEL_NAME = "onnx_kernels:CPU:ai.onnx:Abs"
CUSTOM_ABS_KERNEL_NAME = "example:CPU:ai.onnx:Abs"


def custom_abs(node, x):
    global run_count
    run_count += 1
    return np.abs(x)


custom_abs.name = CUSTOM_ABS_KERNEL_NAME

#####################################
# Show the custom kernel name differs from the official one
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Both follow the same ``"<library>:<device>:<domain>:<op_type>"`` convention
# but use different library prefixes, so the override is unmistakably *not* the
# built-in ``Abs`` kernel.

print(f"official Abs kernel name: {OFFICIAL_ABS_KERNEL_NAME}")
print(f"custom   Abs kernel name: {custom_abs.name}")
assert custom_abs.name != OFFICIAL_ABS_KERNEL_NAME, (
    "the custom kernel name must differ from the official one, otherwise it "
    "would be indistinguishable from the built-in kernel."
)
print("OK: the custom kernel name is different from the official one.")

#####################################
# Register the override and run
# +++++++++++++++++++++++++++++
#
# Registering under the default ONNX domain (the empty string is normalised to
# ``ai.onnx``) takes precedence over the built-in ``Abs`` entry.

sess = ReferenceEvaluator(model)
sess.register_custom_kernel("", "Abs", custom_abs)

x = np.array([-1.0, 2.0, -3.5, 0.0], dtype=np.float32)
(y,) = sess.run(None, {"x": x})
print(f"y = {y}")

#####################################
# Verify the output and that the custom kernel ran
# ++++++++++++++++++++++++++++++++++++++++++++++++
#
# The output must equal ``|x|`` and the run counter must show the custom kernel
# was dispatched exactly once, proving the override replaced the built-in.

np.testing.assert_allclose(y, np.abs(x))
assert run_count == 1, (
    f"expected the custom Abs kernel to run exactly once, but it ran "
    f"{run_count} time(s); the built-in kernel was probably dispatched instead."
)
print(f"PASS: the custom 'Abs' kernel ran {run_count} time(s) and produced |x|.")

#####################################
# See also
# ++++++++
#
# * :ref:`l-example-plot-custom-kernel` for a broader tour of custom and
#   user-defined-domain kernels in Python.
# * :ref:`l-cpp-register-custom-kernel-example` for the equivalent standalone C++
#   example.
