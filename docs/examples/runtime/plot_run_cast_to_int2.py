"""
.. _l-example-plot-run-cast-to-int2:

Run an ONNX model casting a float tensor into an int2 tensor
============================================================

This example shows how to take one backend test case from the suite
shipped with ``onnx-light`` (:mod:`onnx_light.onnx.backend`), run its
ONNX model with the reference runtime, and then re-run the *same* case
as a backend test.

The case we use is ``test_cc_cast_FLOAT_to_INT2``: a single ``Cast``
node that converts a ``float32`` tensor into a 2-bit signed integer
tensor (``INT2``). ``INT2`` is a sub-byte dtype: its representable
range is ``[-2, 1]`` and values outside that range saturate, which is
visible in the runtime output below.

This example:

* retrieves the ``test_cc_cast_FLOAT_to_INT2`` case via
  :func:`onnx_light.onnx.backend.get_test_case`,
* displays its single-node ``Cast`` ``ModelProto``,
* runs the model with
  :class:`onnx_light.onnx.reference.ReferenceEvaluator` and prints the
  resulting ``INT2`` tensor,
* shows how to run the corresponding backend test by passing a tiny
  runtime function to the case's ``assert_allclose`` method (and notes
  the :func:`onnx_light.onnx.backend.make_test_class` helper for running
  the whole registry).
"""

from __future__ import annotations

import numpy as np

from onnx_light.onnx.backend import get_test_case
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.tools import pretty_onnx

#####################################
# Retrieve the float-to-int2 cast case
# ++++++++++++++++++++++++++++++++++++
#
# ``get_test_case`` retrieves a single backend test case by exact name,
# using the C++ name-based filter for efficiency.

tc = get_test_case("test_cc_cast_FLOAT_to_INT2")
print(f"name      : {tc.name}")
print(f"model_name: {tc.model_name}")
print(f"kind      : {tc.kind}")

#####################################
# Display the model
# +++++++++++++++++
#
# The model is a single ``Cast`` node. The ``to`` attribute is ``26``,
# the ONNX ``TensorProto.INT2`` data type. The graph output is declared
# with ``elem_type: 26`` accordingly.

print(pretty_onnx(tc.model))

#####################################
# Run the model with the reference runtime
# ++++++++++++++++++++++++++++++++++++++++
#
# :class:`~onnx_light.onnx.reference.ReferenceEvaluator` runs the model
# with the C++ reference kernels. The input is the same
# ``np.arange(-3, 4)`` float32 sweep the test case uses, reshaped to the
# model's ``(7, 1)`` input shape. The runtime returns an ``INT2`` numpy
# array (backed by ``ml_dtypes.int2``); values below ``-2`` or above
# ``1`` saturate to the representable range.

session = ReferenceEvaluator(tc.model)

x = np.arange(-3, 4, dtype=np.float32).reshape(7, 1)
print("input (float32):")
print(x.ravel())

output = session.run(None, {"input": x})[0]
print(f"output type: {type(output)}")
print(f"output dtype: {output.dtype}")
print(f"output shape: {output.shape}")
print("output (int2, saturated to [-2, 1]):")
# Cast to int8 only for a readable decimal print of the 2-bit values.
print(output.astype(np.int8).ravel())

#####################################
# Run the corresponding backend test
# ++++++++++++++++++++++++++++++++++
#
# The same case retrieved with ``get_test_case`` *is* the backend
# test: every :class:`TestCase` carries the reference input/output data
# sets and an :meth:`~onnx_light.onnx_lib.backend.test.case.base.TestCase.assert_allclose`
# method. A backend test only needs a runtime callable with the
# signature ``rt(model, *inputs) -> list[np.ndarray]``; ``assert_allclose``
# feeds each data set through it and compares the outputs against the
# expected tensors (using the case ``atol`` / ``rtol``).


def reference_runtime(model, *inputs: np.ndarray) -> list[np.ndarray]:
    """Runs *model* on *inputs* with the reference runtime.

    Returns:
        The model outputs as a list of numpy arrays, in graph-output
        order, as expected by ``TestCase.assert_allclose``.
    """
    sess = ReferenceEvaluator(model)
    feeds = {i.name: arr for i, arr in zip(model.graph.input, inputs)}
    return sess.run(None, feeds)


# ``tc`` was retrieved above from ``get_test_case()``; run its
# backend test directly. ``assert_allclose`` raises an ``AssertionError``
# on a mismatch and returns ``None`` on success.
tc.assert_allclose(reference_runtime)
print(f"Backend test {tc.name!r} passed.")

#####################################
# Running every backend test from the command line
# +++++++++++++++++++++++++++++++++++++++++++++++++
#
# To turn the whole registry into a :class:`unittest.TestCase` (one
# ``test_<name>`` method per collected case), pass the same runtime to
# :func:`~onnx_light.onnx.backend.make_test_class`, which calls
# ``collect_test_case`` internally. In practice you would place this in
# its own test file and run it with pytest or unittest, optionally
# narrowing to this case with ``-k``::
#
#     from onnx_light.onnx.backend import make_test_class
#
#     MyBackendTests = make_test_class(reference_runtime)
#
#     # python -m pytest my_backend_tests.py -v -k cast_FLOAT_to_INT2
#
# See :doc:`/api/cpp/onnx_extensions/backend_test/index` for the full backend-test workflow.

#####################################
# Gallery thumbnail
# +++++++++++++++++
#
# Render a simple text figure used as the sphinx-gallery thumbnail for
# this example.

import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(4, 3))
ax.text(0.5, 0.5, "float\n\u2192\nint2", ha="center", va="center", fontsize=28)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("plot_run_cast_to_int2.png")
