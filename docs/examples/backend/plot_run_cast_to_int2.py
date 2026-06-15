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
  :func:`onnx_light.onnx.backend.collect_test_cases_by_name`,
* displays its single-node ``Cast`` ``ModelProto``,
* runs the model with
  :class:`onnx_light.onnx.reference.ReferenceEvaluator` and prints the
  resulting ``INT2`` tensor,
* shows how to run the corresponding backend test by building a tiny
  runtime function and feeding it to
  :func:`onnx_light.onnx.backend.make_test_class`.
"""

from __future__ import annotations

import numpy as np

from onnx_light.onnx.backend import collect_test_cases_by_name
from onnx_light.onnx.reference import ReferenceEvaluator

#####################################
# Retrieve the float-to-int2 cast case
# ++++++++++++++++++++++++++++++++++++
#
# ``collect_test_cases_by_name`` filters the registered backend test
# cases by a regular expression matched against ``TestCase.name``. We
# anchor the pattern with ``^...$`` to select the single
# ``test_cc_cast_FLOAT_to_INT2`` case.

cases = collect_test_cases_by_name("^test_cc_cast_FLOAT_to_INT2$")
print(f"Number of matching cases: {len(cases)}")

tc = cases[0]
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

print(tc.model)

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
print(f"output dtype: {output.dtype}")
print(f"output shape: {output.shape}")
print("output (int2, saturated to [-2, 1]):")
# Cast to int8 only for a readable decimal print of the 2-bit values.
print(output.astype(np.int8).ravel())

#####################################
# Run the corresponding backend test
# ++++++++++++++++++++++++++++++++++
#
# The same case can be executed as a backend test. A backend test only
# needs a runtime callable with the signature
# ``rt(model, *inputs) -> list[np.ndarray]``;
# :func:`~onnx_light.onnx.backend.make_test_class` then builds a
# :class:`unittest.TestCase` subclass with one ``test_<name>`` method
# per collected case. We restrict the collection to our single case with
# ``include_regex``.

import unittest  # noqa: E402

from onnx_light.onnx.backend import make_test_class  # noqa: E402


def reference_runtime(model, *inputs: np.ndarray) -> list[np.ndarray]:
    """Runs *model* on *inputs* with the reference runtime.

    Returns the model outputs as a list of numpy arrays, in graph-output
    order, as expected by :func:`make_test_class`.
    """
    sess = ReferenceEvaluator(model)
    feeds = {i.name: arr for i, arr in zip(model.graph.input, inputs)}
    return sess.run(None, feeds)


CastToInt2BackendTest = make_test_class(
    reference_runtime, include_regex=[r"^test_cc_cast_FLOAT_to_INT2$"]
)

suite = unittest.TestLoader().loadTestsFromTestCase(CastToInt2BackendTest)
print(f"Number of backend tests collected: {suite.countTestCases()}")
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(f"Backend test successful: {result.wasSuccessful()}")

#####################################
# Running the test from the command line
# ++++++++++++++++++++++++++++++++++++++
#
# In practice you would place the ``reference_runtime`` /
# ``make_test_class`` snippet in its own test file and run it with
# pytest or unittest, optionally narrowing to this case with ``-k``::
#
#     python -m pytest my_backend_tests.py -v -k cast_FLOAT_to_INT2
#
# See :ref:`l-design-backend-tests` for the full backend-test workflow.

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
