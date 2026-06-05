"""
.. _l-example-plot-backend-test-case:

Retrieve a backend test case and display its model and data
===========================================================

:mod:`onnx_light.backend.test` exposes the ONNX backend node test
suite (model + input/output reference tensors). Each entry is a
:class:`onnx_light.backend.test.case.base.TestCase` returned by
:func:`onnx_light.backend.test.case.collect_test_case`.

This example picks one of these test cases (``test_abs``) and shows:

* the ``ModelProto`` that defines the graph,
* the reference input tensors,
* the reference output tensors.
"""

from __future__ import annotations

from onnx_light.backend.test.case import collect_test_case

#####################################
# Retrieve a specific test case
# +++++++++++++++++++++++++++++
#
# :func:`collect_test_case` accepts an optional ``name`` argument to
# return a single :class:`TestCase` directly. The ``test_abs`` case
# exercises the ``Abs`` operator on a small ``float32`` tensor.

tc = collect_test_case("test_abs")
print(f"name      : {tc.name}")
print(f"model_name: {tc.model_name}")
print(f"kind      : {tc.kind}")
print(f"rtol/atol : {tc.rtol} / {tc.atol}")

#####################################
# Display the model
# +++++++++++++++++
#
# The ``model`` attribute is a :class:`ModelProto`. Its textual
# representation lists the opset imports and the graph (inputs,
# outputs, nodes).

print(tc.model)

#####################################
# Display the inputs and outputs
# ++++++++++++++++++++++++++++++
#
# ``data_sets`` is a sequence of ``(inputs, outputs)`` pairs of
# numpy arrays. Node tests typically ship a single reference data
# set.

assert tc.data_sets is not None
for ds_idx, (inputs, outputs) in enumerate(tc.data_sets):
    print(f"-- data set #{ds_idx} --")
    for i, x in enumerate(inputs):
        print(f"  input[{i}]: dtype={x.dtype}, shape={tuple(x.shape)}")
        print(x)
    for i, y in enumerate(outputs):
        print(f"  output[{i}]: dtype={y.dtype}, shape={tuple(y.shape)}")
        print(y)
