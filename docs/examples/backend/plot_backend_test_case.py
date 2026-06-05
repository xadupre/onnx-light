"""
.. _l-example-plot-backend-test-case:

Retrieve a backend test case and display its model and data
===========================================================

The ONNX backend test suite shipped with ``onnx-light`` is exposed
as a small C++ data model bound to Python under
:mod:`onnx_light.backend_test`. Each entry is a ``TestCase`` made of
a ``ModelProto`` plus one or more ``DataSet`` (lists of reference
input / output ``Tensor`` instances).

:func:`onnx_light.backend_test.collect_test_cases` returns the C++
test cases as a ``list``. When called with an operator type it
returns only the cases whose top-level graph contains a node with
that ``op_type``.

This example:

* retrieves the ``test_abs`` case via ``collect_test_cases("Abs")``,
* displays its ``ModelProto``,
* displays the reference input and output tensors.
"""

from __future__ import annotations

import numpy as np

from onnx_light.backend_test import collect_test_cases

#####################################
# Retrieve a backend test case
# ++++++++++++++++++++++++++++
#
# ``collect_test_cases("Abs")`` returns every backend test case that
# exercises the ``Abs`` operator. We pick the canonical ``test_abs``
# case from the result.

abs_cases = collect_test_cases("Abs")
print(f"Number of Abs cases: {len(abs_cases)}")
print(f"Names              : {[tc.name for tc in abs_cases]}")

tc = next(tc for tc in abs_cases if tc.name == "test_abs")
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
# ``data_sets`` is a list of ``DataSet`` objects. Each ``DataSet``
# exposes ``inputs`` and ``outputs`` as lists of ``Tensor`` whose
# ``raw_data`` bytes are stored in row-major little-endian layout.
# We decode the float32 buffer to a numpy array for display.

_DTYPE_TO_NP = {1: np.float32}  # ``Abs`` test case uses float32


def _to_numpy(t):
    dtype = _DTYPE_TO_NP[int(t.data_type)]
    return np.frombuffer(t.raw_data(), dtype=dtype).reshape(tuple(int(d) for d in t.shape))


for ds_idx, ds in enumerate(tc.data_sets):
    print(f"-- data set #{ds_idx} --")
    for i, x in enumerate(ds.inputs):
        arr = _to_numpy(x)
        print(f"  input[{i}]: dtype={arr.dtype}, shape={tuple(arr.shape)}")
        print(arr)
    for i, y in enumerate(ds.outputs):
        arr = _to_numpy(y)
        print(f"  output[{i}]: dtype={arr.dtype}, shape={tuple(arr.shape)}")
        print(arr)
