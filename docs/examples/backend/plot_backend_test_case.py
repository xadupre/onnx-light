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
* displays the reference input and output tensors,
* demonstrates how to use :func:`make_test_class` from
  :mod:`onnx_light.backend.test.case` to systematically validate
  shape inference across many backend test cases.
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


#####################################
# Systematic testing with make_test_class
# ++++++++++++++++++++++++++++++++++++++++
#
# For comprehensive validation, you can use
# :func:`~onnx_light.backend.test.case.make_test_class` to create a test
# class that runs shape inference on every backend test case.
#
# The function :func:`make_test_class` accepts a callable that receives a
# model (and optionally input data sets). Here we define a simple validator
# that runs :func:`infer_shapes_model` and checks that the inferred shapes
# for intermediate tensors match the expected ``value_info`` stored in each
# test case.

import onnx_light.onnx as onnxl  # noqa: E402
from onnx_light.onnx_optim.shape_inference import infer_shapes_model  # noqa: E402
from onnx_light.backend.test.case import make_test_class  # noqa: F401, E402


def validate_shape_inference(model_with_expected_shapes: onnxl.ModelProto):
    """
    Validates that infer_shapes_model reproduces the expected value_info.

    Test cases in the backend test suite often include pre-computed
    ``value_info`` entries that represent the "ground truth" for shape
    inference. This function:

    1. Creates a copy of the model.
    2. Clears the copy's ``value_info`` to simulate a model with no intermediate shapes.
    3. Invokes :func:`infer_shapes_model` to re-infer shapes.
    4. Compares the inferred shapes against the original ``value_info``.

    Note: This simple check only verifies that tensor ranks match. A production
    test might enforce exact symbolic or numeric dimension matches.

    Raises:
        AssertionError: If the inferred ranks do not match the expected ranks,
            indicating a regression in the shape inference logic.
    """
    # Keep a reference to the expected value_info.
    expected_value_info = {vi.name: vi for vi in model_with_expected_shapes.graph.value_info}

    # Make a working copy and clear its value_info.
    work = onnxl.ModelProto()
    work.CopyFrom(model_with_expected_shapes)
    work.graph.value_info.clear()

    # Run shape inference.
    infer_shapes_model(work)

    # Compare inferred shapes to the expected shapes.
    inferred_value_info = {vi.name: vi for vi in work.graph.value_info}
    for name, expected in expected_value_info.items():
        if name not in inferred_value_info:
            raise AssertionError(f"Shape inference did not produce value_info for {name!r}")
        inferred = inferred_value_info[name]
        if not expected.type.tensor_type or not inferred.type.tensor_type:
            continue  # Skip non-tensor types.
        exp_shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in expected.type.tensor_type.shape.dim
        )
        inf_shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in inferred.type.tensor_type.shape.dim
        )
        # For this simple check, we only verify that ranks match.
        # A production test might enforce exact symbolic/numeric dimension matches.
        if len(exp_shape) != len(inf_shape):
            raise AssertionError(
                f"Shape mismatch for {name!r}: expected rank {len(exp_shape)}, "
                f"got rank {len(inf_shape)}"
            )


# Uncomment the lines below to create a test class and run it with pytest or unittest.
# This example demonstrates the pattern; in practice, you would run this in a
# separate test file.
#
# TestShapeInferenceBackend = make_test_class(
#     validate_shape_inference,
#     include_regex=["test_abs", "test_add"],  # Filter to a small subset for demo.
# )
#
# if __name__ == "__main__":
#     import unittest
#     unittest.main(verbosity=2)
#
# print(
#     "\nTo systematically test shape inference across all backend cases, "
#     "use make_test_class as shown above."
# )

# Quick inline demonstration: validate the model we just retrieved.
print("\nDemonstrating validation on the test_abs model:")
try:
    validate_shape_inference(tc.model)
    print("  ✓ Validation succeeded (no AssertionError raised)")
except AssertionError as e:
    print(f"  ✗ Validation failed: {e}")
