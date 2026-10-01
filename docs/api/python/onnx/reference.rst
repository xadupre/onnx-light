onnx_light.onnx.reference
=========================

Mixed native and NumPy evaluation
---------------------------------

Use :class:`~onnx_light.onnx.reference.MixedReferenceEvaluator` when
the native ``Gemm``, ``MatMul``, ``Conv``, or ``Attention`` kernels are
too slow for a model. It has the same constructor and ``run`` API as
:class:`~onnx_light.onnx.reference.ReferenceEvaluator`, but automatically
registers NumPy implementations for these four operators in each session.
Other operators continue to use native kernels. Performance varies with
the model, tensor sizes, and the cost of crossing the NumPy/native boundary.

.. code-block:: python

    from onnx_light.onnx.reference import MixedReferenceEvaluator

    outputs = MixedReferenceEvaluator(model).run(None, feeds)

Native dtype behavior
---------------------

The compiled runtime preserves UTF-8 string initializers across repeated
evaluations, including inputs passed to custom callbacks. Explicit feeds may
override initializers for one evaluation without modifying the model or its
cached defaults. ``Compress`` supports string tensors, including empty results.

``Less`` compares ``DOUBLE`` inputs without narrowing. ``ReduceMin`` and
``ReduceMax`` retain numeric input types (and support ``BOOL``), while
``ReduceMean`` supports ``FLOAT``, ``DOUBLE``, ``FLOAT16`` and ``BFLOAT16``.
Half-precision reductions accumulate in float32; double reductions retain
float64 precision. Reductions propagate NaNs. Empty extrema reductions use
positive/negative infinity for floating-point inputs and the corresponding
integer or boolean identity; empty floating-point means produce NaN.
Axes attributes, axes inputs, ``keepdims`` and ``noop_with_empty_axes`` retain
their ONNX semantics.

``LogSoftmax`` supports ``FLOAT``, ``DOUBLE`` and ``FLOAT16`` with stable
maximum-subtracted exponentiation and float32 half-precision accumulation.
Before opset 13 it flattens the dimensions starting at ``axis`` (default 1);
from opset 13 it reduces only that axis (default -1).
``Where`` supports float16 selection without changing the selected bits,
including signed zero, and broadcasts empty dimensions.

.. automodule:: onnx_light.onnx.reference
    :members:
    :imported-members:
