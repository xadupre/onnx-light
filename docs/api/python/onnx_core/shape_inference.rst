onnx_light.onnx_core.shape_inference
====================================

``Shape`` exposes the concrete C++ runtime shape, distinct from ``SymShape``,
which can contain symbolic dimensions. It accepts positional dimensions
(``Shape(2, 3)`` or ``Shape(*dims)``), an integer sequence, or another ``Shape``, stores
at most 16 dimensions, and supports iteration, indexed mutation (including
negative indices), ``append``, ``dims`` and a checked ``product``. An empty
``Shape()`` represents a scalar; ``dims()`` and ``copy.copy`` return independent
copies.

.. code-block:: python

    from onnx_light.onnx_core.shape_inference import Shape

    shape = Shape(2, 3)
    assert shape.product() == 6
    shape[-1] = 4
    assert list(shape) == [2, 4]

.. autofunction:: onnx_light.onnx_core.shape_inference.compute_peak_memory

.. automodule:: onnx_light.onnx_core.shape_inference
    :members:
    :imported-members:
