onnx_light.onnx_core.tensor_statistics
======================================

.. automodule:: onnx_light.onnx_core.tensor_statistics
    :members:
    :imported-members:

For example:

.. code-block:: python

    import numpy
    from onnx_light.onnx_core.tensor_statistics import compute_tensor_statistics

    statistics = compute_tensor_statistics(
        numpy.arange(24, dtype=numpy.float32).reshape(4, 6),
        quantiles=[0.25, 0.75],
        median=True,
        rectangle_shape=[2, 3],
    )
    assert statistics.mean == 11.5
    assert statistics.median == 11.5
    rectangle = numpy.array(statistics.rectangle_map).reshape(statistics.rectangle_shape)

For quantized values, statistics describe reconstructed values, while
``size_bytes`` reports encoded payload storage rather than a temporary
floating-point tensor. Quantization scales and other descriptor metadata are
not included in this payload size.
