onnx_light.onnx.shape_inference
===============================

Custom operator schemas can register a Python callback with
``OpSchema.set_type_and_shape_inference_function(callback)`` before calling
``onnx_light.onnx.defs.register_schema(schema)``. The registered schema retains
the callback even after local Python references are released.

The callback receives an :class:`InferenceContext`. Its ``get_input_type(index)``
method returns a copy of the input type (or ``None`` when unavailable), and
``set_output_type(index, type_proto)`` copies a type to an output. The context is
only valid during the callback and must not be retained.

.. automodule:: onnx_light.onnx.shape_inference
    :members:
    :imported-members:
