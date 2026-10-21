========================
ExtendedInitializerProto
========================

Wraps one nonstandard graph initializer. Its ``value`` oneof contains either
``encoded_value`` (:doc:`encoded_value_proto`, field 1) or ``paged_cache``
(:doc:`paged_cache_proto`, field 2). Selecting one alternative clears the other;
graph validation rejects entries without a payload.

.. code-block:: python

    from onnx_light import onnx

    entry = onnx.ExtendedInitializerProto(encoded_value=encoded)
    model.graph.extended_initializer.append(entry)
    assert entry.WhichOneof("value") == "encoded_value"

The initializer name belongs to the selected payload, not to the wrapper.
Type and quantization references remain scoped to the containing model.
Native copies retain raw payload owners without a serialization round trip,
including dense and encoded key/value pages in a paged cache;
``CopyFrom`` replaces the selected alternative, including when copying an
empty message.

.. autoclass:: onnx_light.onnx.ExtendedInitializerProto
    :members:
