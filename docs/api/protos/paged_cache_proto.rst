===============
PagedCacheProto
===============

Contains ordered KV pages. ``GraphProto.paged_cache_initializer`` embeds a
named cache as an initializer. Each page selects a dense or encoded payload
independently for K and V. Referenced type and quantization catalogues remain
model-scoped. See :doc:`../../howto/persistent_feedback`.

``PagedAttention`` declares ``past`` and ``present`` as structured values.
``PagedCacheProto`` is their serialized page-value representation and carries
the dynamic page sequence directly. The runtime ``RuntimeValue`` sequence is
not a single fixed-layout ``EncodedValueProto``.

.. autoclass:: onnx_light.onnx.PagedCacheProto
    :members:

.. autoclass:: onnx_light.onnx.PagedCacheBlockProto
    :members:
