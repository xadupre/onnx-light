===============
PagedCacheProto
===============

Contains ordered KV pages. ``GraphProto.paged_cache_initializer`` embeds a
named cache as an initializer. Each page selects a dense or encoded payload
independently for K and V. Referenced type and quantization catalogues remain
model-scoped. See :doc:`../../howto/persistent_feedback`.

The logical declaration is the independent, named
``onnx_light.PagedKVCache`` version 1 type returned by
``onnx_light.onnx.PagedKVCacheTypeV1()``. It describes a structure with a
dynamic page sequence; it does not contain state. ``PagedCacheProto`` is the
serialized page-value representation. The runtime ``RuntimeValue`` sequence is
not a single fixed-layout ``EncodedValueProto``.

.. autoclass:: onnx_light.onnx.PagedCacheProto
    :members:

.. autoclass:: onnx_light.onnx.PagedCacheBlockProto
    :members:
