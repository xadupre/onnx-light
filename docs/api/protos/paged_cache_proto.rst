===============
PagedCacheProto
===============

Contains ordered KV pages. ``GraphProto.paged_cache_initializer`` embeds a
named cache as an initializer. Each page selects a dense or encoded payload
independently for K and V. Referenced type and quantization catalogues remain
model-scoped. See :doc:`../../howto/persistent_feedback`.

.. autoclass:: onnx_light.onnx.PagedCacheProto
    :members:

.. autoclass:: onnx_light.onnx.PagedCacheBlockProto
    :members:
