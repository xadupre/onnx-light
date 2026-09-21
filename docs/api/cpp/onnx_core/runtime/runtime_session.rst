runtime_session.h
=================

Sessions, including subgraphs, read directly usable initializer storage from
the immutable source graph. No ownership mode is required. The graph and its
backing buffers must outlive the session and its initializer views, including
when supplied through ``SetInitializers``.

Host initializers are not copied into the execution allocator for CPU execution
(``kCPU`` or the default ``kUndefined`` device). Raw storage and native float,
double, int32, int64 and uint64 fields can be borrowed; representations requiring
decoding, including strings, still use normal tensor conversion. This host
borrowing rule does not implement transfers to another device.

The context can supply a model lifetime token for selected retained results;
ownerless externally borrowed buffers cannot be retained. Nonpersistent graph
outputs still materialize borrowed storage normally so they can outlive the model.

.. doxygenfile:: onnx_core/runtime/runtime_session.h
   :project: onnx-light
