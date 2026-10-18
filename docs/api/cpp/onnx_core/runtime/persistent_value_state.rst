persistent_value_state
======================

``RuntimeSession`` owns its passive ``PersistentValueState`` and reuses it between
calls. Construct a session with initial retained values, then call
``RuntimeSession::Run(context, feeds)`` without passing a separate state:

1. Uses retained values for persistent inputs alongside current feeds.
2. Runs the session, reusing eligible append buffers or allocating new ones.
3. Validates outputs and checks cancellation.
4. Publishes the selected outputs as the next state only on success.

Views share tensor data but keep separate names and shapes. In-place append
writes only the new region; growth may copy the old prefix. Failure leaves
the previous logical state unchanged.

Bindings
--------

* ``model.graph.persistent_bindings`` maps whole outputs to next-call inputs.
  Each persistent input must have at least one value-use.
* Names are literal: ``request.cache`` is a graph name, not a field path.
  Duplicate bindings and partial structured inputs are rejected.
* Current feeds cannot override retained inputs, even with an empty field map.
* ``Values()`` uses input names; returned outputs use output names.
* Supported values are tensors, named structures and inline structured encodings.
  Persistent strings are rejected, including nested strings.

Ownership
---------

* The model stays immutable; declarations are resolved once, without cloning it.
  C++ constructors borrow ``const ModelProto&``. The caller keeps it alive
  through the session and all exported model-backed views.
  Python bindings retain the model for sessions and exported initializer views.
* ``initial`` and ``Reset`` take maps by value in C++. Use ``std::move(initial)``
  to avoid copying owned payloads.
* Returned views share owners and remain valid after later calls, reset or session
  destruction. Callers must not modify shared payloads; publication does not undo
  such writes.
* Retained borrows need owners, except views of immutable initializer-owned
  storage whose lifetime is guaranteed by the C++ caller. Arena allocations
  need self-owning I/O leases. Unsafe storage is rejected, not copied as a fallback.
* All calls use the same allocators, kept alive until the session is destroyed.
* Initializers borrowing external raw data still need their own storage owner.
  Model lifetime alone cannot extend the lifetime of external storage or
  ordinary kernel results.

Execution and lifecycle
-----------------------

* Execution uses the supplied ``RuntimeContext`` directly and the caller-owned
  ``RuntimeSession`` and its cached kernels. No child context is created.
  The state exposes no context-binding or execution methods; the session handles
  inputs, validation and publication as part of ``Run``.
  Only declared persistent outputs bypass normal output materialization.
* ``If`` and local functions transport selected outputs without payload copies.
  Other operators retain their ordinary computation and transport costs.
* ``RuntimeSession::Reset(initial)`` replaces retained values without rebuilding
  the session or its cached kernels.
* Session destruction releases its retained values and cached kernels.
* Events use ``context.events()`` when ``events_enabled`` is true, including
  work before failure. ``ClearEvents()`` clears this log, not the persistent values.

See :doc:`../../../../howto/cpp/persistent_feedback` for usage and storage details,
and :doc:`runtime_context` for event fields.

.. doxygenfile:: onnx_core/runtime/persistent_value_state.h
    :project: onnx-light
