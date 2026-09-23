persistent_value_state
======================

Each call follows four steps:

1. Binds retained values to the declared persistent inputs and adds current feeds.
2. Runs the session, reusing eligible append buffers or allocating new ones.
3. Validates outputs and checks cancellation.
4. Publishes the selected outputs as the next state only on success.

Views share tensor data but keep separate names and shapes. In-place append
writes only the new region; growth may copy the old prefix. Failure leaves
the previous logical state unchanged.

Bindings
--------

* ``model.graph.persistent_bindings`` maps whole outputs to next-call inputs.
  Each persistent input must have exactly one value-use.
* Names are literal: ``request.cache`` is a graph name, not a field path.
  Duplicate bindings and partial structured inputs are rejected.
* Current feeds cannot override retained inputs, even with an empty field map.
* ``Values()`` uses input names; returned outputs use output names.
* Supported values are tensors, named structures and inline structured encodings.
  Persistent strings are rejected, including nested strings.

Ownership
---------

* The model stays immutable; declarations are resolved once, without cloning it.
  A shared-model constructor or ``model_owner`` retains its lifetime.
  Otherwise, the caller keeps it alive through all exported model-backed views.
* ``initial`` and ``Reset`` take maps by value in C++. Use ``std::move(initial)``
  to avoid copying owned payloads.
* Returned views share owners and remain valid after later calls, reset or close.
  Callers must not modify shared payloads; publication does not undo such writes.
* Retained borrows need owners, and arena allocations need self-owning I/O leases.
  Unsafe storage is rejected, not copied as a fallback.
* All calls use the same allocators, kept alive until the state closes.
  ``RetainOwner`` can retain model/context owners before execution, including
  failed attempts. Owners must not own the state itself.
* Model-backed initializer views retain the constructor's model owner.
  Borrowed raw data additionally needs its own storage owner; ordinary kernel
  results cannot use the model owner to extend their lifetime.

Execution and lifecycle
-----------------------

* Execution uses the ordinary ``RuntimeSession`` and cached kernels.
  Only declared persistent outputs bypass normal output materialization.
* ``If`` and local functions transport selected outputs without payload copies.
  Other operators retain their ordinary computation and transport costs.
* ``Reset(initial)`` replaces retained values but keeps the prepared session.
* ``Close()`` releases retained values and kernels and permanently closes the state.
* Events use ``context.events()`` when ``events_enabled`` is true, including
  work before failure. ``ClearEvents()`` clears this log, not the persistent values.

See :doc:`../../../../howto/persistent_feedback` for usage and storage details,
and :doc:`runtime_context` for event fields.

.. doxygenfile:: onnx_core/runtime/persistent_value_state.h
    :project: onnx-light
