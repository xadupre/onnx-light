feedback_state
==============

``FeedbackState(model, initial, options)`` reads only the root graph's
``persistent_bindings``. It resolves declarations once. The C++ model and its
nodes must remain immutable and alive until the state is closed or destroyed;
the state neither clones nor serializes the model to detect later changes.
The reference-only constructor also requires the model to outlive all exported
model-backed views. A ``shared_ptr<const ModelProto>`` constructor retains this
lifetime automatically. Bindings can instead pass a fourth ``model_owner``
lifetime-token argument to the reference constructor; this token must retain the
immutable model and any externally borrowed model storage. Only owner handles,
not model or tensor payloads, are copied.
Borrowed initializer ``raw_data`` must additionally carry its own backing-owner
control block. Retaining the model cannot make an ownerless external buffer safe;
such initializers are rejected instead of copied.

Initial values, current feeds and ``Values()`` use exact graph input names.
Dots and backslashes are literal, with no escaping or path resolution.
``request.cache`` selects only the graph input literally named ``request.cache``,
never a field of ``request``. Each binding retains a whole input from a whole
output; duplicate input or output selections are rejected.

Nested field maps represent complete structured values beneath a graph name.
Partial root maps cannot combine current fields with retained fields.
Supplying any value for a retained input is an error, even an empty field map.
``Values()`` returns complete values keyed by retained graph input names.
Returned graph-output dictionaries use their literal model output names.

Initial values, reset values, invocation inputs, selected outputs, and
``Values()`` share payload owners. Their field maps and tensor metadata are
independent, but their payload addresses are identical. Callers and kernels
must treat shared numeric buffers, strings, and encoded messages as read-only.
Publication is transactional for owner handles, not a rollback mechanism for
mutating aliases. First-time storage promotion must not race another operation
on the same source value.

Old output views remain valid after subsequent invocations, reset, or close.
Inline buffers move into shared owners; borrowed buffers must supply a lifetime
token. I/O-arena allocations acquire a self-owning lease. Ownerless retained
borrows and execution-arena allocations without such a lease are rejected,
never copied as a fallback. C++ callers must keep the first context's allocators
alive until the state is closed.
Bindings can call ``FeedbackState::RetainOwner`` with external model/context
lifetime tokens before execution. These owners survive failed initialization
and are released only after retained values and kernels are destroyed.
Reusing the same shared token avoids accumulating duplicate owners. Tokens must
not own the state itself.

The ownership-preserving execution path also transports values through ``If``
and stateless model-local functions. It rejects ``Loop``, ``Scan``, and function
attribute-reference binding rather than using their legacy copying transports.
Feedback initializers are read-only model views: native float, double, int32, int64 and
uint64 typed fields and raw storage are supported. Other numeric typed initializer
representations require conversion to raw storage before constructing the
immutable model. String initializers are not supported in this mode; string
inputs and retained string outputs are supported. Ordinary ``RuntimeSession`` construction and public
``SetInitializers`` retain their independent, eager initializer snapshots; the
borrowed mode is selected explicitly for feedback and its subgraphs.
``RuntimeContext::set_model_owner`` can supply a lifetime token for the immutable
model in an explicitly borrowed runtime session. Feedback always supplies its
own constructor's model token, not an unrelated calling context's token.
Known initializer views retain this token
(including through functions and ``If``), so their output aliases remain valid after the context and state are
destroyed. The token must keep the model and its borrowed storage alive. It is
not attached to arbitrary kernel outputs, which must provide their own owners.
A reference-only caller supplies a lifetime promise rather than an owning token:
it must keep the model alive until all of these views are released.

.. doxygenfile:: onnx_core/runtime/feedback_state.h
    :project: onnx_light
