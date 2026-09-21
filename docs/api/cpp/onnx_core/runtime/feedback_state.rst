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
an ownerless initializer view is rejected if selected for retention. Ordinary
kernel reads and nonpersistent outputs keep their normal behavior.

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

The C++ constructor and ``Reset`` take initial maps by value. Move an owned
map (``std::move(initial)``) to transfer its payload without copying. Passing an
lvalue uses its normal C++ copy semantics; explicitly owner-backed borrowed
tensors and encoded messages share their owners. Neither operation changes a
const source. ``std::move(tensor).RetainStorage()`` explicitly consumes a tensor
and returns an owner-retaining view; ``BorrowView()`` never promotes storage.
Invocation inputs, selected outputs and ``Values()`` share payload owners.
Their field maps and tensor metadata are independent. Callers and kernels
must treat shared numeric buffers and encoded messages as read-only.
Publication is transactional for owner handles, not a rollback mechanism for
mutating aliases.

String tensors cannot be persistent, directly or inside a selected whole
structure or encoded layout (including constant fields and catalogue references).
The shared graph validator rejects these declarations before initialization.
Runtime retention also rejects strings; it never copies them as a fallback.
Nonpersistent string feeds and outputs keep ordinary materialized storage.

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

There is no context-wide persistence mode or alternative kernel dispatcher.
Only the exact declared output names bypass ordinary allocator migration and
output materialization. Their whole values are moved into retained ownership
after validation. Other outputs keep normal allocation/materialization rules;
ordinary feeds may borrow storage for the invocation without an owner.
``If`` and local functions translate the selected names to their formal
outputs, then move those results back. Other outputs retain normal transport.
``Loop``, ``Scan`` and function attribute references use the ordinary runtime;
persistence adds no blanket prohibition. Tensor ``Identity`` remains an ordinary
computing kernel, not a persistence-specific aliasing kernel.

All runtime sessions use read-only model views for initializer raw storage and native
float, double, int32, int64 and uint64 typed fields. Other representations,
including strings, use normal tensor conversion. CPU execution does not migrate
initializers into the execution allocator. The source graph and its storage must
remain immutable and alive, also for subgraphs and public ``SetInitializers``.
``RuntimeContext::set_model_owner`` can supply a lifetime token for the immutable
model in a runtime session. Feedback always supplies its
own constructor's model token, not an unrelated calling context's token.
Known initializer views retain this token
(including through functions and ``If``), so their output aliases remain valid after the context and state are
destroyed. The token must keep the model and its borrowed storage alive. It is
not attached to arbitrary kernel outputs, which must provide their own owners.
A reference-only caller supplies a lifetime promise rather than an owning token:
it must keep the model alive until all of these views are released.

.. doxygenfile:: onnx_core/runtime/feedback_state.h
    :project: onnx-light
