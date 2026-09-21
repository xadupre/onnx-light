.. _l-howto-persistent-feedback:

Persistent input/output feedback
===============================

Feedback state retains selected model outputs as inputs for the next call.
The graph declares the relationship in ``GraphProto.persistent_bindings``;
types come from its final input/output declarations. The caller supplies
initial values, not a separate feedback mapping. Execution uses the existing
session and allocator infrastructure, not a separate executor.

The native :cpp:class:`onnx_light::core::runtime::FeedbackState` uses the
existing runtime execution and value ownership contracts. Create one state
per independent request. Python initialization/reset, C++ ownership transfer,
state forwarding and state-value access retain buffer owners without copying payloads.
Shapes, metadata and owner handles may be copied. Kernels can allocate new
computed results; the state layer does not duplicate those results merely
to retain or return them. In-place KV append and capacity management are
separate optimizations.

.. warning::

   Inputs, retained state and returned views can share the same payload.
   Callers and kernels must not modify that payload while shared or retained.
   ``state.values`` is a shared view, not an independently mutable snapshot.
   Metadata such as a returned tensor's name and shape is independent.

The model is immutable for the entire bound session lifetime. State creation
and execution do not serialize, clone or hash the model to check for changes.
To rewrite the graph, create a new state/session after rewriting instead.

A basic feedback loop
---------------------

The Python binding is available from the native runtime module:

.. code-block:: python

    import numpy
    from onnx_light.onnx_lib import parser
    from onnx_light.onnx_py._onnxpykernels import runtime

    model = parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 18]>'
        "accumulate (float[2] delta, float[2] past) => (float[2] present)"
        "{ present = Add(delta, past) }"
    )
    binding = model.graph.persistent_bindings.add()
    binding.input_name = "past"
    binding.output_name = "present"
    context = runtime.RuntimeContext(
        runtime.KernelContext(runtime.default_opset(18))
    )
    initial = numpy.zeros(2, dtype=numpy.float32)
    state = runtime.FeedbackState(model, initial={"past": initial})
    delta = numpy.ones(2, dtype=numpy.float32)
    first = state.run(context, {"delta": delta})
    second = state.run(context, {"delta": delta})
    numpy.testing.assert_array_equal(
        numpy.from_dlpack(second["present"]),
        [2, 2],
    )
    state.reset({"past": initial})
    state.close()

The mapping is **whole input destination to whole output source**. An input
is entirely persistent or entirely supplied by current feeds, never partly
both. Binding names and initial/current-feed/``state.values`` dictionary keys
are exact, literal graph names, with no path syntax or escaping.
``"request.cache"`` names the graph input literally named ``request.cache``;
it cannot select a field of ``request``. Unknown names are rejected.
Nested dictionaries represent complete structured values, not partial feeds.

Inputs accept runtime ``Tensor`` objects, contiguous CPU NumPy arrays and
compatible DLPack producers such as CPU PyTorch tensors. Noncontiguous,
byte-swapped or unsupported representations raise an error rather than
being copied. PyTorch tensors requiring gradients must be detached by the
caller before DLPack export; ``detach()`` shares storage.

Selected outputs and ``state.values`` contain tensors with retained storage owners.
They remain valid after callers drop their input references, after another
run, or after reset/close. Unsupported ownerless output storage is rejected
rather than silently copied for a selected output; an allocator cannot recycle a live retained
allocation.
The binding keeps the model alive. The supplied context configures allocators
and custom kernels; each call executes in a fresh child context, rather than
leaving old feeds or intermediate values in the caller's context.
The binding also retains supplied contexts so cached kernel allocator
references remain valid. Reuse the same context for a state's calls.
Initializers use model-backed views when their representation is directly
readable; other numeric representations and strings use normal conversion.
String tensors cannot be persistent, including nested tensor fields and string
constants in a selected whole structure or encoded layout. Catalogue references
are checked recursively. Declarations fail before state initialization, and
runtime retention rejects string payloads rather than copying them.
Ordinary nonpersistent string feeds and outputs remain supported and use normal
materialized string storage.

Persistence applies only to the declared whole outputs, not the entire context.
Nonpersistent outputs keep normal allocator and materialization behavior.
There is no alternate kernel dispatcher: for example, tensor ``Identity`` still
computes an ordinary output rather than promising to alias its input.

The corresponding C++ entry points are:

.. code-block:: cpp

    using namespace onnx_light::core::runtime;

    auto *binding = model.mutable_graph()->add_persistent_bindings();
    binding->set_input_name("past");
    binding->set_output_name("present");
    FeedbackState state(
        model,
        {{"past", RuntimeValue(Tensor::FromFloat("past", {2}, {0.f, 0.f}))}});
    RuntimeContext context(KernelContext(18));
    auto outputs = state.Run(
        context,
        {{"delta", RuntimeValue(Tensor::FromFloat("delta", {2}, {1.f, 1.f}))}});
    state.Reset(
        {{"past", RuntimeValue(Tensor::FromFloat("past", {2}, {0.f, 0.f}))}});
    state.Close();

The C++ constructor and ``Reset`` accept initial maps by value. Build an owned
``RuntimeValueMap`` and pass ``std::move(initial)`` for zero-copy transfer.
Passing an lvalue uses ordinary C++ copy semantics. To share an existing owned
tensor explicitly, use ``std::move(tensor).RetainStorage()`` once, then
``BorrowView()`` on the returned owner-backed view. Const reads never move or
promote storage.

C++ callers register the operator kernels as usual before executing the model
(see :doc:`register_builtin_operators`). The model and configured allocators
must outlive the state/session using them. With the reference-based C++
constructor, the model must also outlive any returned views of its initializer
storage. Use the ``shared_ptr<const ModelProto>`` constructor to retain the
model automatically in such views; the Python binding retains its model
automatically.

Lifecycle and validation
------------------------

* Construct the state **after** graph rewrites. Every graph-declared input and output
  must exist in the final model, with compatible types and shape constraints.
  A removed or changed input/output needs an updated graph binding and a new state.
* Supply initial contents for every feedback destination. Each subsequent
  call supplies the remaining whole inputs; current feeds must not override
  retained inputs. Duplicate binding input names or output names are rejected.
* State advances only after successful execution and validation of the next
  values. A failed or cancelled call must not publish a partial update.
* Reset explicitly supplies new initial contents. Closing releases retained
  values. Calls, resets and closes on the same state must not overlap.
* Independent states have separate state containers. They can explicitly share
  read-only input storage; neither may mutate a shared buffer.

Publication replaces owner handles atomically after validation. Failure or
cancellation leaves the old state available; it does not require backup
copies. This guarantee does not roll back external writes that violate the
read-only contract. Persistent declarations currently belong to the root
graph; declarations inside control-flow subgraphs are rejected.

Cancellation uses the existing task-completion primitive:

.. code-block:: python

    completion = runtime.TaskCompletion()
    completion.cancel("request no longer needed")
    # state.run(context, feeds, completion) now rejects the cancelled call.

A completion is single-use. A pending completion can also be cancelled from
another thread while a call runs. Already executing kernels finish normally,
but cancellation winning the publication race prevents the state update.
Successful publication completes the token, so cancelling it afterwards is
an error. A failed call leaves the previous valid state available for retry.
Execution releases the Python GIL; use independent contexts for concurrent
requests, and do not mutate model/context configuration during a call.

Structured feedback
-------------------

A binding retains a whole input, including every dynamic field of a structured
value. To retain a cache while supplying fresh tokens, declare ``cache`` and
``tokens`` as separate graph inputs and bind ``cache`` to a whole ``next_cache``
output. The struct declaration comes from
``StructTypeProto`` in the model; persistence is a property of the feedback
declaration in ``GraphProto``, not a flag on the type or encoded payload.

This is equivalent to manually taking the selected outputs from each
stateless invocation and passing them to the next invocation. Separate unselected
outputs, such as logits, are not part of retained state. Fields inside a selected
output are all retained.

Python represents named structs as nested dictionaries. For a model declaring
structured ``cache`` and ``next_cache`` values and a separate ``tokens`` input:

.. code-block:: python

    binding = model.graph.persistent_bindings.add()
    binding.input_name = "cache"
    binding.output_name = "next_cache"
    state = runtime.FeedbackState(model, {"cache": initial_cache})
    output = state.run(context, {"tokens": tokens})

Custom kernels use ``context.get_value(name)`` and
``context.put_value(name, value)`` to exchange structured values; ordinary
tensor kernels continue to use the existing tensor API. In C++, structured
and encoded edges live in ``RuntimeContext::values()`` as ``RuntimeValue``
objects containing existing tensors or ``EncodedValueProto`` payloads.
Inline structured encoded payloads can be retained as whole values; external
payloads must first be loaded. This API currently supports tensors, named
structs and inline structured encodings, not sequence/map/optional state.
``If`` and model-local functions forward selected whole output names and move
those results without persistence-related copies. Function attributes,
``Loop`` and ``Scan`` use their ordinary runtime implementations: their normal
computation/transport costs remain, but unrelated operators are not prohibited.

``GraphBuilder`` preserves and validates persistence declarations during
import/export and supported rewrites. Direct edits to graph input/output
names require corresponding binding edits; dangling names are rejected.
An export to standard ONNX
that cannot preserve this contract must be rejected, not silently strip
the bindings. Explicit model saving serializes the declarations, not the
current request's retained state; execution itself does not serialize them.
