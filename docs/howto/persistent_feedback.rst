.. _l-howto-persistent-feedback:

Persistent input/output feedback
===============================

Feedback state retains selected model outputs as inputs for the next call.
The model remains stateless: types come from its final input/output
declarations, and the caller supplies the mapping and initial values.
There is no persistent-state proto or separate executor.

The native :cpp:class:`onnx_light::core::runtime::FeedbackState` uses the
existing runtime execution and value ownership contracts. Create one state
per independent request. Values are copied where necessary to separate
retained state from caller-owned inputs, returned outputs and execution
scratch storage. This initial implementation promises correct feedback,
not zero-copy KV-cache reuse.

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
    context = runtime.RuntimeContext(
        runtime.KernelContext(runtime.default_opset(18))
    )
    initial = numpy.zeros(2, dtype=numpy.float32)
    state = runtime.FeedbackState(
        model, feedback={"past": "present"}, initial={"past": initial}
    )
    delta = numpy.ones(2, dtype=numpy.float32)
    first = state.run(context, {"delta": delta})
    second = state.run(context, {"delta": delta})
    numpy.testing.assert_array_equal(
        numpy.frombuffer(second["present"].raw_data(), dtype=numpy.float32),
        [2, 2],
    )
    state.reset({"past": initial})
    state.close()

The mapping is **input destination to output source**. Inputs accept runtime
``Tensor`` objects or NumPy arrays. Outputs and ``state.values`` contain
owning runtime tensors; snapshots can be changed without changing the state.
The binding keeps the model alive. The supplied context configures allocators
and custom kernels; each call executes in a fresh child context, rather than
leaving old feeds or intermediate values in the caller's context.
The binding also retains supplied contexts so cached kernel allocator
references remain valid. Reuse the same context for a state's calls.

The corresponding C++ entry points are:

.. code-block:: cpp

    using namespace onnx_light::core::runtime;

    FeedbackState state(
        model, {{"past", "present"}},
        {{"past", RuntimeValue(Tensor::FromFloat("past", {2}, {0.f, 0.f}))}});
    RuntimeContext context(KernelContext(18));
    auto outputs = state.Run(
        context,
        {{"delta", RuntimeValue(Tensor::FromFloat("delta", {2}, {1.f, 1.f}))}});
    state.Reset(
        {{"past", RuntimeValue(Tensor::FromFloat("past", {2}, {0.f, 0.f}))}});
    state.Close();

C++ callers register the operator kernels as usual before executing the model
(see :doc:`register_builtin_operators`). The model and configured allocators
must outlive the state/session using them.

Lifecycle and validation
------------------------

* Construct the state **after** graph rewrites. Every mapped input and output
  must exist in the final model, with compatible types and shape constraints.
  A removed or changed path needs an updated mapping and a new state.
* Supply initial contents for every feedback destination. Each subsequent
  call supplies the remaining inputs; current feeds must not overlap
  retained destinations.
* State advances only after successful execution and validation of the next
  values. A failed or cancelled call must not publish a partial update.
* Reset explicitly supplies new initial contents. Closing releases retained
  values. Calls, resets and closes on the same state must not overlap.
* Independent states do not share mutable retained values.

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

A binding may retain a whole input or a selected struct field. For example,
``request.cache <- response.cache`` retains only the cache, while
``request.tokens`` must be supplied anew. The struct declaration comes from
``StructTypeProto`` in the model; persistence is a property of the feedback
mapping, not a flag on the type.

This is equivalent to manually taking the selected outputs from each
stateless invocation and passing them to the next invocation. Unselected
outputs, such as logits, are not part of retained state.

Python represents named structs as nested dictionaries. For a model declaring
the ``request`` and ``response`` structures described above:

.. code-block:: python

    state = runtime.FeedbackState(
        model,
        {"request.cache": "response.cache"},
        {"request.cache": initial_cache},
    )
    output = state.run(context, {"request.tokens": tokens})

Custom kernels use ``context.get_value(name)`` and
``context.put_value(name, value)`` to exchange structured values; ordinary
tensor kernels continue to use the existing tensor API. In C++, structured
and encoded edges live in ``RuntimeContext::values()`` as ``RuntimeValue``
objects containing existing tensors or ``EncodedValueProto`` payloads.
Selecting a struct field does not decode or slice a byte-encoded payload.
Inline structured encoded payloads can be retained as whole values; external
payloads must first be loaded. This API currently supports tensors, named
structs and inline structured encodings, not sequence/map/optional state.
