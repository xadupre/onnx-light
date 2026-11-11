generation
==========

Pause and resume
----------------

``GenerationState`` is opt-in. Ordinary ``Generate`` calls remain independent.
For a model with graph-declared persistent bindings, the first invocation
prefills the prompt and later invocations process only the pending token:

.. code-block:: cpp

    GenerationOptions options;
    options.temperature = 0.8;
    options.seed = 7;
    GenerationState state(model, context, feeds, options);
    Tensor first = state.Generate(1);   // Prompt plus one sampled, pending token.
    Tensor result = state.Generate(9);  // Same result as an uninterrupted 10 tokens.
    Tensor unchanged = state.Generate(0); // No model invocation.
    state.Reset();                     // Restarts the original prompt and seed.

A pipeline constructor accepts ``PipelineProto`` and explicit generation options,
including pipelines with distinct prefill/decode models. A ``LoadedRuntimePipeline``
constructor also reuses its session settings, CPU executor policy and preparation
store; the loaded owner must outlive the state. Without persistent bindings,
single-model generation still recomputes the full prefix.

The native Python runtime binding provides the same request object:

.. code-block:: python

    from onnx_light.onnx_py._onnxpykernels import runtime

    options = runtime.GenerationOptions()
    options.seed = 7
    state = runtime.GenerationState(model, context, feeds, options)
    first = state.generate(1)
    result = state.generate(9)
    state.reset()

The context must have the same kernel registrations and execution setup as for
``runtime.generate``. Returned tensors include the complete original prompt and
all tokens emitted so far; Python callers can use ``numpy.from_dlpack(result)``.

Each state is bound to one immutable model/pipeline, context, copied initial feeds
and sampling/EOS settings. A resume call cannot replace any of these. New prompt
suffixes and conversation turns are not supported; construct a new state with
the new prompt instead. Initial caches must describe an empty prefix. Reset
discards caches, pending tokens, finished rows and RNG progress, and starts the
original request again. EOS-finished rows stay finished until reset.

In C++, models, context and allocator owners must outlive the state and remain
unchanged. Python keeps the model and context alive, but external allocator owners
are still the caller's responsibility. The state owns its sessions and mutable
caches; caches are not copied at pause/resume boundaries. Different conversations
need separate states. State calls are exclusive, and execution sharing a context
or static preparation store must be serialized, including ordinary runtime calls.
Initializer consumption is rejected for retained requests.

``Generate(budget, observer, logit_byte_budget)`` also supports native step
observation. Indices continue across calls, and logit capture budgets are per call.
Returning false pauses after committing the emitted token. Observer exceptions
propagate with that token committed, and the request can resume. Other execution
or sampling failures invalidate continuation until reset: a runtime run may have
published a cache before sampling fails. This does not weaken the runtime's
per-run guarantee that persistent values publish only on successful execution.
Invalid budgets and other validation errors before execution do not advance state.

.. doxygenfile:: onnx_core/runtime/generation.h
    :project: onnx-light
