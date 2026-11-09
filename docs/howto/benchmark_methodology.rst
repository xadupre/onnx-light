.. _l-benchmark-methodology:

Benchmark methodology
=====================

CPU benchmarks are sensitive to persistent thread pools, lazy preparation,
allocator reuse, and runtime-specific caches. Comparative gallery examples
follow the rules below so a backend does not perturb the backend measured
after it.

Separate runtime phases
-----------------------

Prepare identical models and inputs first. Then measure every case for
``onnx-light`` before constructing any ONNX Runtime session. The ONNX Runtime
phase starts only after all ``onnx-light`` sessions from the measured phase
have been released. Other multithreaded libraries, such as NumPy linked to a
threaded BLAS, run in their own phase as well.

Do not alternate runtimes in a timing loop. Both runtimes keep their documented
default spin policy: changing spin behavior changes the runtime being measured.
Phase separation, rather than disabling spin, prevents one runtime's active
workers from consuming CPU while another runtime is timed.

Bound and report measurements
-----------------------------

Warm-up calls are untimed. A timed loop stops when its requested repetition
count is reached or when measured execution reaches two seconds cumulatively,
whichever comes first. A call already in progress is allowed to finish. Report
the median of the collected samples; retain raw samples when the benchmark
produces an artifact.

Keep model parsing, session construction, first-run preparation, and steady
state separate unless startup is the explicit subject of the benchmark.

Validate execution
------------------

Use the same logical inputs for every backend and validate complete outputs
with an explicit numerical tolerance. Kernel libraries must also verify the
library-qualified kernel name outside the timed region. Correct output alone
does not prove that the intended accelerated kernel ran.

Shared CI machines provide diagnostic measurements, not stable performance
gates. Publish the runtime versions, thread counts, CPU topology, and build
configuration when results are intended for comparison across machines.

Explicit generation-only TinyLLM benchmark
-----------------------------------------

``benchmarks/bench_generation_only.py`` is a separately named benchmark. It does
not change the existing benchmarks or the upstream ``TinyLLM`` and
``TinyLLMGenAI`` defaults. Full-sequence logits remain the contract for
all-token inference and validation (including ``TinyLLM.time_prefill``'s
``[1, 128, vocabulary]`` output). Last-token preparation is an explicit opt-in
for generation; it is not appropriate for training, perplexity, or all-token
validation.

Use the existing exporter unchanged. In upstream
``xadupre/onnx-asv-benchmark``'s ``tiny_llm.py``, ``create_model`` accepts
``model_name``, ``input_path``, ``output_dir``, ``precision``,
``execution_provider``, ``cache_dir``, and ``num_hidden_layers``.
For example, with the exporter installed and its input paths configured:

.. code-block:: python

    from modelbuilder.builder import create_model

    create_model(
        model_name=model_name,
        input_path=input_path,
        output_dir=output_dir,
        precision=precision,
        execution_provider="cpu",
        cache_dir=cache_dir,
        num_hidden_layers=1,
    )

It emits
``model.onnx`` in ``output_dir``. No generation-only exporter option is required
or assumed. Apply the local native preparation *after* export, to a separate
in-memory copy:

.. code-block:: python

    from onnx_light import onnx
    from onnx_light.onnx.reference import prepare_generation_model

    full_model = onnx.load("export/model.onnx")
    generation_model = onnx.ModelProto()
    generation_model.CopyFrom(full_model)
    prepare_generation_model(generation_model, logits_name="logits")

The native API mutates that copy in place, selecting the last hidden-state
position before the terminal rank-3 MatMul (including a supported trailing
Cast). The original exported model remains available for full-logits checks.
Preparation must happen before constructing an evaluator/session or exporting
the reduced graph; it does not update sessions already constructed from the
original graph. Repeated preparation of the same model is rejected.
Do not silently enable this transformation in the existing exporter or
``TinyLLM`` benchmark.

The hidden tensor must have a nonempty sequence dimension at execution time.
The selection is ``hidden[:, -1:, :]``: its index refers to the current
hidden tensor's sequence width, never the accumulated cache length.
During cached decode this width is normally one even when the cache is long.
The last position must be a real token in every batch row: use unpadded or
left-padded prompts, not right-padded or all-padding prompts.
``ReferenceEvaluator.generate`` validates binary left-padding masks and rejects
empty/all-padding prompts. Direct ``run`` callers must enforce these
preconditions themselves; the preparation API has no feed values to inspect.

Save actual prefill feeds with ``numpy.savez("prefill.npz", **feeds)``. Supply
every model input under its exact ONNX name, including INT64 ``input_ids``,
any mask and position inputs, and explicit empty
``past_key_values.0.key`` / ``past_key_values.0.value`` caches for every layer.
The benchmark loads NPZ files with pickling disabled. Run from this repository:

.. code-block:: bash

    PYTHONPATH=. python benchmarks/bench_generation_only.py \
        --model export/model.onnx --feeds prefill.npz --threads 1 --repeat 5

The script measures CPU native and CPU ONNX Runtime phases on the *same*
prepared model and also reports the full-model baseline. JSON reports warmed
prefill latency, isolated LM-head MatMul/optional Cast latency, and logits
tensor bytes separately. Hidden states and projection weights are captured
outside timing; the reduced isolated head receives an already sliced hidden
state, so its timing excludes Gather, decoder layers, and cache handling.
Logits bytes are not peak allocator memory or process RSS. Model loading,
session creation, validation and native generation checks are excluded from
latency samples.

Validation compares complete outputs (only slicing the full logits'
sequence axis), native versus ORT outputs, and greedy native
``ReferenceEvaluator.generate`` results for one-token prefill and multi-token
prefill/decode. Both full and reduced models are called twice to check that
generation state resets. The decoding loop is native, not a Python token loop.
Generation-local copies map standard ``past_key_values.<layer>.key/value``
inputs to ``present.<layer>.key/value`` or
``present_key_values.<layer>.key/value`` outputs; ambiguous or missing mappings
fail explicitly rather than claiming cached decode.

The ASV-compatible class ``TinyLLMGenerationOnly`` has independent ``backend``
and ``variant`` parameters and separate ``time_prefill``,
``time_lm_head_projection`` and ``track_logits_bytes`` methods. Set
``ONNX_LIGHT_TINYLLM_MODEL``, ``ONNX_LIGHT_TINYLLM_FEEDS`` and optionally
``ONNX_LIGHT_TINYLLM_THREADS`` in its ASV environment. Missing artifacts skip
the case; the class never substitutes a synthetic model.

``--synthetic`` runs a deterministic projection/cache-wiring smoke test with a
trailing Cast and explicit empty caches, not a transformer. Its JSON is explicitly labeled
``SYNTHETIC_PROJECTION_SMOKE_TEST_NOT_TINYLLM``. Without real exported artifacts,
these are not TinyLLM integration measurements and must not be reported as such.
