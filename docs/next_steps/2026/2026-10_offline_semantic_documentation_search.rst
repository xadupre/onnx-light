.. _l-next-steps-offline-semantic-documentation-search:

Offline-first semantic documentation search
===========================================

:Date: 2026-10
:Updated: 2026-10-18

**discussion**

Objective
+++++++++

Searches technical documentation locally in the browser using ONNX Runtime Web,
without a compute server, external inference API or generative model.
This is a proposal, not an implemented feature.

The reusable engine belongs in a separate project. onnx-light only integrates
that project's published package into its documentation.

Separate project: engine and tools
++++++++++++++++++++++++++++++++++

Implement the following outside onnx-light:

* A generic HTML indexer and CLI that extract passages, headings, metadata and
  URLs with anchors, then generate document embeddings locally before publication.
  Markdown input can follow the first working prototype.
* A browser library with initialization, asynchronous search and resource disposal.
  Results contain passages, titles, URLs and scores; the core has no DOM or
  Sphinx dependency.
* Lexical search preserving API identifiers, exhaustive vector search and
  reciprocal rank fusion. Do not add approximate search before measurements
  justify it.
* Model loading, tokenization, pooling and normalization, with ONNX Runtime Web
  running in a Worker on CPU/WASM. WebGPU is optional and deferred.
* A versioned bundle contract recording model/tokenizer revisions, embedding
  settings, resource hashes and sizes. Reject incompatible bundles.
* Reusable offline caching with a Service Worker, complete-bundle activation and
  reuse of unchanged resources. Report download and cache failures explicitly.
* A minimal reusable search interface and standalone static-site example.
  Render snippets as text and validate result URLs.

First prove model and tokenizer feasibility. The
`multilingual-e5-small model card
<https://huggingface.co/intfloat/multilingual-e5-small>`_
specifies 384 dimensions, at most 512 tokens, ``query:`` and ``passage:``
prefixes followed by a space, masked mean pooling and L2 normalization.
An available
`INT8 ONNX export
<https://huggingface.co/api/models/Xenova/multilingual-e5-small/tree/main/onnx>`_
is about 118 MB before tokenizer/runtime resources; browser compatibility
and acceptable memory usage remain to be demonstrated.

Inspect the selected graph's actual inputs, outputs and types. Compare a
standalone JavaScript tokenizer with a WASM tokenizer binding; Transformers.js
is not required. Verify token IDs and masks against Python, including Unicode,
code and truncation, then compare browser and reference embeddings.

Own the generic relevance and performance tests here: exact symbols,
paraphrases, French questions over English text, weak matches, ranking fusion,
model loading, offline reload and interrupted updates. Measure Recall@5,
MRR@10, download size, memory and cold/warm latency on the target browsers.

onnx-light: documentation integration only
++++++++++++++++++++++++++++++++++++++++++

Implement the following in onnx-light, after the standalone prototype works:

* Configure the generic indexer for generated Sphinx HTML: select article content,
  exclude navigation and preserve Python/C++ API names and final anchors.
  Keep theme selectors, documentation base URL and corpus choices here.
* Add a documentation-build step invoking the external CLI and publishing the
  generated bundle with pinned model, tokenizer and runtime assets. Generate
  embeddings during publication, not at every browser visit.
* Load the external search package from the documentation's static assets and
  connect its interface to the PyData theme. Configure its cache scope without
  overwriting an existing Service Worker.
* Add onnx-light-specific relevance queries and integration checks: real symbols,
  correct page anchors, version updates and offline behavior on the published
  documentation.

No search algorithm, tokenizer, model runtime or generic cache implementation
belongs in onnx-light. No change to its C++ runtime or Python bindings is required.
If publication is managed in another repository, update that publishing workflow
as part of the integration, rather than adding an unused local build command.

Implementation PRs
++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Order
     - Project
     - Deliverable and acceptance
   * - 1
     - Separate project
     - CPU/WASM prototype ranking twenty passages. Verify graph signatures,
       tokenizer parity, embeddings, browser compatibility and resource budgets.
   * - 2
     - Separate project
     - HTML indexing CLI, static bundle, hybrid ranking, public API and demo.
       Compare lexical, vector and hybrid results on a real documentation subset.
   * - 3
     - Separate project
     - Offline caching and coherent updates. Reload and search with networking
       blocked; retain the previous bundle if an update is interrupted.
   * - 4
     - onnx-light
     - Configure extraction and publication using the external package.
       Verify corpus coverage and that generated URLs resolve to real sections.
   * - 5
     - onnx-light
     - Integrate the search UI and cache scope. Validate relevance and offline
       navigation requirements against the actual published site.

Each PR depends on the preceding deliverable. Generic defects found during
integration are fixed in the separate project, not by copying its code here.

Decisions before implementation
+++++++++++++++++++++++++++++++

Agree on the separate repository, model redistribution licenses, corpus size,
browser/device matrix and maximum download, memory and CPU latency budgets.
E5 is a candidate, not a mandatory choice.

Distinguish offline search over cached passages from opening complete pages
offline: the latter also requires caching those pages and their assets.
Confirm HTTPS hosting, Service Worker scope and storage limits. Serve all
required resources locally; no query-time CDN or model-hub downloads.
