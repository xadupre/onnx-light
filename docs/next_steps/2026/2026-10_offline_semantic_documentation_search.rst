.. _l-next-steps-offline-semantic-documentation-search:

Offline-first semantic documentation search
===========================================

:Date: 2026-10
:Updated: 2026-10-18

**discussion**

Objective and scope
+++++++++++++++++++

Builds an independent, reusable search library that runs entirely in the
browser with ONNX Runtime Web. Questions retrieve relevant documentation
passages, titles and URLs without a compute server, external inference API,
or mandatory generative model. Searches remain available offline after the
required resources have been downloaded and cached.

This document proposes the work; it does not implement the engine or claim
browser compatibility has already been demonstrated. Implementation belongs
in a separate repository. The onnx-light documentation is the first integration
example, not a dependency of the core library.

Repository baseline
+++++++++++++++++++

The current project uses Python and C++, CMake and scikit-build-core.
Documentation uses Sphinx, MyST, Breathe, Sphinx-Gallery and the PyData theme.
The first extractor should therefore consume generated HTML: it already
contains Python and C++ API documentation, rendered examples and final anchors.

Select article content rather than navigation, repeated sidebars or footers.
Preserve headings, API symbols, useful code and links to actual sections.
Extraction rules for a theme belong in an adapter, not the search engine.
Markdown support follows the working HTML prototype; its URL mapping must
come from the target site's routing rules rather than guessed filenames.

Minimal architecture
++++++++++++++++++++

.. code-block:: text

   Build machine, without a remote compute service:
     HTML -> passages + metadata -> local ONNX embeddings -> static search bundle

   Browser:
     question -> Worker -> lexical ranking + ONNX query embedding
                       -> vector ranking -> rank fusion -> passages and URLs

Use a small Python CLI for corpus preparation and local batch embeddings.
Use TypeScript for the browser library and Worker, with a separate minimal
HTML interface. The CLI is a publishing tool, not a required backend.
Index construction may run ahead of publication; query inference and all
search-time ranking run in the browser.

Each passage retains a stable ID, document title, section hierarchy, text,
URL with anchor, language and content hash. Split along section boundaries
and the selected tokenizer's token limit; avoid silently truncating long
sections or duplicating overlapping results from the same section.

Start with BM25 or an equivalent lexical index and an exhaustive vector scan
over normalized embeddings. Preserve full identifiers such as qualified C++
names while also indexing useful components. Keep lexical analysis distinct
from the embedding model's tokenizer. Fuse rankings with reciprocal rank
fusion rather than adding incomparable BM25 and cosine scores.

Do not introduce an approximate nearest-neighbor index until corpus size and
measurements demonstrate a need. At 384 dimensions, 10,000 float32 vectors
already occupy about 15.4 MB before text, metadata and runtime overhead.

The public API should support initialization from a bundle manifest,
asynchronous queries with a result limit, and explicit resource disposal.
Results include passage IDs, text, titles, URLs and scores. A fused score
is a ranking value, not a probability that a passage answers the question.
The core has no DOM or documentation-generator dependency.

Model and tokenizer feasibility gate
++++++++++++++++++++++++++++++++++++

Evaluate multilingual-e5-small first, without committing to it. The
`model card <https://huggingface.co/intfloat/multilingual-e5-small>`_
documents 384-dimensional embeddings, a 512-token limit, ``query:`` and
``passage:`` prefixes (each followed by a space), attention-mask-aware mean
pooling and L2 normalization. Apply the same pipeline when preparing the corpus and encoding
queries; do not assume an exported graph already includes pooling.

The public
`ONNX artifact listing
<https://huggingface.co/api/models/Xenova/multilingual-e5-small/tree/main/onnx>`_
lists approximately 470 MB for FP32, 235 MB for FP16 and 118 MB for INT8,
excluding the tokenizer and runtime. These are artifact sizes, not measured
browser memory or transfer budgets. Quantization labels alone do not predict
the smallest file or the best execution provider.

Exports exist, but their presence does not establish compatibility with
ONNX Runtime Web. Pin the selected revision and inspect actual input/output
names, tensor types, shapes, operators and any external weight files. Verify
integer input handling, masking, padding, special tokens and truncation.
Do not assume the model accepts strings or requires a particular optional
input such as token type IDs.

Compare the standalone
`Hugging Face JavaScript tokenizer
<https://github.com/huggingface/tokenizers.js>`_
with a browser-capable WASM binding to the reference tokenizer implementation.
Neither choice requires Transformers.js. Check tokenizer JSON support,
Unicode normalization, Unigram behavior, special-token processing, bundle
size and maintenance cost. Do not implement a new tokenizer from scratch.

Before accepting an option, compare token IDs and masks against the Python
reference for French, English, accents, whitespace, punctuation, API symbols,
code snippets and overlength inputs. Compare browser embeddings with local
ONNX reference embeddings using a documented numerical tolerance, and measure
ranking changes caused by quantization separately.

Execution and offline lifecycle
+++++++++++++++++++++++++++++++

Use ONNX Runtime Web directly in a Web Worker. CPU/WASM is the required
baseline; a GPU must not be necessary. Begin with a configuration that does
not require cross-origin isolation or shared-memory threading. WebGPU and
multithreaded WASM are later, measured optimizations with a tested CPU fallback.

Host application code, Worker code, runtime WASM, tokenizer, model, index and
any external weights as static resources under controlled URLs. Do not fetch
missing resources from a public model hub or CDN during a query.

Use a manifest containing format version, documentation version, model and
tokenizer revisions, pooling and normalization settings, vector dimension,
resource sizes and hashes. Reject incompatible bundles explicitly.
Address resources by content so unchanged model weights can survive an index
update. Stage and verify a complete new bundle before activating its manifest;
an interrupted update must leave the previous complete bundle usable.

Use a Service Worker and Cache Storage for offline resources. Add IndexedDB
only if measurements justify persisting additional derived index structures.
Expose download progress, initialization failures, quota errors and cache
eviction; do not claim offline readiness until required resources are present.
Persistent storage is subject to browser policies, not an unlimited guarantee.

Offline search and offline document navigation are separate commitments.
Search can show cached passages without caching every full documentation page.
If opening results offline is required, include the target pages and their
necessary assets in the offline contract and budget. Service Workers require
HTTPS or a supported local development origin, not arbitrary file URLs.

Treat extracted HTML and metadata as untrusted: render snippets as text, validate
result URL schemes and containment within the documentation's configured base,
and bound downloads, token counts and passage sizes. Keep queries local and
avoid telemetry that transmits their content.

Implementation sequence
+++++++++++++++++++++++

Each row describes one implementation PR in the independent project. The
proposal in onnx-light and the eventual documentation integration remain
separate changes.

.. list-table::
   :header-rows: 1
   :widths: 15 15 35 35

   * - PR
     - Depends on
     - Deliverable
     - Acceptance gate
   * - 1: feasibility
     - Agreed budgets
     - A static page, Worker, direct ONNX Runtime Web session and tokenizer
       that rank twenty fixed passages on CPU.
     - Records actual graph signatures, tokenizer parity, embedding agreement,
       artifact revisions, download size, initialization, memory and query
       latency. Exercises Chromium, Firefox and WebKit; confirm Safari on a
       target Apple device if supported.
   * - 2: vertical slice
     - PR 1
     - HTML extraction, local corpus embeddings, a versioned static bundle,
       lexical/vector fusion, a small public API and a standalone demo.
     - Searches a real documentation subset in French and English; exact
       symbols remain discoverable and every result opens its correct anchor.
       Reports lexical-only, vector-only and hybrid baselines.
   * - 3: offline lifecycle
     - PR 2
     - Service Worker caching, progress, coherent bundle activation and
       content-addressed updates.
     - Reloads and searches with network access blocked after preparation.
       Tests first-visit failure, missing weights or WASM, interrupted update,
       incompatible manifests, quota failure and unchanged-resource reuse.
   * - 4: evaluation and integration
     - PR 3
     - A reproducible relevance/performance report, integration instructions
       and a minimal onnx-light documentation adapter.
     - Uses about fifty labeled technical queries and reports Recall@5,
       MRR@10, cold/warm latency, peak memory and transferred bytes. Tests
       navigation and the chosen offline-page guarantee.

Include API names, errors, paraphrases, French questions over English pages,
ambiguous questions and queries with no relevant passage in the evaluation set.
Define how to display weak matches; rank fusion alone does not supply a
reliable no-answer threshold. Keep fixtures deterministic and avoid network
dependencies in routine tests.

Only after the vertical slice and measurements should further PRs add Markdown,
WebGPU, approximate search or larger corpora. A generative answer model is not
part of the required implementation.

Decisions before implementation
+++++++++++++++++++++++++++++++

Agree on the independent repository and package names, licenses for redistributed
model/tokenizer artifacts, browser/device matrix, corpus size, supported languages
and maximum acceptable first download, memory and CPU latency.

The proposed starting point is HTML input, French/English evaluation,
CPU/WASM in a Worker, precomputed document embeddings and no GPU requirement.
Whether the roughly 118 MB INT8 E5 artifact fits the download budget remains
an explicit decision, not an assumption. If it does not, compare a smaller
candidate on the same relevance corpus before selecting the model.

Also decide whether offline support covers cached search passages only or
full destination pages, and whether the host permits a Service Worker with
the necessary scope. Do not replace an existing site's search or Service
Worker without an explicit integration decision.
