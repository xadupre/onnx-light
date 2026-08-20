.. _l-next-steps-model-loading-bug-fixes:

Model-loading bug fixes
=======================

:Date: 2026-08

**ready to implement**

Objective
+++++++++

This is step 1 of the fast-loading sequence. It fixes three known defects
before introducing a new ORT ownership contract or asynchronous preparation:

* the protobuf-compatible C++ file-descriptor and array adapters stage complete
  inputs unnecessarily;
* Python external-data auto-discovery parses the main model twice;
* the native evaluator recreates runtime initializer tensors on every run.

These are defects in existing paths, not speculative architecture. Each fix
must preserve malformed-input handling, parser limits, payload lifetime, and
the public synchronous API.

Bug PR01 -- remove C++ compatibility-parser staging (done)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

**Implemented**

``ProtoMessageAdapter::ParseFromFileDescriptor`` now parses directly through
``FdReadStream``, and ``ParseFromArray`` uses a bounded non-owning stream. The
native parser accepts both ``BinaryStream`` implementations without a complete
input staging copy.

Parse seekable descriptors directly through ``FdReadStream`` with adaptive
buffering. Preserve the descriptor's current offset and EOF behavior. Pipes
remain bounded streaming inputs. Parse arrays through a non-owning bounded
stream; the compatibility overload may not retain borrowed payload bytes
because it receives no lifetime token.

Acceptance:

* no allocation proportional to the complete input precedes parsing;
* malformed/truncated inputs, non-zero offsets, pipes, read failures, tensor
  limits, and recursion limits retain regression coverage;
* large inline-model parsing improves by at least 20%, while representative
  small models regress by no more than 3%;
* the existing ``onnxruntime_USE_ONNX_LIGHT`` integration benefits without an
  onnxruntime source change.

Bug PR02 -- remove Python external-data double parsing
++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

**implemented**

When ``load_external_data=True`` and ``location`` is absent, ``load()`` calls
``_find_external_location()``. That helper parses a temporary metadata-only
model, returns the first external location, and discards the model. ``load()``
then parses the main protobuf again for the real result.

Parse once and retain every validated external-data descriptor, including
multiple files and nested graphs. Hydrate selected descriptors on that same
model, or let a path-aware stream resolve a descriptor when its
``TensorProto`` completes. Locations remain confined to the model directory
and are checked for traversal, symlink escape, offset overflow, and truncation.

Acceptance:

* automatic external-data loading parses the main protobuf exactly once;
* multiple locations and nested subgraphs are supported;
* all views into one external file share one source/mapping owner;
* failures identify the tensor, resolved source, byte range, and reason.

Bug PR03 -- cache runtime initializer tensors
++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

Payload ownership already exists: inline bytes belong to ``ModelProto`` and
mapped bytes retain their mapping owner. The defect is repeated runtime
materialization. ``ReferenceEvaluatorRunner::Run`` clears its
``RuntimeContext``, iterates over every graph initializer, and calls
``TensorFromProto`` again. Raw payloads are borrowed again, names and shapes
are rebuilt, and typed fields may be copied again.

Create a session-owned initializer tensor store. Materialize each initializer
once, retain or reference its existing owner, and lend immutable tensor views
to invocation contexts.

Acceptance:

* repeated runs do not rebuild initializer names, shapes, views, or typed
  payload buffers;
* payload addresses and ownership tokens remain stable for the session;
* releasing a source object is safe when the session owns its source and fails
  explicitly when a required caller-lifetime token is absent;
* outputs, mutation rules, and initializer/input override semantics remain
  unchanged.

Completion
++++++++++

After these three PRs, proceed to
:ref:`l-next-steps-prepared-execution`. The explicit onnxruntime integration
follows once the native prepared-object and ownership contracts are stable.
Adaptive I/O tuning, selective payload loading, prepared caches, and
first-token overlap intentionally belong to the later native completion
roadmap.
