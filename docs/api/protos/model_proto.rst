==========
ModelProto
==========

Deferred weights
----------------

``load(path, defer_weights=True)`` reads binary ONNX structure without materializing
raw tensor payloads. Unlike ``skip_raw_data``, it retains a non-serialized immutable
source range for every inline ``raw_data`` field and ordinary external tensor.
External locations remain relative to the original model directory, including
when several tensors share one weights file. The raw-data threshold is ignored.
The main model file is memory-mapped through its securely pinned descriptor:
metadata decoding does not issue a system call per varint byte. Parsing accesses
only metadata and skips raw payload ranges without copying or explicitly reading
them. Mapping reserves virtual address space, not a resident copy of the weights;
the operating system may still fetch neighboring bytes at page granularity.

.. code-block:: python

    from onnx_light.onnx import load, save

    model = load("model.onnx", defer_weights=True)
    model.load_weight("layer.weight")  # Other weights remain unloaded.
    assert model.is_weight_loaded("layer.weight")
    model.unload_weight("layer.weight")
    model.load_weight("layer.weight")  # Reloads from the original byte range.
    model.load_weights()
    save(model, "materialized.onnx")

The equivalent native API uses a file-backed stream and ``ParseOptions``:

.. code-block:: cpp

    onnx_light::ModelProto model;
    onnx_light::ParseOptions options;
    options.defer_weights = true;
    onnx_light::utils::FileStream input("model.onnx");
    model.ParseFromStream(input, options);
    model.LoadWeight("layer.weight");
    model.UnloadWeight("layer.weight");
    model.LoadWeights();
    model.SerializeToFile("materialized.onnx");

``weight_names()`` / ``WeightNames()`` includes nested graphs, tensor attributes,
sparse tensors, functions and rewrite plans. A name must match exactly one tensor:
missing and duplicate names raise errors, including repeated empty names. Bulk
operations traverse all deferred tensors without name lookup. Load/unload methods
return whether they changed the payload state; bulk methods return a count.
Zero-sized raw tensors have a distinct loaded state. Unloading an eager tensor
without a recoverable source raises an error.

Only fixed-width raw payloads are supported. Typed protobuf payload fields
(``float_data``, ``int64_data``, ``string_data``, etc.), encrypted external weights,
ORT/text input, bytes input, explicit external streams, ``no_copy=True`` and
tiny-external-data eager loading are explicitly rejected in deferred mode.
``PipelineProto`` input is unsupported; deferred loading applies to ``ModelProto``.
Pipelines constructed from source-backed models are also rejected before file,
descriptor or stream export, even after their weights have been loaded.
Encoded values (including affine layouts), paged-cache blocks and struct-type
declarations are also rejected: their extension-owned tensors are not covered
by the weight-name traversal.
Models constructed in memory by copying deferred tensors into these unsupported
extensions are rejected during weight traversal and before export destination I/O,
even when those tensors are loaded.
Existing eager parsing and ``skip_raw_data`` behavior are unchanged.

Keep all backing files immutable and available until reloading is no longer
needed. File identity, size and modification metadata are checked on reload;
missing, replaced, truncated or normally modified sources raise errors. This
is not a content hash or a guarantee against adversarial restoration of file
metadata. Secure external-file containment, link checks, byte-range validation,
alignment and ``max_tensor_size_bytes`` also apply to empty payloads.
Do not edit a deferred tensor's shape or type before reloading it.

Unloading releases the proto's allocation ownership, not merely its logical
size. Existing owning buffer views, explicit shared-payload copies and runtime
consumers may retain their own storage: unloading a proto does **not** unload
weights already retained by a runtime session. Runtime preparation rejects
unloaded deferred initializers; load them before preparing a new context.
Ordinary tensor/model copies and
``CopyFrom`` retain recoverable sources; serialization does not. Other subproto
``CopyFrom`` operations use serialization and reject unloaded deferred tensors.
Serialization node callbacks preserve deferred sources in tensor attributes and
nested graphs, including when callback changes are restored afterwards.

Export requires all deferred tensors to be loaded. Safe single-file ONNX export
inlines their bytes without changing in-memory external metadata or sources.
Export to a retained source, including symlink/hardlink aliases or the same open
file descriptor, is rejected before writing. Native callers constructing their
own output streams must call ``ValidateWeightSavePath`` **before opening or
truncating** a destination; use ``SerializeToFile`` for automatic preflight.
Python file-like export supports standard binary files and ``io.BytesIO``.
Standalone deferred tensor file, descriptor and stream exports (including
``save_tensor``) are rejected, even after loading. Path-based helpers reject
before opening the destination; descriptor/stream methods reject before writing.
This also applies recursively to tensor-containing submessages, including graphs,
nodes, attributes, sparse tensors, sequences, maps and optional values.
Export the owning model or eagerly reload a serialized, loaded tensor to detach
its source before exporting it independently.
External-output, text and ORT conversions require first saving a single-file
export and reloading it eagerly. Unloaded weights are never silently serialized.

``rewritings`` is an onnx-light extension at field **1001**, following
``struct_types`` at 1000 within the local 1000-1099 extension range. Its
protobuf map entries contain a string key (field 1) and a
:doc:`rewriting_proto` value (field 2), without an additional public message.
Names must be nonempty and unique; duplicate wire keys are rejected rather
than overwritten.

The graph and functions remain canonical. Reading a model never executes a
plan. ``GraphBuilder.apply_rewritings(name)`` explicitly applies one plan to this
canonical source and rejects unknown names, invalid edits and unavailable
implementations, without falling back to another variant.
The same method accepts a ``RewritingProto`` directly instead of a name.
For ``SequenceMap``, named-plan compatibility also requires a registered
output-packing callback, not just the control-flow handler and body kernels.

``model.add_rewriting(name, plan)`` rejects duplicate names.
``model.rewritings`` is a mutable map, not a detached Python dictionary.
As with nanobind's standard map bindings, reading an entry returns a plan
copy; modify a plan and assign it back with ``model.rewritings[name] = plan``.
This keeps retained Python values safe when an entry or the entire map is cleared.
GraphBuilder retains plans across import/export, clears them on canonical
mutations (including nested mutations), and never transfers them to a replay
result. Finalization that changes node order also invalidates positional plans;
finalize the canonical builder before attaching plans. Direct proto field
mutations cannot be tracked: callers must call ``model.clear_rewritings()``
before editing the canonical graph, functions, opsets or type declarations.
ORT serialization and standard-model builder export reject bundled variants
rather than silently discarding them. Export modes and external-data packaging
remain the following implementation step.

Binary-size budget
------------------

Named-plan maps and explicit shared-payload snapshots add serialization,
comparison, copy-on-write storage and export-lease code to ``lib_onnx_proto``.
Linux CI run ``39752559187`` at commit ``f470c5a8`` established the installed-size
reference of 1,773,952 bytes. Authenticated encrypted external weights add shard
I/O, validation and extended tensor traversal. Linux CI run ``40030558220`` at
commit ``9048f1bc4`` measures 1,851,608 installed bytes, 1,331,306 ``.text`` bytes
and 898 defined dynamic symbols.
The initial deferred-source implementation measured 1,903,824 installed bytes,
1,388,540 ``.text`` bytes and 884 defined dynamic symbols with GCC 14.2.0
on Linux (stripped Release, core-only Python build). Linux CI run ``40126419070``
at commit ``e353b5cc4`` measured 1,904,888 installed bytes, 1,373,642 ``.text``
bytes and 912 defined dynamic symbols with GCC 13.3. CI retains the initial
installed-size reference and uses that CI text reference, with a 5% margin
(``--size-budget-margin-percent 5``), giving limits of 1,999,015 installed
bytes and 1,442,324 ``.text`` bytes.
Exceeding either limit still fails CI. The error reports the reference, margin and effective limit,
and explains that new proto fields, native code or compiler/build changes can
increase binary size. Review those changes before updating a reference.
The strict symbol limit is 914: the CI measurement plus the pipeline and
recursive-submessage deferred-export validators.
The shared-library dependency allowlist is unchanged.

.. autoclass:: onnx_light.onnx.ModelProto
    :members:
