==========
ModelProto
==========

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
Linux CI run ``39752559187`` at commit ``f470c5a8`` measures 1,773,952 installed
bytes, 1,262,138 ``.text`` bytes and 897 defined dynamic symbols.
CI uses these installed-size and text measurements as references
with a 5% margin (``--size-budget-margin-percent 5``), giving limits of
1,862,649 installed bytes and 1,325,244 ``.text`` bytes. Exceeding either limit
still fails CI. The error reports the reference, margin and effective limit,
and explains that new proto fields, native code or compiler/build changes can
increase binary size. Review those changes before updating a reference.
The strict symbol limit remains 897 for native snapshot and export-lease code.
The shared-library dependency allowlist is unchanged.

.. autoclass:: onnx_light.onnx.ModelProto
    :members:
