==========
GraphProto
==========

.. autoclass:: onnx_light.onnx.GraphProto
    :members:

Persistent bindings
===================

``GraphProto.persistent_bindings`` (extension field 1001) declares root-graph
output-to-input feedback. It stores wiring only, never state contents or owners.
``PersistentBindingProto`` uses only ``input_name`` (field 1) and ``output_name``
(field 2). They select whole graph values by exact name. Dots and backslashes
are literal characters, not field-path syntax. An input is entirely persistent
or entirely supplied by current feeds. Former field-path wire fields 3 and 4
are rejected on load rather than silently changing partial persistence to whole
persistence.

Validation resolves catalogue references and requires compatible tensor or
structure declarations. Tensor ranks are compared when both are known, and
dimensions when both are concrete; symbolic dimensions and unknown ranks are
accepted. Byte-encoded values still require fixed geometry under the encoded
layout validators. Catalogue identities must agree for whole referenced
formats. Whole structures include all dynamic fields and their declared constants.
Duplicate input selections and duplicate output selections are rejected; names
such as ``state`` and ``state.cache`` are distinct graph values.
Declarations in control-flow subgraphs are not supported.

``GraphBuilder.make_persistent_binding(binding)`` appends a declaration;
``persistent_bindings()`` returns a copy. Graph/model imports and exports retain
the declarations. Existing optimization passes preserve declared IO names;
the builder has no public operation for renaming graph IO. Renaming or removing
an IO by directly editing a proto is not an automatic binding rewrite: callers
must also update the affected declarations. Export rejects dangling names
rather than dropping declarations. Standard ONNX, ORT and function
exports reject persistence semantics.

.. autoclass:: onnx_light.onnx.PersistentBindingProto
    :members:
