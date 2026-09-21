==========
GraphProto
==========

.. autoclass:: onnx_light.onnx.GraphProto
    :members:

Persistent bindings
===================

``GraphProto.persistent_bindings`` (extension field 1001) declares root-graph
output-to-input feedback. It stores wiring only, never state contents or owners.
``PersistentBindingProto`` uses ``input_name`` (field 1), ``output_name``
(field 2), ``input_field_path`` (field 3), and ``output_field_path`` (field 4).
Paths are arrays of literal structure-field names: ``["a.b"]`` differs from
``["a", "b"]``. An empty path selects the entire value.

Validation resolves catalogue references and requires compatible tensor or
structure declarations. Tensor ranks are compared when both are known, and
dimensions when both are concrete; symbolic dimensions and unknown ranks are
accepted. Byte-encoded values still require fixed geometry under the encoded
layout validators. Catalogue identities must agree for whole referenced
formats. Constant fields cannot be destinations or sources selected by a path.
Duplicate destinations and ancestor/descendant destination paths are rejected.
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
