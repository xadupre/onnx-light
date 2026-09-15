.. _l-design-shape-sequences:

Sequences, maps and subgraphs
=============================

The :ref:`algorithm overview <l-design-shape-inference>` describes how
``onnx_shapes`` infers the element type and shape of every *tensor* value
in a graph. ONNX also carries **sequence** and **map** values, and nests
graphs inside ``If`` / ``Loop`` / ``Scan`` nodes and
model-local functions. This page describes how those richer values and
nested graphs are handled.

Sequence values
---------------

A tensor descriptor (:cpp:class:`SymTensor`) cannot represent the
output of ``SequenceConstruct``, ``SequenceEmpty``,
``SplitToSequence`` and the other sequence operators, because those
values are *sequences of tensors* rather than tensors. They are described
by a separate :cpp:class:`SymSequence` descriptor (see
``onnx_light/onnx_core/symbolic/sym_sequence.h``), the sequence analogue of
:cpp:class:`SymTensor`. It records:

* the common **element dtype** shared by every tensor of the sequence
  (``kUndefined`` when unknown, e.g. for an empty ``SequenceConstruct``);
* either one :cpp:class:`SymShape` **per element** (when the per-element
  shapes are known) or a single, possibly symbolic, **length**
  :cpp:class:`SymDim` (e.g. for ``SplitToSequence`` whose length depends
  on a runtime ``split`` value).

:cpp:func:`SymSequence::HasElemDtype` and
:cpp:func:`SymSequence::HasElemShapes` distinguish *unknown* from
*empty sequence* (length 0).

:cpp:class:`ShapesContext` therefore keeps **two** parallel maps: the
``name → SymTensor`` map for tensors and a ``name → SymSequence`` map
for sequences. The sequence map has its own C++ accessors
(``SetSequence`` / ``HasSequence`` / ``GetSequence`` /
:cpp:func:`ShapesContext::Sequences`); the Python bindings expose the
read-only introspection helpers ``has_sequence`` / ``sequence_names`` /
``sequences_size``. ``CheckInputsAvailable`` and
``CheckOutputsNotAvailable`` consult **both** maps, so a value is
considered "known" whether it is a tensor or a sequence.

Map values
----------

ONNX *map* values (``map(K, V)``) and *sequences of maps*
(``seq(map(K, V))``) are produced by the traditional-ML operators
(``ZipMap``, ``CastMap``, ``DictVectorizer``, ...). They are
not given a dedicated descriptor; instead they reuse
:cpp:class:`SymTensor` with a map-valued :cpp:enum:`TensorType`
enumerator (see ``onnx_light/onnx_core/light_op_schema/light_op_schema.h``):

* ``kMapStringInt64`` / ``kMapInt64String`` / ``kMapInt64Float`` / ... —
  the ``map(K, V)`` element types;
* ``kSeqMapStringFloat`` / ``kSeqMapInt64Float`` — the
  ``seq(map(K, V))`` types, for example the output of ``ZipMap``, whose
  leading (sequence-length) dimension is the input batch size.

Because the element type already encodes the key/value pair, map-valued
descriptors flow through the same tensor map and the same write-back path
as ordinary tensors; only the dtype differs.

Subgraphs
---------

Control-flow operators embed graphs in their attributes: ``then_branch``
/ ``else_branch`` for ``If``, ``body`` for ``Loop`` and
``Scan``. Each subgraph is inferred in a **child**
:cpp:class:`ShapesContext` derived from the parent (see
``onnx_light/onnx_extensions/shapes/shapes/controlflow/shape_controlflow.cc``):

#. The child context is **copied from the parent** so the subgraph can
   read outer-scope values it captures by name, then seeded with the
   subgraph's own initializers and formal inputs (the loop iteration
   count / condition, the scan slices, ...).
#. :cpp:func:`ShapesContext::set_current_subgraph` records the owning
   node index and the attribute name (``"body"``, ``"then_branch"`` or
   ``"else_branch"``) so that every :cpp:class:`ShapeEvent` recorded
   inside carries its ``subgraph_node_index`` / ``subgraph_attr_name``
   and can be traced back to its originating subgraph (see
   :ref:`l-design-shape-events`).
#. ``ComputeShapes`` runs over the subgraph nodes in the child context.
#. The subgraph outputs are read back from the child context and mapped
   onto the control-flow node's outputs. ``If`` merges the matching
   ``then`` / ``else`` outputs (a differing dimension becomes a fresh
   symbol bounded above by the ``max`` of the two branch expressions);
   ``Loop`` / ``Scan`` reshape the body outputs into the loop-carried and
   scan-output shapes.
#. The child context is **retained** on the parent through
   :cpp:func:`ShapesContext::RegisterSubgraphContext`, keyed by the owning
   node index and the attribute name, so the descriptors inferred *inside*
   the subgraph stay inspectable once the control-flow node has been
   processed. They are reachable through
   :cpp:func:`ShapesContext::HasSubgraphContext`,
   :cpp:func:`ShapesContext::GetSubgraphContext` and
   :cpp:func:`ShapesContext::SubgraphContexts` (exposed in Python as
   ``has_subgraph_context`` / ``subgraph_context`` /
   ``subgraph_context_keys`` / ``subgraph_contexts_size``).

Local functions
---------------

A call to a model-local function (a ``FunctionProto`` listed in
``model.functions``) is expanded by ``ExpandLocalFunctionCall`` (in
``onnx_light/onnx_core/shapes/shape_inference.cc``). Unlike a subgraph,
a function body does **not** capture outer-scope values, so it runs in a
**fresh, isolated** child :cpp:class:`ShapesContext` that:

* inherits the caller's opset versions, then lets the function's own
  ``opset_import`` override them;
* receives a copy of the parent's local-function map, so nested local
  calls are dispatched too;
* binds the call-site inputs to the function's formal input names
  **positionally** (carrying either an :cpp:class:`SymTensor` or an
  :cpp:class:`SymSequence` descriptor);
* resolves ``ref_attr_name`` attribute references in the body against the
  call-site node's attributes before inference.

After the body is inferred, the function outputs are copied back onto the
caller-visible output names.

.. note::

   Each ``If`` / ``Loop`` / ``Scan`` node retains the child
   :cpp:class:`ShapesContext` it builds for every attribute subgraph
   (``then_branch`` / ``else_branch`` / ``body``) on its parent context,
   keyed by ``(node_index, attr_name)``, so the inferred descriptors
   *inside* the subgraph remain inspectable (and traceable through the
   :ref:`event log <l-design-shape-events>`) after inference completes.
   Local-function calls still build a child context that is discarded once
   its outputs have been merged back; extending the same retention to
   ``ExpandLocalFunctionCall`` is a natural follow-up.

API reference
-------------

* **C++ API**: :cpp:class:`SymSequence`
  (:doc:`/api/cpp/onnx_core/sym_sequence`),
  :cpp:class:`ShapesContext`
  (:doc:`/api/cpp/onnx_core/shapes/shapes_context`) and
  :doc:`/api/cpp/onnx_core/shapes/shape_inference`.
* **Tensor types**: :cpp:enum:`TensorType`
  (:doc:`/api/cpp/onnx_core/light_op_schema`).
