.. _l-design-shape-events:

Shape-inference events (``ShapeEvent``)
=======================================

:cpp:class:`ShapesContext` keeps an **opt-in event log** that records,
step by step, what the shape-inference pass did: which tensor
descriptors it inserted or overwrote, which nodes it dispatched, and
which symbolic-dimension constraints it discovered. The log is the
primary tool for understanding *where* an inferred shape (or a
shape-inference error) comes from: replaying the events shows the exact
node that produced a wrong dimension or the constraint that renamed a
symbol unexpectedly.

The mechanism mirrors the runtime event log of
:cpp:class:`onnx_light::core::runtime::RuntimeContext` (see the
``plot_runtime_events`` gallery example).

Enabling the log
----------------

Event recording is **disabled by default** so it adds no overhead to the
hot path. Enable it on the context before running inference:

.. code-block:: python

    from onnx_light.onnx_core.shape_inference import (
        ShapeEventAction,
        ShapesContext,
    )

    ctx = ShapesContext()
    ctx.events_enabled = True
    ctx.compute_shape_model(model, prefill_with_value_info_output=True)
    # Next method populates the original model with the inferred shapes.
    # ctx.apply_inferred_shapes_to_model(model)
    # But that's not needed to get the event series.
    # Shapes are available with ``ctx.get(name)``

    for ev in ctx.events():
        print(ev)

``ctx.clear_events()`` empties the log without touching the inferred
descriptors. In C++ the same controls are
:cpp:func:`ShapesContext::set_events_enabled`,
:cpp:func:`ShapesContext::Events` and
:cpp:func:`ShapesContext::ClearEvents`.

Event kinds
-----------

Each entry is a :cpp:class:`ShapeEvent` whose ``action`` is one of
:cpp:enum:`ShapeEventAction` (exposed in Python as
:class:`ShapeEventAction`):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Action
     - Meaning
   * - ``kAdd``
     - A new tensor descriptor was inserted for a previously unknown
       name via :cpp:func:`ShapesContext::Set`. Carries ``name``,
       ``data_type`` and the ``shape`` (one string per dimension, so
       symbolic dims such as ``"2*N"`` are preserved).
   * - ``kReplace``
     - An existing descriptor was overwritten via
       :cpp:func:`ShapesContext::Set` (for example when an anchor merge
       tightens a shape). Same fields as ``kAdd``.
   * - ``kComputeNode``
     - A single ``NodeProto`` was dispatched. Carries the node's
       ``op_domain`` / ``op_type`` and the ordered ``inputs`` it
       consumed; it does not mutate the tensor map itself. These entries
       reconstruct the dispatch order and pinpoint the operator
       responsible for a downstream descriptor.
   * - ``kConstraint``
     - A symbolic-dimension equality (``a == b``) was recorded via
       :cpp:func:`ShapesContext::AddConstraint`; the two operands are in
       ``inputs``.
   * - ``kConstraintMax``
     - A symbolic-dimension upper bound (``lhs <= rhs``) was recorded via
       :cpp:func:`ShapesContext::AddLessEqualConstraint`; the two
       operands are in ``inputs``.

Locating events
---------------

Two fields tie each event back to its origin in the graph:

* ``node_index`` — position of the producing / dispatched node in its
  graph node list, or ``-1`` for graph inputs and ``-2`` for
  initializers.
* ``subgraph_node_index`` / ``subgraph_attr_name`` — identify the
  control-flow operator and attribute subgraph (``"body"`` for
  ``Loop`` / ``Scan``, ``"then_branch"`` / ``"else_branch"`` for
  ``If``) an event originated from; ``-1`` / empty for the top-level
  graph.

In Python, :meth:`ShapeEvent.as_dict` renders an event as a plain
dictionary for easy filtering and printing. A typical debugging session
filters the log by action — for instance keeping only ``kComputeNode``
entries to see the dispatch order, or only ``kReplace`` entries to find
where a descriptor changed:

.. code-block:: python

    compute = [ev for ev in ctx.events()
               if ev.action == ShapeEventAction.kComputeNode]
    for ev in compute:
        print(ev.node_index, ev.op_type, ev.inputs)

The :ref:`l-example-plot-shape-inference` gallery example walks through a
complete capture-and-inspect session on a backend test case.

API reference
-------------

* **C++ API**: :cpp:class:`ShapeEvent`, :cpp:enum:`ShapeEventAction`
  (:doc:`/api/cpp/onnx_core/shapes/shapes_context`).
* **Python API**: :class:`ShapeEvent`, :class:`ShapeEventAction` and
  :class:`ShapesContext`
  (:doc:`/api/python/onnx_core/shape_inference`).
