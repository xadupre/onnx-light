.. _l-design-gradient:

Gradient Design
===============

Overview
--------

The ``onnx_gradient`` module implements **reverse-mode automatic
differentiation** (backpropagation) directly on ONNX graphs.  Given a
forward computation expressed as a list of :cpp:class:`NodeProto` objects (or
as a :cpp:class:`FunctionProto`), it produces a new
:cpp:class:`FunctionProto` that computes the partial derivatives of a
scalar-valued output ``y`` with respect to a set of input tensors ``xs``.

The design goal is to keep the gradient computation *purely symbolic*:
no numerical tensors are evaluated during the differentiation step.  The
resulting :cpp:class:`FunctionProto` can then be executed by any ONNX
runtime (``onnx_light`` kernels, OnnxRuntime, …).

Architecture
------------

.. code-block:: text

    ┌─────────────────────────────────────────────────┐
    │  Public API  (onnx_gradient/gradient.h)         │
    │                                                 │
    │  GradientOfNodes(nodes, inputs, initializers,   │
    │                  xs, y, zs, registry)           │
    │                                                 │
    │  GradientOfFunction(function, xs, y, zs,        │
    │                     registry)                   │
    └──────────────┬──────────────────────────────────┘
                   │  delegates to
                   ▼
    ┌─────────────────────────────────────────────────┐
    │  Core algorithm  (onnx_core/gradient/)          │
    │                                                 │
    │  1. Build a value→producer node map.            │
    │  2. Walk backward from y using DFS.             │
    │  3. For each node on the path, look up the      │
    │     GradFn in the registry and emit backward    │
    │     nodes into the output FunctionProto.        │
    │  4. Accumulate gradients for nodes with         │
    │     multiple consumers (Add node emitted).      │
    └──────────────┬──────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────┐
    │  Registration mechanism  (onnx_core/gradient/   │
    │                           grad_dispatcher)      │
    │                                                 │
    │  RegisterGradientFunction() inserts a GradFn.   │
    │  ApplyBackward() dispatches to the registered   │
    │  function for the current operator.             │
    └──────────────┬──────────────────────────────────┘
                   │  populated by
                   ▼
    ┌─────────────────────────────────────────────────┐
    │  Per-operator gradient rules                    │
    │  onnx_gradient/gradient/{math,nn,tensor,        │
    │                           reduction}/           │
    │                                                 │
    │  DefaultGradRegistry() registers all standard   │
    │  operators: Conv, MatMul, Gemm, Add, Sub, Mul,  │
    │  Div, Neg, Identity, Relu, Sigmoid, Tanh,       │
    │  ReduceSum, ReduceMean, Reshape, Transpose, …   │
    └─────────────────────────────────────────────────┘

Key data structures
-------------------

``GradRegistry``
    An ``std::unordered_map<std::pair<std::string, std::string>, GradFn,
    PairStringHash>`` mapping ``(domain, op_type)`` pairs to gradient
    functions.  The empty string ``""`` denotes the default ONNX operator
    domain.  The default registry is built once at program startup by
    ``onnx_gradient::DefaultGradRegistry()`` and can be copied and
    extended by callers via ``RegisterGradientFunction``.

``GradFn``
    A ``std::function`` with signature::

        bool fn(const NodeProto& node,
                const std::string& output_grad,
                std::unordered_map<std::string, std::string>& grad_accum,
                int& counter,
                FunctionProto& func) -> bool;

    It appends backward ONNX nodes to ``func``, records the gradient name
    for each input of ``node`` in ``grad_accum``, and uses ``counter`` to
    generate unique intermediate names.  It returns ``true`` on success.

Reverse traversal algorithm
----------------------------

1. **Producer map** — scan the forward ``nodes`` list and build a map
   from output-tensor name to the :cpp:class:`NodeProto` that produces it.

2. **Reachability** — starting from ``y``, perform a DFS over the producer
   map; collect all nodes that lie on a path from ``y`` back to any tensor
   in ``xs``.  Only reachable nodes are differentiated.

3. **Backward emission** — replay the reachable nodes in *reverse* order.
   For each node, call the registered ``GradFn``.  The ``GradFn`` writes
   new ONNX nodes into the output :cpp:class:`FunctionProto` and records the
   gradient name for each of the node's inputs.

4. **Gradient accumulation** — if the same tensor is consumed by more than
   one node on the backward path, an ``Add`` node is emitted to sum the
   incoming partial gradients before they are recorded in ``grad_accum``.

5. **Output collection** — after traversal, look up ``"grad_<x>"`` for each
   ``x`` in ``xs`` and add them as outputs of the returned
   :cpp:class:`FunctionProto`.

FunctionProto inputs and outputs
---------------------------------

The returned :cpp:class:`FunctionProto` follows a fixed convention:

* **inputs** — *xs* names (parameters), then *zs* names
  (non-differentiable inputs needed by backward ops), then ``"dy"``
  (the upstream gradient of ``y``; caller passes ``ones_like(y)`` for a
  plain MSE loss).
* **outputs** — ``"grad_<x>"`` for each ``x`` in ``xs``, in the same
  order.

Example: ``y = X @ W + b`` differentiated w.r.t. ``W`` and ``b``:

.. code-block:: text

    inputs  = ["W", "b", "X", "dy"]
    outputs = ["grad_W", "grad_b"]

Extending with custom operators
--------------------------------

Use ``register_gradient_function`` (C++) or the Python binding
``register_gradient_function`` to insert a custom ``GradFn`` into a copy
of the default registry, then pass that registry to
``GradientOfNodes`` / ``GradientOfFunction``::

    # Python example
    from onnx_light.onnx_core.gradient import GradRegistry, register_gradient_function

    registry = GradRegistry.default()

    def my_op_grad(node, output_grad, grad_accum, counter, func):
        # emit backward nodes into func ...
        return True

    register_gradient_function("com.example", "MyOp", my_op_grad, registry)

    grad_fn = gradient_of_nodes(
        nodes=forward_nodes,
        inputs=[...], initializers=[], xs=[...], y="loss", zs=[...],
        registry=registry,
    )

References
----------

* Python API: :doc:`/api/python/onnx_core/gradient`
* C++ API: :doc:`/api/cpp/onnx_extensions/onnx_gradient/index`
* Gallery example: :ref:`l-example-gradient-linear-regression`
