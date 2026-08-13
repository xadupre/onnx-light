.. _l-next-steps-gradient:

Symbolic gradients for ONNX graphs
==================================

:Date: 2026-06

**complete**

Objective
+++++++++

Training an ONNX model requires the backward pass: given a forward graph and a
loss output, produce the graph that computes the derivatives of that output
with respect to a chosen set of variables.  ``onnx-light`` needed this as a
native, protobuf-free operation that returns *another ONNX graph* — so the
backward pass can be executed by the same runtime, saved, inspected, or
inlined into a larger training loop — rather than a numerical result tied to a
particular tensor library.

The objective was a reverse-mode automatic differentiation pass that:

* takes a forward node list (or an existing
  :class:`~onnx_light.onnx_lib.FunctionProto`), a set of variables ``xs`` to
  differentiate, and an output ``y``;
* returns a ``FunctionProto`` whose inputs are ``[xs..., zs..., "dy"]`` and
  whose outputs are the gradients ``grad_x1, grad_x2, ...``;
* is extensible, so new operators can register their own backward rule.

Post-mortem
+++++++++++

The design separates the differentiation *algorithm* from the per-operator
*rules*, mirroring the kernel/dispatch split used elsewhere in the project.

Two layers
^^^^^^^^^^

``core::gradient`` (``onnx_light/onnx_core/gradient``) owns the algorithm and
the registry types.  ``onnx_gradient``
(``onnx_light/onnx_extensions/gradient``) owns the operator rules and the
``DefaultGradRegistry`` that ships with them.  The public entry points are
``GradientOfNodes`` (differentiate a raw node list) and ``GradientOfFunction``
(differentiate a ``FunctionProto``, treating its initializers as ordinary
inputs — the pure-function form convenient for training).

A backward rule is a ``GradFn``:

.. code-block:: cpp

    using GradFn = std::function<bool(const NodeProto &node,
                                      const std::string &output_grad,
                                      std::unordered_map<std::string,std::string> &grad_accum,
                                      int &counter,
                                      FunctionProto &func)>;

Rules are looked up in a ``GradRegistry`` keyed by ``(domain, op_type)``.  Each
rule emits the backward nodes for one forward operator into ``func`` and
records the contribution to each input's gradient.

The reverse-mode pass
^^^^^^^^^^^^^^^^^^^^^^

``GradientOfNodes`` indexes each tensor name to the node that produces it,
computes the nodes reachable from ``y`` in topological order, then walks them
in reverse.  It maintains two maps: ``grad_table`` (variable to its current
gradient tensor) and ``grad_accum`` (per-variable accumulators).  It seeds
``grad_table[y] = "dy"`` and, for every node in reverse order, calls
``ApplyBackward`` which dispatches to the registered ``GradFn``.  Partial
gradients are merged by ``AccumulateGrad``, which emits an ``Add`` node only
when a variable receives a second contribution; unique intermediate names come
from ``NewGradName(prefix, counter)`` with a single monotonically increasing
counter.  A final pass emits ``Identity`` nodes to rename the accumulators to
the canonical ``grad_<variable>`` outputs.  The emitted backward
``FunctionProto`` targets ONNX opset 21.

Because the result is symbolic, the caller runs it like any other graph: the
gradient example builds a small forward model, calls the gradient pass once,
and executes the resulting backward function each epoch.

Operator coverage
^^^^^^^^^^^^^^^^^

``DefaultGradRegistry`` registers backward rules for the operators most needed
by a training loop, grouped like the kernels:

* **math** — Add, Sub, Mul, Div, Neg, Gemm, MatMul;
* **nn** — Conv, Relu, Sigmoid, Tanh and the normalization family
  (BatchNormalization, GroupNormalization, InstanceNormalization,
  LayerNormalization, LpNormalization, MeanVarianceNormalization,
  RMSNormalization);
* **tensor** — Identity, Reshape, Transpose;
* **reduction** — ReduceMean, ReduceSum.

Each rule expresses the backward pass purely in terms of existing ONNX
operators.  ``MatMul`` (``C = A @ B``) is representative: it emits the
``Transpose`` and ``MatMul`` nodes for ``dA = dC @ B^T`` and ``dB = A^T @ dC`` and
accumulates each into the corresponding variable.

What worked
^^^^^^^^^^^

* Returning a ``FunctionProto`` instead of numbers kept the gradient pass
  independent of any execution backend; the same runtime that runs the forward
  graph runs the backward graph.
* The ``(domain, op_type)`` registry made new operators additive: a caller can
  ``RegisterGradientFunction`` before invoking the pass without touching the
  algorithm.
* Expressing every rule with standard ONNX operators meant the backward graph
  needed no new kernels — it reused the operator kernels already validated by
  the backend tests.
* Deferring the ``Add`` accumulation node until a second contribution actually
  arrives kept the generated graphs small and readable.

What remains
^^^^^^^^^^^^

Coverage is the set of operators with a registered rule; differentiating a
graph that uses an unregistered operator raises rather than guessing.  The
``inputs`` parameter is reserved for future gradient pruning based on graph
connectivity, and higher-order derivatives (differentiating the backward graph
again) are possible in principle but not exercised.

See also
++++++++

* :ref:`l-next-steps-kernels-backend-tests` — the operator kernels that execute
  the generated backward graph.
