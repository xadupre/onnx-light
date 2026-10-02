.. _l-next-steps-gradient-completion:

Complete symbolic gradient support
==================================

:Date: 2026-10
:Updated: 2026-10-02

**started**

Objective
+++++++++

Finishes the symbolic gradient implementation introduced by
:ref:`l-next-steps-gradient`. The existing pass provides a native registry,
C++ and Python bindings, and 23 operator rules, but most tests only inspect the
generated node list or check that construction does not throw. Completion
means that the generated ``FunctionProto`` is a valid, executable ONNX
backward graph whose numerical results agree with the forward graph.

This plan does not require a gradient for every ONNX operator. Every operator
in the supported training corpus must instead have one machine-readable
disposition: differentiable with a tested rule, intentionally
non-differentiable, or deferred with a precise missing prerequisite.

Current baseline
++++++++++++++++

At the 2026-10-02 source revision, ``DefaultGradRegistry`` contains 23 rules:

* **math** — ``Add``, ``Sub``, ``Mul``, ``Div``, ``Neg``, ``Gemm`` and
  ``MatMul``;
* **neural network** — ``Conv``, ``Relu``, ``Sigmoid``, ``Tanh`` and seven
  normalization operators;
* **tensor** — ``Identity``, ``Reshape`` and ``Transpose``;
* **reduction** — ``ReduceMean`` and ``ReduceSum``.

The original API and registry milestones are complete. The remaining work is
correctness and executable coverage:

* ``GradientOfNodes`` currently ignores its ``inputs`` and ``initializers``
  arguments and does not derive the forward values that the backward graph
  must receive or recompute.
* backward dispatch supplies only the first output gradient found for a node,
  so multi-output operators have no complete contract;
* broadcasting is not reduced back to each input shape for elementwise,
  ``MatMul`` and ``Gemm`` gradients;
* ``Gemm`` attributes, vector/batched ``MatMul``, reduction axes and reduction
  element counts are not fully represented;
* several generated reduction nodes use attribute-form axes while the returned
  function declares opset 21, where axes are tensor inputs;
* registry tests prove that a rule emits nodes, not that the function validates,
  runs, or computes the derivative.

Completion contract
+++++++++++++++++++

``GradientOfNodes`` and ``GradientOfFunction`` must produce the same backward
interface from the same logical forward graph. The implementation must:

* validate unique value producers, topological order, domains and opsets, and
  ensure ``xs``, ``zs`` and ``y`` refer to legal, non-conflicting values;
* compute only the intersection of values reachable from ``y`` and values
  depending on ``xs``;
* use ``inputs`` and ``initializers`` to distinguish runtime values from
  constants instead of discarding them;
* add required saved forward values to the function interface in a stable
  order. Values already present in ``xs`` or ``zs`` are not duplicated;
* accept one gradient contribution per non-empty node output. The ``GradFn``
  contract identifies output indices rather than silently selecting the first
  output;
* accumulate repeated contributions deterministically and return a correctly
  shaped zero for a requested but disconnected ``x``;
* emit nodes valid for the declared opset and preserve the forward node domain
  where a domain-specific rule is registered;
* reject unsupported types, malformed attributes and missing rules with the
  node name, domain, operator, opset and source value in the error.

Saved tensors are explicit function inputs for this milestone. Recomputing
forward subgraphs may be added later as an opt-in memory policy, but must not be
an implicit side effect of a gradient rule.

Shared gradient primitives
++++++++++++++++++++++++++

Add a small internal builder used by every rule. It owns unique names, typed
constants, schema-correct axes inputs, shape queries and these operations:

``ReduceSumToShape(value, like)``
  Reduces broadcast dimensions and reshapes the result to exactly the shape of
  ``like``. Elementwise binary, ``Gemm`` bias and batched ``MatMul`` rules use
  this instead of returning the broadcast output shape.

``ExpandReductionGradient(value, input, axes, keepdims)``
  Normalizes positive and negative axes, restores removed dimensions and
  expands to the input shape. ``ReduceMean`` divides by the product of the
  reduced dimensions, not by the size of the whole input.

``LastTwoDimsTranspose(value)``
  Swaps only the final two dimensions. This is required for batched
  ``MatMul``; the default ``Transpose`` reverses every dimension.

``CastConstantLike(value, like)``
  Emits scalar constants in the runtime dtype of ``like`` and keeps gradient
  arithmetic valid for FLOAT, DOUBLE, FLOAT16 and BFLOAT16.

The builder validates every emitted node against the selected ONNX schema
before the function is returned. Gradient rules do not construct
opset-sensitive attributes directly.

Ordered implementation batches
+++++++++++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 10 25 40 25

   * - Batch
     - Scope
     - Required work
     - Exit evidence
   * - 1
     - Core pass and the existing 23 rules
     - Implement the completion contract and shared primitives. Correct
       broadcasting, opset-21 reduction inputs, ``Gemm`` attributes,
       vector/batched ``MatMul``, arbitrary reduction axes, ``Conv``
       attributes, normalization axes and optional inputs.
     - Every existing rule validates and executes; representative scalar,
       broadcast, dynamic-shape and batched cases pass numerical checks.
   * - 2
     - Common differentiable building blocks
     - Add rules for unary math, ``Pow``, ``Min``/``Max``, ``Where``,
       ``Concat``, ``Split``, ``Flatten``, ``Squeeze``, ``Unsqueeze``,
       ``Slice``, ``Gather``, ``Expand``, ``Softmax``, ``LogSoftmax``,
       pooling and common activation operators.
     - MLP, CNN and transformer projection blocks execute end to end without
       an unclassified operator.
   * - 3
     - Losses and training graphs
     - Add numerically stable gradients for
       ``SoftmaxCrossEntropyLoss``, ``NegativeLogLikelihoodLoss`` and the
       attention/training operators selected by the model corpus. Cover
       optional outputs and saved forward values.
     - Linear regression, classifier, convolutional and attention training
       steps reduce their loss with native forward and backward execution.
   * - 4
     - Coverage closure
     - Generate the operator disposition inventory from schemas, registry and
       backend cases. Classify shape/index outputs, random generators,
       comparisons, control flow, quantized types and custom domains.
     - No operator in the declared corpus is unclassified; unsupported paths
       fail before emitting a partial backward function.

An operator is added only with an executable numerical test. Increasing the
registry count without validating the result is not progress toward
completion.

Numerical validation
++++++++++++++++++++

Create a reusable C++ gradient test harness. For each case it:

1. builds and validates the forward graph;
2. generates and validates the backward ``FunctionProto``;
3. executes forward and backward with the native runtime;
4. computes a scalar objective ``sum(y * dy)``;
5. compares each analytical gradient with a central finite-difference result;
6. verifies output dtype, shape, finite values and deterministic naming.

Use DOUBLE for the strict oracle and FLOAT for the runtime contract. FLOAT16
and BFLOAT16 use wider tolerances and comparison through a FLOAT reference.
Non-smooth operators use values away from discontinuities and document the
chosen subgradient at ties or zero.

The matrix includes scalars, empty tensors, rank-one inputs, broadcast leading
dimensions, repeated inputs, fan-out/fan-in, optional inputs, negative axes,
zero-sized dimensions and dynamic shapes. Invalid graphs and unsupported
integer, string, sequence, optional or quantized differentiation paths must
fail with focused diagnostics.

The existing backend sweep remains useful for discovery, but its acceptance
condition changes from "does not throw" to "the generated function validates
and executes" wherever the case supplies floating-point inputs. Each
registered rule also has at least one finite-difference case independent of
the backend data.

Operator disposition inventory
+++++++++++++++++++++++++++++++

Generate a checked-in report with one row per latest standard-domain schema
and every registered custom-domain schema. Each row records:

* domain, operator and supported opset range;
* ``differentiable``, ``non_differentiable`` or ``deferred``;
* differentiable input and output indices;
* rule name and numerical test identifiers;
* dtype, rank, broadcasting and attribute limitations;
* the reason and issue for a deferred rule.

Shape, index and condition inputs can be non-differentiable even when an
operator has differentiable data inputs. Random generators remain
non-differentiable because their output is not a deterministic function of a
tensor input. Integer, string and quantized storage values do not silently
receive floating-point gradients.

Acceptance
++++++++++

This next step is complete when:

1. All generated backward functions pass native ONNX validation and shape
   inference for the supported opset range.
2. The existing 23 rules pass executable finite-difference tests covering
   broadcasting, attributes, optional inputs, ranks and supported dtypes.
3. MLP, convolutional, normalization, loss and attention training graphs run a
   full forward/backward/update step and reduce loss over multiple iterations.
4. Multi-output nodes, shared values, repeated inputs, disconnected ``xs`` and
   saved forward tensors have explicit tested semantics.
5. C++ and Python bindings expose the same deterministic function interface
   and the same diagnostics.
6. Sanitizer, reduced-build and full-build configurations preserve their
   intended gradient-library boundaries.
7. The generated disposition inventory contains no unclassified operator in
   the declared training corpus and cannot drift from ``DefaultGradRegistry``.
8. Documentation lists supported operators and limitations from that inventory
   rather than a manually maintained count.

Implementation unit
+++++++++++++++++++

Use one pull request for the core contract and test harness, then one pull
request per operator family. Each operator pull request includes its registry
entry, schema/opset coverage, numerical cases, invalid-input cases and
inventory update. Loss and end-to-end model pull requests start only after the
shared broadcasting and reduction primitives are complete.
