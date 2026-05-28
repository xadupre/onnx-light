.. _l-design-expressions:

Symbolic expression library (``onnx_light.onnx_optim.expressions``)
====================================================================

This page describes the design of the symbolic dimension-expression library
introduced in ``onnx_light/onnx_optim/`` and exposed as the
Python module :mod:`onnx_light.onnx_optim.expressions`.

The library was ported from
`yobx/xexpressions <https://github.com/xadupre/yet-another-onnx-builder/tree/main/yobx/xexpressions>`_
and re-implemented in C++ for speed and to avoid a runtime Python dependency in
the shape-inference path.

Motivation
----------

ONNX shape inference and model transformation frequently need to manipulate
*symbolic dimension expressions* — strings such as ``"2*batch//batch"``,
``"CeilToInt(seq_len, 8)"``, or ``"cache_length + seq_length"`` — that encode
relationships between dynamic tensor dimensions.

A pure-string approach (regex substitution, ``eval``) is fragile.  An
AST-based approach allows systematic:

* **Simplification** — ``2*batch//batch`` → ``2``, ``a + b - a`` → ``b``.
* **Evaluation** — substitute concrete integer values and compute the result.
* **Renaming** — replace internal names (``s0``, ``s1``, …) with user-visible
  names (``batch``, ``seq_length``, …).
* **Arithmetic** — add, subtract, multiply, divide, and compare symbolic
  dimension values without losing the symbolic form when the result is still
  symbolic.

----

Expression grammar
------------------

The parser accepts a subset of Python arithmetic expressions:

.. code-block:: text

    expr     ::= xor_expr ('^' xor_expr)*
    xor_expr ::= and_expr ('&' and_expr)*
    and_expr ::= add_expr (('+' | '-') add_expr)*
    add_expr ::= mul_expr (('*' | '//' | '%') mul_expr)*
    mul_expr ::= unary
    unary    ::= ('+' | '-') unary | atom
    atom     ::= INTEGER | NAME | '(' expr ')' | NAME '(' arg_list ')'
    arg_list ::= expr (',' expr)*

Operator precedence (low → high):

+----------+--------------------------------------------------+
| Operator | Meaning                                          |
+==========+==================================================+
| ``^``    | Encodes ``max(a, b)``                            |
+----------+--------------------------------------------------+
| ``&``    | Encodes ``min(a, b)``                            |
+----------+--------------------------------------------------+
| ``+``    | Addition                                         |
+----------+--------------------------------------------------+
| ``-``    | Subtraction                                      |
+----------+--------------------------------------------------+
| ``*``    | Multiplication                                   |
+----------+--------------------------------------------------+
| ``//``   | Floor (integer) division                         |
+----------+--------------------------------------------------+
| ``%``    | Modulo                                           |
+----------+--------------------------------------------------+
| Unary    | Unary ``-`` / ``+``                              |
+----------+--------------------------------------------------+

``^`` and ``&`` borrow Python's bitwise-xor and bitwise-and syntax and
re-interpret them as *max* and *min* respectively, following the
``yobx/xexpressions`` convention.  This lets the simplifier represent max/min
without adding new operator tokens.

The function call syntax ``Name(arg, ...)`` is used for two built-in functions:

* ``CeilToInt(n, div)`` — ceiling division; the simplifier expands it to
  ``(n + div - 1) // div`` before other passes run.
* ``Max(a, b)`` / ``max(a, b)`` — rewritten to ``a^b`` before evaluation.

----

AST node types
--------------

The parsed expression is represented as a tree of ``Node`` sub-classes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``Constant``
     - Leaf: a signed 64-bit integer literal.
   * - ``Name``
     - Leaf: a symbolic variable reference (e.g. ``"batch"``).
   * - ``BinOp``
     - Interior: left ``op`` right, where ``op`` is one of ``BinOpKind``.
   * - ``UnaryOp``
     - Interior: unary ``+`` or ``-`` applied to a single operand.
   * - ``Call``
     - Interior: a named function call with a list of argument sub-trees.

All nodes are heap-allocated and owned through ``NodePtr``
(``std::unique_ptr<Node>``).  Every node provides a virtual ``clone()``
method for deep copying.

----

Simplification pipeline
-----------------------

:func:`~onnx_light.onnx_optim.expressions.simplify_expression` applies a fixed
sequence of AST transformers, then runs the sequence a second time to allow
multi-step cancellations to converge.

Each transformer is a pure tree-to-tree rewrite that produces a new
``NodePtr`` without mutating the original:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Transformer
     - What it does
   * - ``CeilToIntTransformer``
     - Rewrites ``CeilToInt(x, n)`` → ``(x + n - 1) // n``.
   * - ``MaxToXorTransformer``
     - Rewrites ``Max(a, b)`` and ``max(a, b)`` → ``a ^ b``.
   * - ``SimpleSimplifyTransformer``
     - Folds identities: ``x ^ x → x``, ``x + 0 → x``, ``x * 1 → x``,
       ``0 * x → 0``, ``x * 0 → 0``, etc.
   * - ``MulDivCancellerTransformer``
     - Collects all factors in a ``*`` / ``//`` chain and cancels common
       symbolic sub-expressions (e.g. ``2*x//x → 2``).
   * - ``ExactMulDivConstantFolderTransformer``
     - Folds integer constants in ``*`` / ``//`` chains when the division
       is exact (e.g. ``1024*a//2 → 512*a``).
   * - ``ReorderCommutativeOpsTransformer``
     - Sorts operands of ``+`` and ``*`` alphabetically so that
       ``"b + a"`` and ``"a + b"`` reduce to the same canonical form.
   * - ``MaxIntTransformer``
     - Evaluates ``int_const ^ int_const`` at compile time (returns the
       larger of the two constants).

After two passes of this pipeline, a final
``ExpressionSimplifierAddVisitor`` walks the tree and collects a linear
combination ``{symbol → coefficient}``.  This lets ``a + b - a`` simplify
to ``b`` and ``3*x + 2*x`` simplify to ``5*x`` even across multiple
transformer passes.

If the linear combination reduces to a pure integer constant (no remaining
symbolic terms), the result is returned as an ``int64_t``; otherwise the
normalised sum is unparsed back to a string.

----

Unparser
--------

:func:`~onnx_light.onnx_optim.expressions.simplify_expression` (and all other
functions that produce an expression string) use ``unparse()`` to convert
an AST back to a string.  The unparser inserts parentheses exactly where
required by the precedence rules above, so the output round-trips through
``parse()`` to an equivalent AST.

For example:

.. code-block:: python

    from onnx_light.onnx_optim.expressions import simplify_expression

    simplify_expression("(a + b) * c")   # "(a+b)*c"  — parens kept (needed)
    simplify_expression("a * b + c")     # "a*b+c"   — no parens (not needed)

----

Dimension operations
--------------------

The :class:`~onnx_light.onnx_optim.expressions.DimType` type represents a tensor
dimension as either a concrete ``int`` or a symbolic ``str``.  The dimension
operation functions (:func:`~onnx_light.onnx_optim.expressions.dim_add`,
:func:`~onnx_light.onnx_optim.expressions.dim_sub`,
:func:`~onnx_light.onnx_optim.expressions.dim_mul`,
:func:`~onnx_light.onnx_optim.expressions.dim_div`,
:func:`~onnx_light.onnx_optim.expressions.dim_mod`,
:func:`~onnx_light.onnx_optim.expressions.dim_max`,
:func:`~onnx_light.onnx_optim.expressions.dim_min`,
:func:`~onnx_light.onnx_optim.expressions.dim_multi_mul`) share a common pattern:

1. If both operands are integers, compute the result exactly and return an
   integer.
2. Otherwise, build an expression string ``"(a) op (b)"``, call
   :func:`~onnx_light.onnx_optim.expressions.simplify_expression`, and return the
   result (still an integer if the simplifier reduces it fully, otherwise a
   string).

This ensures that symbolic arithmetic never accumulates unnormalised
intermediate expressions:

.. code-block:: python

    from onnx_light.onnx_optim.expressions import dim_add, dim_mul, dim_div

    dim_add("batch", 1)          # "1+batch"
    dim_mul(2, "seq_length")     # "2*seq_length"
    dim_div("2*seq_length", 2)   # "seq_length"  (simplified)
    dim_div("2*n", "n")          # 2  (int — fully reduced)

----

Renaming
--------

Two renaming functions cover different use cases:

:func:`~onnx_light.onnx_optim.expressions.rename_expression`
    Renames variable names according to a mapping, also converting
    ``Max(a, b)`` to ``a^b`` beforehand.  Raises ``RuntimeError`` on
    parse failure.  Intended for deterministic, one-shot renames where a
    parse error is truly unexpected.

:func:`~onnx_light.onnx_optim.expressions.rename_dynamic_expression`
    Like ``rename_expression``, but also applies a lightweight
    simplification pass and silently returns the original string on parse
    failure.  Intended for best-effort renaming during shape inference
    where the expression may occasionally be a raw ONNX node name rather
    than a real expression.

:func:`~onnx_light.onnx_optim.expressions.rename_dynamic_dimensions`
    Higher-level helper: given a set of equivalence classes (dimension
    names that are known to be equal to each other) and a set of
    user-visible preferred names, it produces a mapping from all internal
    names to their canonical user-visible equivalents.  Names starting with
    a configurable ban prefix (default ``"DYN"``) are never selected as
    canonical targets.

----

Build layout
------------

The expressions library is compiled as part of the ``onnx_optim`` CMake
``STATIC`` target (``lib_onnx_optim``).  ``lib_onnx_optim`` depends on
``lib_onnx_op`` and is linked into the Python extension so that the
expressions code is available to all callers that consume the optimisation
library.

The C++ header and implementation files live in:

.. code-block:: text

    onnx_light/onnx_optim/
    ├── expressions.h    ← public API (AST types + all free functions)
    └── expressions.cc  ← full implementation (tokenizer, parser,
                             transformers, evaluator, unparser)

The Python module ``onnx_light.onnx_optim.expressions`` wraps the C++ functions
exposed via the ``_onnxpy.expressions`` nanobind submodule (defined in
``onnx_light/onnx_py/_onnxpy_submodules.cc``).

Python wrapper:

.. code-block:: text

    onnx_light/onnx_optim/expressions.py  ← documented Python wrappers

----

API reference
-------------

* **C++ API**: :doc:`/api/cpp/onnx_optim/expressions`
* **Python API**: :doc:`/api/python/optim/expressions`
