.. _l-design-expressions:

Expressions
===========

The symbolic dimension-expression library is implemented in
``onnx_light/onnx_core/expressions/`` and exposed as the Python module
:mod:`onnx_light.onnx_core.expressions`.

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
    add_expr ::= mul_expr (('*' | '//' | '/: ' | '%') mul_expr)*
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
| ``/:``   | Exact (integer) division — see below             |
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

:func:`~onnx_light.onnx_core.expressions.simplify_expression` applies a fixed
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
   * - ``SimpleSimplifyTransformer``
     - Folds identities: ``x ^ x → x``, ``x + 0 → x``, ``x * 1 → x``,
       ``0 * x → 0``, ``x * 0 → 0``, etc.
   * - ``MulDivCancellerTransformer``
     - Collects all factors in a ``*`` / ``//`` chain and cancels common
       symbolic sub-expressions (e.g. ``2*x//x → 2``).  Applied twice, before
       and after ``ExactMulDivConstantFolderTransformer``.
   * - ``ExactMulDivConstantFolderTransformer``
     - Folds integer constants in ``*`` / ``//`` chains when the division
       is exact (e.g. ``1024*a//2 → 512*a``).
   * - ``DistributeFloorDivOverAddTransformer``
     - Distributes a floor division over an addition when every non-constant
       term in the numerator is an exact multiple of the denominator
       (e.g. ``(2*x + 4) // 2 → x + 2``).
   * - ``MaxToXorTransformer``
     - Rewrites ``Max(a, b)`` and ``max(a, b)`` → ``a ^ b``.
   * - ``ReorderCommutativeOpsTransformer``
     - Sorts operands of ``+`` and ``*`` alphabetically so that
       ``"b + a"`` and ``"a + b"`` reduce to the same canonical form.
   * - ``MaxIntTransformer``
     - Evaluates ``int_const ^ int_const`` at compile time (returns the
       larger of the two constants).
   * - ``FloorDivAddRingTransformer``
     - Collapses a complete ring of consecutive floor divisions that share a
       denominator ``d`` and symbolic numerator, summing the ``d`` terms with
       offsets ``0 .. d-1`` back to the symbolic part
       (e.g. ``x//2 + (x + 1)//2 → x``).

After two passes of this pipeline, a final
``ExpressionSimplifierAddVisitor`` walks the tree and collects a linear
combination ``{symbol → coefficient}``.  This lets ``a + b - a`` simplify
to ``b`` and ``3*x + 2*x`` simplify to ``5*x`` even across multiple
transformer passes.

If the linear combination reduces to a pure integer constant (no remaining
symbolic terms), the result is returned as an ``int64_t``; otherwise the
normalised sum is unparsed back to a string.

----

Floor-division semantics
------------------------

``//`` uses Python's floor-division semantics: it rounds down toward negative
infinity, so ``-1//2 == -1`` (unlike C++ integer ``/``, which truncates toward
zero).  It is also not exact division, so it does **not** commute with
multiplication.  A constant factor can be cancelled against the denominator
only when the numerator is provably an exact multiple of it.  This explains a
pair of expressions that look symmetric but simplify differently:

* ``(2*H)//2`` **simplifies to** ``H``.  The numerator ``2*H`` is always an
  even multiple of ``2``, so the division is exact for every integer ``H`` and
  ``ExactMulDivConstantFolderTransformer`` cancels the common factor.
* ``2*(H//2)`` is **left unchanged**.  Here the floor division ``H//2`` is
  evaluated first and discards the remainder, so multiplying the result by
  ``2`` only recovers ``H`` when ``H`` is even (for example ``2*(3//2) == 2``,
  not ``3``).  Because the equality does not hold for all integers, the
  simplifier must preserve the expression.

In general ``a*(x//a)`` equals ``x`` only when ``x`` is a multiple of ``a``,
whereas ``(a*x)//a`` always equals ``x``.  The simplifier is conservative and
never rewrites an expression unless the rewrite is valid for *every* integer
value of the symbolic dimensions.

Python's negative-value semantics also enable rewrites such as
``(a-3)//2 + 1 -> (a-1)//2`` because both sides evaluate to ``0`` when
``a = 2`` and remain equal for every integer ``a``.

----

Exact-division semantics (``/:``  )
-------------------------------------

The ``/:`` operator represents *exact* integer division: the caller asserts
that the division has no remainder (``a % b == 0``).  This additional
guarantee allows the simplifier to move the division freely across
multiplication, which is not possible for ``//``:

* ``(2*H)/:2`` simplifies to ``H`` (same as ``//``).
* ``2*(H/:2)`` **also simplifies to** ``H``.  Because the caller guarantees
  the division is exact, ``2*(H/:2) == (2*H)/:2 == H`` is valid for every
  integer ``H`` that is a multiple of ``2``.  The simplifier exploits this and
  cancels the common factor even though it appears *outside* the division.

Concretely, the difference between ``//`` and ``/:`` only matters when a
factor appears on the multiplicative spine *outside* the division:

.. code-block:: python

    from onnx_light.onnx_core.expressions import simplify_expression

    simplify_expression("2*(H//2)")   # "2*(H//2)"  — NOT simplified (not exact)
    simplify_expression("2*(H/:2)")   # "H"         — simplified (exact division)

The primary use case for ``/:`` is **Reshape shape inference**.  When a tensor
is reshaped, the total number of elements is preserved, so the product of the
input dimensions equals the product of the output dimensions.  If one output
dimension is ``-1`` (inferred), its value is the input product divided by the
product of all *known* output dimensions.  This division is always exact (it
yields an integer dimension), so using ``/:`` instead of ``//`` lets the
simplifier produce cleaner symbolic shapes, e.g. for a reshape of
``(batch, seq, 4)`` to ``(batch, seq, 2, 2)``:

* With ``//``: inferred dim = ``"batch*seq*4//(batch*seq*2)"`` (not simplified
  further because ``//`` blocks cancellation of ``batch*seq``).
* With ``/:`` : inferred dim = ``"batch*seq*4/:(batch*seq*2)"`` → ``"2"``
  (the common ``batch*seq`` factor is cancelled).

----

Unparser
--------

:func:`~onnx_light.onnx_core.expressions.simplify_expression` (and all other
functions that produce an expression string) use ``unparse()`` to convert
an AST back to a string.  The unparser inserts parentheses exactly where
required by the precedence rules above, so the output round-trips through
``parse()`` to an equivalent AST.

For example:

.. runpython::
    :showcode:

    from onnx_light.onnx_core.expressions import simplify_expression

    print(simplify_expression("(a + b) * c"))   # "(a+b)*c"  — parens kept (needed)
    print(simplify_expression("a * b + c"))     # "a*b+c"   — no parens (not needed)

----

Dimension operations
--------------------

The ``DimType`` type represents a tensor
dimension as either a concrete ``int`` or a symbolic ``str``.  The dimension
operation functions (:func:`~onnx_light.onnx_core.expressions.dim_add`,
:func:`~onnx_light.onnx_core.expressions.dim_sub`,
:func:`~onnx_light.onnx_core.expressions.dim_mul`,
:func:`~onnx_light.onnx_core.expressions.dim_div`,
:func:`~onnx_light.onnx_core.expressions.dim_exact_div`,
:func:`~onnx_light.onnx_core.expressions.dim_mod`,
:func:`~onnx_light.onnx_core.expressions.dim_max`,
:func:`~onnx_light.onnx_core.expressions.dim_min`,
:func:`~onnx_light.onnx_core.expressions.dim_multi_mul`) share a common pattern:

1. If both operands are integers, compute the result exactly and return an
   integer.
2. Otherwise, build an expression string ``"(a) op (b)"``, call
   :func:`~onnx_light.onnx_core.expressions.simplify_expression`, and return the
   result (still an integer if the simplifier reduces it fully, otherwise a
   string).

This ensures that symbolic arithmetic never accumulates unnormalised
intermediate expressions:

.. runpython::
    :showcode:

    from onnx_light.onnx_core.expressions import dim_add, dim_mul, dim_div, dim_exact_div

    print(dim_add("batch", 1))               # "1+batch"
    print(dim_mul(2, "seq_length"))          # "2*seq_length"
    print(dim_div("2*seq_length", 2))        # "seq_length"  (simplified)
    print(dim_div("2*n", "n"))               # 2  (int — fully reduced)
    print(dim_exact_div("batch*4", 2))       # "2*batch"  (/: allows cancellation)
    print(dim_exact_div("2*batch*seq", "batch*seq"))  # 2 (int — fully reduced)

----

Renaming
--------

Two renaming functions cover different use cases:

:func:`~onnx_light.onnx_core.expressions.rename_expression`
    Renames variable names according to a mapping, also converting
    ``Max(a, b)`` to ``a^b`` beforehand.  Raises ``RuntimeError`` on
    parse failure.  Intended for deterministic, one-shot renames where a
    parse error is truly unexpected.

:func:`~onnx_light.onnx_core.expressions.rename_dynamic_expression`
    Like :func:`~onnx_light.onnx_core.expressions.rename_expression`, but also applies a lightweight
    simplification pass and silently returns the original string on parse
    failure.  Intended for best-effort renaming during shape inference
    where the expression may occasionally be a raw ONNX node name rather
    than a real expression.

``rename_dynamic_dimensions``
    Higher-level helper: given a set of equivalence classes (dimension
    names that are known to be equal to each other) and a set of
    user-visible preferred names, it produces a mapping from all internal
    names to their canonical user-visible equivalents.  Names starting with
    a configurable ban prefix (default ``"DYN"``) are never selected as
    canonical targets.

----

Build layout
------------

The expressions library lives in ``onnx_light/onnx_core/expressions/`` and
is compiled as part of the ``lib_onnx_core`` CMake target (built ``SHARED``
for Python builds and ``STATIC`` for pure C++ consumers).  Every
higher-level library — including ``lib_onnx_shape`` (which depends on
``lib_onnx_core``) — therefore has the expressions code available, and it
is linked into the Python extensions so all callers that consume the
shape-inference library can use it.

The C++ header and implementation files live in:

.. code-block:: text

    onnx_light/onnx_core/expressions/
    ├── expressions.h    ← public API (AST types + all free functions)
    └── expressions.cc  ← full implementation (tokenizer, parser,
                             transformers, evaluator, unparser)

The Python module ``onnx_light.onnx_core.expressions`` wraps the C++ functions
exposed via the ``_onnxpycore.expressions`` nanobind submodule (defined in
``onnx_light/onnx_py/_onnxpy_core.cc``).

Python wrapper:

.. code-block:: text

    onnx_light/onnx_core/expressions.py  ← documented Python wrappers

----

API reference
-------------

* **C++ API**: :doc:`/api/cpp/onnx_core/expressions`
* **Python API**: :doc:`/api/python/onnx_core/expressions`
