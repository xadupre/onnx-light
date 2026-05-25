Expressions
===========

.. currentmodule:: onnx_light.onnx_optim.expressions

Symbolic dimension-expression utilities exposed by the
:mod:`onnx_light.onnx_optim.expressions` module.  See
:ref:`l-design-expressions` for the design overview.

Operator tokens
+++++++++++++++

The parser recognises the following operator tokens (sorted alphabetically):

* ``%`` — modulo
* ``&`` — encodes ``min(a, b)``
* ``(``, ``)`` — grouping
* ``*`` — multiplication
* ``+`` — addition (also unary plus)
* ``,`` — argument separator
* ``-`` — subtraction (also unary minus)
* ``//`` — floor (integer) division
* ``^`` — encodes ``max(a, b)``

The built-in function tokens are ``CeilToInt``, ``Max`` and ``max``.

Available simplifications
+++++++++++++++++++++++++

:func:`simplify_expression` applies the following transformers (in order),
twice, followed by a final linear-combination pass:

* ``CeilToIntTransformer`` — rewrites ``CeilToInt(x, n)`` → ``(x + n - 1) // n``.
* ``MaxToXorTransformer`` — rewrites ``Max(a, b)`` and ``max(a, b)`` → ``a ^ b``.
* ``SimpleSimplifyTransformer`` — folds identities such as ``x ^ x → x``,
  ``x + 0 → x``, ``x * 1 → x``, ``0 * x → 0``.
* ``MulDivCancellerTransformer`` — cancels common symbolic factors in
  ``*`` / ``//`` chains (e.g. ``2*x//x → 2``).
* ``ExactMulDivConstantFolderTransformer`` — folds integer constants in
  ``*`` / ``//`` chains when the division is exact (e.g. ``1024*a//2 → 512*a``).
* ``ReorderCommutativeOpsTransformer`` — sorts operands of ``+`` and ``*``
  alphabetically so equivalent expressions share a canonical form.
* ``MaxIntTransformer`` — evaluates ``int_const ^ int_const`` at compile time.

Members
+++++++

.. automodule:: onnx_light.onnx_optim.expressions
   :members:
