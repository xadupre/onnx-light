"""Symbolic dimension expression utilities backed by a C++ library.

This package exposes an AST-based expression engine for simplifying,
evaluating, and renaming symbolic shape expressions such as those produced
during ONNX model shape inference.

Expressions are strings containing integer constants, symbolic variable names
(e.g. ``"batch"``, ``"seq_length"``), and the arithmetic operators ``+``,
``-``, ``*``, ``//`` (floor division), ``/:`` (exact division), ``%``
(modulo), ``^`` (max), and ``&`` (min).

Typical usage::

    from onnx_light.core.expressions import (
        simplify_expression,
        evaluate_expression,
        rename_expression,
        dim_add,
    )

    # Simplify a symbolic expression.
    result = simplify_expression("2*batch//batch")
    # result is 2 (int)

    result = simplify_expression("a + b - a")
    # result is "b"

    # Evaluate with concrete variable assignments.
    value = evaluate_expression("x + y", {"x": 3, "y": 5})
    # value is 8

    # Rename dimension variables.
    expr = rename_expression("s0 + seq_len", {"s0": "batch"})
    # expr is "batch+seq_len"

    # Arithmetic on dimensions (int or str).
    d = dim_add("batch", 1)
    # d is "1+batch"

The module is exposed as ``onnx_light.core.expressions``.
"""

from __future__ import annotations

from typing import TypeAlias

from ..onnx_py._onnxpyoptim import expressions as _C  # type: ignore[attr-defined]

CompareResult: TypeAlias = _C.CompareResult
ExpressionComparison: TypeAlias = _C.ExpressionComparison
DimRange: TypeAlias = _C.DimRange
INFINITY: str = _C.INFINITY


def simplify_expression(expr: "str | int") -> "str | int":
    """Simplifies a symbolic or numeric expression.

    Applies a pipeline of AST transformations:

    1. ``CeilToInt(x, n)`` is expanded to ``(x + n - 1) // n``.
    2. Identity folds: ``x^x → x``, ``x + 0 → x``, ``x * 1 → x``.
    3. Common symbolic factors in ``*``/``//`` chains are cancelled
       (e.g. ``2*x//x → 2``).
    4. Integer constants in mul/div chains are folded when the division is
       exact (e.g. ``1024*a//2 → 512*a``).
    5. ``Max(a, b)`` and ``max(a, b)`` are rewritten to ``a^b``.
    6. Commutative ``+`` and ``*`` chains are sorted alphabetically.
    7. ``int_const ^ int_const`` is evaluated as ``max``.

    The pipeline is applied twice to allow multi-step cancellations.  A
    final linear-combination visitor normalises the remaining sum of terms.

    :param expr: An expression string (e.g. ``"2*batch//batch"``) or an
        integer that is already simplified.
    :returns: An ``int`` when the expression reduces to a numeric constant,
        or a simplified ``str`` otherwise.  Returns ``expr`` unchanged when
        it contains syntax that the parser does not recognise.
    :rtype: str | int

    Examples::

        >>> simplify_expression("2*batch//batch")
        2
        >>> simplify_expression("a + b - a")
        'b'
        >>> simplify_expression("5 + x - 2 + 3")
        'x+6'
        >>> simplify_expression("CeilToInt(b+c, 2)")
        '(1+b+c)//2'
        >>> simplify_expression("1024*a//2")
        '512*a'
        >>> simplify_expression("b + a")
        'a+b'
    """
    return _C.simplify_expression(expr)


def simplify_two_expressions(expr1: str, expr2: str) -> "dict[str, int]":
    """Returns the non-zero coefficient map of ``expr1 - expr2``.

    Builds the expression ``expr1 - (expr2)``, runs a linear-combination
    visitor, and returns only those variable coefficients that are non-zero.
    An empty dict indicates that the two expressions are equal under linear
    arithmetic.

    :param expr1: The first expression string.
    :param expr2: The second expression string.
    :returns: A dict mapping each variable name (or sub-expression key) to
        its integer coefficient in ``expr1 - expr2``.  Zero-coefficient terms
        are omitted.
    :rtype: dict[str, int]

    Examples::

        >>> simplify_two_expressions("s52+seq_length", "s52+s70")
        {'s70': -1, 'seq_length': 1}
        >>> simplify_two_expressions("e*2", "e+e")
        {}
    """
    return _C.simplify_two_expressions(expr1, expr2)


def compare_expressions(expr1: str, expr2: str) -> "ExpressionComparison":
    """Compares *expr1* to *expr2* assuming all tokens are positive or null.

    Builds the linear combination of ``expr1 - expr2`` and reasons about its
    sign knowing that every symbolic token is greater than or equal to zero:

    * :attr:`CompareResult.Equal` when the difference is identically zero.
    * :attr:`CompareResult.Greater` when ``expr1 - expr2`` is strictly positive
      for every non-negative assignment of the tokens.
    * :attr:`CompareResult.Smaller` when ``expr1 - expr2`` is strictly negative
      for every non-negative assignment of the tokens.
    * :attr:`CompareResult.Unknown` otherwise (e.g. tokens with mixed-sign
      coefficients, or a zero constant term that allows equality when all tokens
      are zero).

    :param expr1: The first expression string.
    :param expr2: The second expression string.
    :returns: An :class:`ExpressionComparison` whose ``result`` is a
        :class:`CompareResult` and whose ``difference`` is the simplified value
        of ``expr2 - expr1`` (an ``int`` when numeric, otherwise a ``str``).
        The difference is most useful when ``result`` is
        :attr:`CompareResult.Unknown`.
    :rtype: ExpressionComparison
    :raises RuntimeError: If either expression cannot be parsed.

    Examples::

        >>> c = compare_expressions("a+1", "a")
        >>> c.result
        CompareResult.Greater
        >>> c.difference
        '-1'
        >>> compare_expressions("a", "b").result
        CompareResult.Unknown
    """
    return _C.compare_expressions(expr1, expr2)


def evaluate_expression(expression: str, context: "dict[str, int]") -> int:
    """Evaluates an expression given variable assignments.

    Supports signed 64-bit integer constants, variable references resolved
    via *context*, binary operators ``+``, ``-``, ``*``, ``//`` (floor
    division), ``%`` (modulo), ``^`` (max), ``&`` (min), unary ``-``, and
    the built-in ``CeilToInt(n, div)`` function (ceiling division). Floor
    division follows Python semantics and rounds down toward negative infinity,
    so ``-1//2 == -1``.

    :param expression: The expression string to evaluate.
    :param context: A mapping from variable name to its integer value.
    :returns: The integer result of evaluating the expression.
    :rtype: int
    :raises RuntimeError: If the expression has a syntax error, references an
        unknown variable, or contains an unsupported construct.

    Examples::

        >>> evaluate_expression("x - y", {"x": 5, "y": 6})
        -1
        >>> evaluate_expression("-x", {"x": 5})
        -5
        >>> evaluate_expression("-1//2", {})
        -1
        >>> evaluate_expression("CeilToInt(7, 2)", {})
        4
    """
    return _C.evaluate_expression(expression, context)


def parse_expression_tokens(expr: str) -> "set[str]":
    """Returns the set of variable names referenced in *expr*.

    Parses *expr* and collects every variable name (``Name`` AST node).
    If the expression has a syntax error the function returns ``{expr}``
    (a set containing the original string), matching the reference behaviour.

    :param expr: The expression string to scan.
    :returns: An unordered set of variable name strings.  Contains only
        *expr* itself when parsing fails.
    :rtype: set[str]

    Examples::

        >>> sorted(parse_expression_tokens("a + b * c"))
        ['a', 'b', 'c']
        >>> parse_expression_tokens("a +")
        {'a +'}
    """
    return _C.parse_expression_tokens(expr)


def rename_expression(expr: str, mapping: "dict[str, str]") -> str:
    """Renames variables in *expr* according to *mapping*.

    Also converts ``Max(a, b)`` calls to the ``a^b`` form before renaming so
    that composite keys such as ``"E^D"`` can be matched.  The result has all
    spaces removed.

    :param expr: The expression string to rename.
    :param mapping: A mapping from old variable name to new variable name.
    :returns: The renamed expression string (no spaces).
    :rtype: str
    :raises RuntimeError: If *expr* cannot be parsed.

    Examples::

        >>> rename_expression("s52+seq_length", {"s52": "B"})
        'B+seq_length'
        >>> rename_expression("Max(s10, s3)", {"s10": "E", "s3": "D"})
        'E^D'
    """
    return _C.rename_expression(expr, mapping)


def rename_dynamic_expression(expression: str, replacements: "dict[str, str]") -> str:
    """Renames variables in *expression* and simplifies the result.

    Applies the following pipeline in order:

    1. Parse *expression*.
    2. Rewrite ``Max(a, b)`` → ``a ^ b``.
    3. Apply the *replacements* mapping.
    4. Apply a lightweight simplification pass.
    5. Unparse and strip spaces.

    Returns *expression* unchanged if it has a syntax error.

    :param expression: The expression string to transform.
    :param replacements: A mapping from old variable name to new variable name.
    :returns: The renamed and simplified expression string (no spaces), or
        *expression* unchanged on parse failure.
    :rtype: str

    Examples::

        >>> rename_dynamic_expression("s9+seq_length",
        ...     {"s9": "cache_length", "seq_length": "seq_length"})
        'cache_length+seq_length'
        >>> rename_dynamic_expression("a +", {"a": "b"})
        'a +'
    """
    return _C.rename_dynamic_expression(expression, replacements)


def dim_add(a: "int | str", b: "int | str") -> "int | str":
    """Adds two dimensions.

    Returns ``a + b`` as an ``int`` when both operands are integers;
    otherwise builds ``"(a)+(b)"`` and simplifies symbolically.

    :param a: The first dimension (integer or symbolic string).
    :param b: The second dimension (integer or symbolic string).
    :returns: The sum as an ``int`` when both are concrete, or as a
        simplified ``str`` otherwise.
    :rtype: int | str

    Examples::

        >>> dim_add(3, 4)
        7
        >>> dim_add("n", 1)
        'n+1'
    """
    return _C.dim_add(a, b)


def dim_sub(a: "int | str", b: "int | str") -> "int | str":
    """Subtracts *b* from *a*.

    Returns ``a - b`` as an ``int`` when both operands are integers;
    otherwise builds ``"(a)-(b)"`` and simplifies symbolically.

    :param a: The minuend.
    :param b: The subtrahend.
    :returns: The difference as an ``int`` when both are concrete, or as a
        simplified ``str`` otherwise.
    :rtype: int | str

    Examples::

        >>> dim_sub(10, 3)
        7
        >>> str(dim_sub("n", "n"))
        '0'
    """
    return _C.dim_sub(a, b)


def dim_mul(a: "int | str", b: "int | str") -> "int | str":
    """Multiplies two dimensions.

    Returns ``a * b`` as an ``int`` when both operands are integers;
    otherwise builds ``"(a)*(b)"`` and simplifies symbolically.

    :param a: The first dimension.
    :param b: The second dimension.
    :returns: The product as an ``int`` when both are concrete, or as a
        simplified ``str`` otherwise.
    :rtype: int | str

    Examples::

        >>> dim_mul(3, 4)
        12
        >>> dim_mul(0, 5)
        0
    """
    return _C.dim_mul(a, b)


def dim_multi_mul(*args: "int | str") -> "int | str":
    """Multiplies a sequence of dimensions.

    Computes the product of all positional arguments.  If every argument is an
    ``int`` the result is an exact integer product; otherwise the expression
    is built and simplified symbolically.  Returns ``1`` for an empty argument
    list.

    :param args: Zero or more dimension values (each an ``int`` or ``str``).
    :returns: The product as an ``int`` when all operands are concrete, or as
        a simplified ``str`` otherwise.
    :rtype: int | str

    Examples::

        >>> dim_multi_mul(2, 3, 4)
        24
        >>> dim_multi_mul(7)
        7
    """
    return _C.dim_multi_mul(list(args))


def dim_div(a: "int | str", b: "int | str") -> "int | str":
    """Floor-divides *a* by *b*.

    Returns ``a // b`` as an ``int`` when both operands are integers;
    otherwise builds ``"(a)//(b)"`` and simplifies symbolically.

    :param a: The dividend.
    :param b: The divisor.
    :returns: The floor-division result as an ``int`` when both are concrete,
        or as a simplified ``str`` otherwise.
    :rtype: int | str

    Examples::

        >>> dim_div(12, 4)
        3
        >>> dim_div(7, 2)
        3
        >>> dim_div("2*n", 2)
        'n'
    """
    return _C.dim_div(a, b)


def dim_exact_div(a: "int | str", b: "int | str") -> "int | str":
    """Exactly divides *a* by *b*, asserting the division has no remainder.

    Returns ``a / b`` as an ``int`` when both operands are integers.  Raises
    ``RuntimeError`` when both are integers but the division is not exact
    (``a % b != 0``) or *b* is zero.  Otherwise builds ``"(a)/:(b)"`` and
    simplifies symbolically.

    The ``/:`` operator differs from ``//`` (floor division) in that the caller
    guarantees the result is always an integer.  This allows the simplifier to
    freely distribute the division across multiplication:
    ``c * (a /: b) == (c * a) /: b``.

    Typical use case: Reshape shape inference, where preserving the total
    element count means the input dimension product is an exact multiple of all
    known output dimensions.

    :param a: The dividend.
    :param b: The divisor.
    :returns: The quotient as an ``int`` when both are concrete, or as a
        simplified ``str`` otherwise.
    :rtype: int | str
    :raises RuntimeError: When both operands are integers and the division is
        not exact or *b* is zero.

    Examples::

        >>> dim_exact_div(12, 4)
        3
        >>> dim_exact_div("2*n", 2)
        'n'
        >>> dim_exact_div("batch*4", 2)
        '2*batch'
    """
    return _C.dim_exact_div(a, b)


def dim_mod(a: "int | str", b: "int | str") -> "int | str":
    """Computes *a* modulo *b*.

    Returns ``a % b`` as an ``int`` when both operands are integers;
    otherwise builds ``"(a)%(b)"`` and simplifies symbolically.

    :param a: The dividend.
    :param b: The divisor.
    :returns: The remainder as an ``int`` when both are concrete, or as a
        simplified ``str`` otherwise.
    :rtype: int | str

    Examples::

        >>> dim_mod(10, 3)
        1
        >>> dim_mod(12, 4)
        0
    """
    return _C.dim_mod(a, b)


def dim_max(a: "int | str", b: "int | str") -> "int | str":
    """Returns the maximum of two dimensions.

    Returns ``max(a, b)`` as an ``int`` when both operands are integers;
    otherwise builds ``"(a)^(b)"`` (the ``^`` encoding of max) and simplifies
    symbolically.

    :param a: The first dimension.
    :param b: The second dimension.
    :returns: The maximum as an ``int`` when both are concrete, or as a
        simplified ``str`` otherwise.
    :rtype: int | str

    Examples::

        >>> dim_max(7, 3)
        7
        >>> dim_max(2, 9)
        9
        >>> str(dim_max("n", "n"))
        'n'
    """
    return _C.dim_max(a, b)


def dim_min(a: "int | str", b: "int | str") -> "int | str":
    """Returns the minimum of two dimensions.

    Returns ``min(a, b)`` as an ``int`` when both operands are integers;
    otherwise builds ``"(a)&(b)"`` (the ``&`` encoding of min) and simplifies
    symbolically.

    :param a: The first dimension.
    :param b: The second dimension.
    :returns: The minimum as an ``int`` when both are concrete, or as a
        simplified ``str`` otherwise.
    :rtype: int | str

    Examples::

        >>> dim_min(2, 9)
        2
        >>> dim_min(8, 3)
        3
    """
    return _C.dim_min(a, b)


def dim_ranges_from_expressions(
    equalities: "list[tuple[str, str]]", tokens: "list[str] | None" = None
) -> "dict[str, DimRange]":
    """Infers dimension ranges from a set of equality constraints.

    Each element of *equalities* is a ``(lhs, rhs)`` pair that represents the
    equality ``lhs == rhs`` between two dimension expressions, as arises when
    matching the shapes of two ONNX inputs or outputs.

    For every variable that appears as the leaf of a floor-division chain on
    either side of an equality, the function derives tight integer bounds:

    * **Direct equality** ``var == value`` (chain length 0):
      ``DimRange(lower=value, upper=value)``.  The variable is **exactly
      constrained** to ``value``; ``upper`` equals ``lower`` and there is no
      separate upper bound beyond the equality itself.
    * **Floor-division chain** ``var // d₁ // … // dₙ == value``
      (all dᵢ are strictly positive integer literals, P = d₁·…·dₙ):
      ``DimRange(lower=P·value, upper=P·value + P − 1)``.  This is a proper
      range with a finite upper bound whenever P > 1.

    Variables that do not match any supported pattern are **absent** from the
    returned dict; they have no determinable range and are not represented as
    unbounded entries.

    The symmetry of each equality is exploited; both sides are tried as the
    chain side.  Both expressions are simplified before pattern matching.

    :param equalities: A list of ``(lhs, rhs)`` string pairs, each
        representing the equality ``lhs == rhs``.
    :param tokens: Optional list of variable names to include in the result.
        When *None* (the default), ranges for all recognized variables are
        returned.
    :returns: A ``dict`` mapping each variable name to a :class:`DimRange`
        with inclusive ``lower`` and ``upper`` bounds.  Each bound is an
        ``int`` when concrete or a simplified ``str`` when symbolic.
        When ``upper`` equals ``lower`` the variable is exactly constrained
        (no separate upper bound).  A variable absent from the dict has no
        determinable range.
    :rtype: dict[str, DimRange]

    Examples::

        >>> r = dim_ranges_from_expressions([("a", "d//5")])
        >>> r["a"].lower, r["a"].upper   # exact: upper equals lower, no separate upper bound
        ('d//5', 'd//5')
        >>> r["d"].lower, r["d"].upper   # proper range: upper > lower
        ('5*a', '4+5*a')
        >>> r = dim_ranges_from_expressions([("a", "3")])
        >>> r["a"].lower, r["a"].upper   # exact numeric constraint
        (3, 3)
        >>> # Variables without a recognized pattern are absent:
        >>> "x" not in dim_ranges_from_expressions([("x+y", "5")])
        True
    """
    return _C.dim_ranges_from_expressions(equalities, tokens if tokens is not None else [])
