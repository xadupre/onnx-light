"""Python unit tests for onnx_light.onnx.expressions.

These tests are translated from
https://github.com/xadupre/yet-another-onnx-builder/tree/main/unittests/xexpressions
"""

import unittest

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx.expressions import (
    simplify_expression,
    simplify_two_expressions,
    evaluate_expression,
    parse_expression_tokens,
    rename_expression,
    rename_dynamic_expression,
    dim_add,
    dim_sub,
    dim_mul,
    dim_multi_mul,
    dim_div,
    dim_mod,
    dim_max,
    dim_min,
)

# ─────────────────────────────────────────────────────────────────────────────
# test_simplify_expressions.py (adapted)
# ─────────────────────────────────────────────────────────────────────────────


class TestSimplifyExpressions(ExtTestCase):
    def test_simplify_expression(self):
        self.assertEqual(simplify_expression("x - y + y"), "x")
        self.assertEqual(simplify_expression("2*x + 3*x - x"), "4*x")
        self.assertEqual(simplify_expression("a + b - a"), "b")
        self.assertEqual(simplify_expression("5 + x - 2 + 3"), "x+6")
        self.assertEqual(simplify_expression("x - x"), "0")
        self.assertEqual(simplify_expression("Max(x,y)"), "x^y")
        self.assertEqual(str(simplify_expression("2*batch//batch")), "2")

    def test_simplify_expression_to_int(self):
        self.assertEqual(simplify_expression("2*batch//batch"), 2)
        self.assertEqual(simplify_expression("2^2"), 2)
        self.assertEqual(simplify_expression("2*batch//batch^2"), 2)

    def test_simplify_expression2(self):
        self.assertEqual(simplify_expression("5 + x - (2 + 3)"), "x")

    def test_simplify_expression3(self):
        self.assertEqual(simplify_expression("x - 1"), "x-1")
        self.assertEqual(simplify_expression("1 - x"), "-x+1")

    def test_simplify_expression4(self):
        self.assertEqual(simplify_expression("-1+1+length"), "length")
        self.assertEqual(simplify_expression("1+length-1"), "length")
        self.assertEqual(simplify_expression("-1+length+1"), "length")

    def test_simplify_two_expressions(self):
        self.assertEqual(
            simplify_two_expressions("s52+seq_length", "s52+s70"), {"s70": -1, "seq_length": 1}
        )
        self.assertEqual(simplify_two_expressions("e*2", "e+e"), {})

    def test_simplify_expression_bracket(self):
        self.assertEqual("x", simplify_expression("2*x//2"))
        self.assertEqual("x", simplify_expression("(2*x)//2"))
        self.assertEqual("x", simplify_expression("(x*y)//y"))
        self.assertEqual("x", simplify_expression("(x*(y+1))//(y+1)"))
        self.assertEqual("c//2", simplify_expression("((c)//(2))"))

    def test_simplify_expression_bracket_max(self):
        self.assertEqual("x^1+y", simplify_expression("(x)^(y+1)"))
        self.assertEqual("1+x^y", simplify_expression("(x+1)^(y)"))

    def test_simplify_add_sub(self):
        self.assertEqual("b+c", simplify_expression("b+c-CeilToInt(b+c,2)+CeilToInt(b+c,2)"))

    def test_simplify_function(self):
        self.assertEqual("(1+b+c)//2", simplify_expression("CeilToInt(b+c,2)"))

    def test_simplify_function_order(self):
        self.assertEqual("a+b", simplify_expression("b+a"))

    def test_simplify_function_order3(self):
        self.assertEqual("a+b+c", simplify_expression("c+b+a"))
        self.assertEqual("a+b+c", simplify_expression("b+c+a"))
        self.assertEqual("a+b+c", simplify_expression("a+c+b"))

    def test_simplify_function_floordiv_int(self):
        self.assertEqual("512*a", simplify_expression("1024*a//2"))
        self.assertEqual("a", simplify_expression("1024*a//1024"))
        self.assertEqual("a+b", simplify_expression("1024*(a+b)//1024"))
        self.assertEqual("2*a+2*b", simplify_expression("1024*(a+b)//1024*2"))

    def test_simplify_expression_negation(self):
        self.assertEqual("length", simplify_expression("-1+1+length"))
        self.assertEqual("x-1", simplify_expression("-1+x"))
        self.assertEqual("x", simplify_expression("-x+2*x"))


# ─────────────────────────────────────────────────────────────────────────────
# test_evaluate_expressions.py (adapted)
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateExpressions(ExtTestCase):
    def test_evaluate_expression(self):
        self.assertEqual(-1, evaluate_expression("x - y", {"x": 5, "y": 6}))
        self.assertEqual(-5, evaluate_expression("- x", {"x": 5}))

    def test_evaluate_expression_syntax_error(self):
        with self.assertRaises(RuntimeError):
            evaluate_expression("x +", {"x": 5})


# ─────────────────────────────────────────────────────────────────────────────
# test_rename_expressions.py (adapted)
# ─────────────────────────────────────────────────────────────────────────────


class TestRenameExpressions(ExtTestCase):
    def test_rename_expression2(self):
        self.assertEqual("B+seq_length", rename_expression("s52+seq_length", {"s52": "B"}))

    def test_rename_expression_max(self):
        cst = {
            "A": "s77",
            "s77": "A",
            "B": "s27",
            "s27": "B",
            "D": "s3",
            "s3": "D",
            "E": "s10",
            "s10": "E",
            "E^D": "Max(s10,s3)",
            "Max(s10,s3)": "E^D",
        }
        self.assertEqual("E^D", rename_expression("s10^s3", cst))
        self.assertEqual("E^D", rename_expression("Max(s10,s3)", cst))
        self.assertEqual("E^D", rename_dynamic_expression("Max(s10,s3)", cst))

    def test_rename_expression(self):
        self.assertEqual("B+seq_length", rename_expression("s52+seq_length", {"s52": "B"}))

    def test_parse_expression_tokens_syntax_error(self):
        invalid_expr = "a +"
        result = parse_expression_tokens(invalid_expr)
        self.assertEqual({invalid_expr}, result)

    def test_rename_dynamic_expression_syntax_error(self):
        invalid_expr = "a +"
        result = rename_dynamic_expression(invalid_expr, {"a": "b"})
        self.assertEqual(invalid_expr, result)

    def test_rename_dynamic_expression(self):
        replacements = {
            "DYN0": "DYN0",
            "batch": "batch",
            "cache_length": "cache_length",
            "s0": "batch",
            "s1": "seq_length",
            "s10": "batch",
            "s12": "batch",
            "s14": "batch",
            "s2": "batch",
            "s3": "DYN0",
            "s8": "batch",
            "seq_length": "seq_length",
            "s11": "cache_length",
            "s15": "cache_length",
            "s9": "cache_length",
            "s13": "cache_length",
        }
        expression = "s9+seq_length"
        renamed = rename_dynamic_expression(expression, replacements)
        self.assertEqual(renamed, "cache_length+seq_length")


# ─────────────────────────────────────────────────────────────────────────────
# test_operations.py (adapted)
# ─────────────────────────────────────────────────────────────────────────────


class TestDimOperations(ExtTestCase):
    # ------------------------------------------------------------------ dim_mul
    def test_dim_mul_int_int(self):
        self.assertEqual(dim_mul(3, 4), 12)

    def test_dim_mul_zero(self):
        self.assertEqual(dim_mul(0, 5), 0)

    def test_dim_mul_symbolic(self):
        result = dim_mul("n", 2)
        self.assertIsInstance(result, str)
        self.assertIn("n", result)
        self.assertIn("2", result)

    def test_dim_mul_both_symbolic(self):
        result = dim_mul("a", "b")
        self.assertIsInstance(result, str)
        self.assertIn("a", result)
        self.assertIn("b", result)

    # --------------------------------------------------------------- dim_multi_mul
    def test_dim_multi_mul_all_int(self):
        self.assertEqual(dim_multi_mul([2, 3, 4]), 24)

    def test_dim_multi_mul_single_int(self):
        self.assertEqual(dim_multi_mul([7]), 7)

    def test_dim_multi_mul_with_symbolic(self):
        result = dim_multi_mul([2, "n", 3])
        self.assertIsInstance(result, str)
        self.assertIn("n", result)

    # ------------------------------------------------------------------ dim_add
    def test_dim_add_int_int(self):
        self.assertEqual(dim_add(3, 4), 7)

    def test_dim_add_symbolic(self):
        result = dim_add("n", 1)
        self.assertIsInstance(result, str)
        self.assertIn("n", result)
        self.assertIn("1", result)

    # ------------------------------------------------------------------ dim_sub
    def test_dim_sub_int_int(self):
        self.assertEqual(dim_sub(10, 3), 7)

    def test_dim_sub_same_symbol(self):
        result = dim_sub("n", "n")
        self.assertEqual(str(result), "0")

    # ------------------------------------------------------------------ dim_div
    def test_dim_div_int_int_exact(self):
        self.assertEqual(dim_div(12, 4), 3)

    def test_dim_div_int_int_floor(self):
        self.assertEqual(dim_div(7, 2), 3)

    def test_dim_div_symbolic(self):
        result = dim_div("2*n", 2)
        self.assertEqual(str(result), "n")

    # ------------------------------------------------------------------ dim_mod
    def test_dim_mod_int_int(self):
        self.assertEqual(dim_mod(10, 3), 1)

    def test_dim_mod_exact(self):
        self.assertEqual(dim_mod(12, 4), 0)

    # ------------------------------------------------------------------ dim_max
    def test_dim_max_int_int(self):
        self.assertEqual(dim_max(7, 3), 7)
        self.assertEqual(dim_max(2, 9), 9)
        self.assertEqual(dim_max(5, 5), 5)

    def test_dim_max_same_symbol(self):
        result = dim_max("n", "n")
        self.assertEqual(str(result), "n")

    # ------------------------------------------------------------------ dim_min
    def test_dim_min_int_int(self):
        self.assertEqual(dim_min(2, 9), 2)
        self.assertEqual(dim_min(8, 3), 3)
        self.assertEqual(dim_min(4, 4), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
