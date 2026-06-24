// SPDX-License-Identifier: Apache-2.0
//
// C++ unit tests for onnx_light::onnx_optim::expressions.
// These test cases are translated from
// https://github.com/xadupre/yet-another-onnx-builder/tree/main/unittests/xexpressions

#include "onnx_optim/expressions.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

using namespace onnx_light::onnx_optim::expressions;

// ─────────────────────────── helpers ─────────────────────────────────────

// Extract string from a SimplifyResult (asserts it IS a string).
static std::string get_str(const SimplifyResult &r) {
  EXPECT_TRUE(std::holds_alternative<std::string>(r));
  return std::get<std::string>(r);
}

// Extract int64 from a SimplifyResult (asserts it IS an int).
static int64_t get_int(const SimplifyResult &r) {
  EXPECT_TRUE(std::holds_alternative<int64_t>(r));
  return std::get<int64_t>(r);
}

// ═══════════════════════════════════════════════════════════════════════════
// test_simplify_expressions.py (adapted)
// ═══════════════════════════════════════════════════════════════════════════

TEST(SimplifyExpressions, SimplifyExpression_basic) {
  EXPECT_EQ(get_str(simplify_expression("x - y + y")), "x");
  EXPECT_EQ(get_str(simplify_expression("2*x + 3*x - x")), "4*x");
  EXPECT_EQ(get_str(simplify_expression("a + b - a")), "b");
  EXPECT_EQ(get_str(simplify_expression("5 + x - 2 + 3")), "x+6");
  EXPECT_EQ(get_str(simplify_expression("x - x")), "0");
  EXPECT_EQ(get_str(simplify_expression("Max(x,y)")), "x^y");
  EXPECT_EQ(simplify_result_to_string(simplify_expression("2*batch//batch")), "2");
}

TEST(SimplifyExpressions, SimplifyExpression_to_int) {
  EXPECT_EQ(get_int(simplify_expression("2*batch//batch")), 2);
  EXPECT_EQ(get_int(simplify_expression("2^2")), 2);
  EXPECT_EQ(get_int(simplify_expression("2*batch//batch^2")), 2);
}

TEST(SimplifyExpressions, SimplifyExpression2) {
  EXPECT_EQ(get_str(simplify_expression("5 + x - (2 + 3)")), "x");
}

TEST(SimplifyExpressions, SimplifyExpression3) {
  EXPECT_EQ(get_str(simplify_expression("x - 1")), "x-1");
  EXPECT_EQ(get_str(simplify_expression("1 - x")), "-x+1");
}

TEST(SimplifyExpressions, SimplifyExpression4) {
  EXPECT_EQ(get_str(simplify_expression("-1+1+length")), "length");
  EXPECT_EQ(get_str(simplify_expression("1+length-1")), "length");
  EXPECT_EQ(get_str(simplify_expression("-1+length+1")), "length");
}

TEST(SimplifyExpressions, SimplifyTwoExpressions) {
  auto r1 = simplify_two_expressions("s52+seq_length", "s52+s70");
  EXPECT_EQ(r1.size(), 2u);
  EXPECT_EQ(r1.at("s70"), -1);
  EXPECT_EQ(r1.at("seq_length"), 1);

  auto r2 = simplify_two_expressions("e*2", "e+e");
  EXPECT_TRUE(r2.empty());
}

TEST(SimplifyExpressions, CompareExpressions) {
  // Equal expressions.
  auto eq = compare_expressions("a+b", "b+a");
  EXPECT_EQ(eq.result, CompareResult::Equal);
  EXPECT_EQ(simplify_result_to_string(eq.difference), "0");

  // Pure constant difference.
  EXPECT_EQ(compare_expressions("5", "3").result, CompareResult::Greater);
  EXPECT_EQ(compare_expressions("3", "5").result, CompareResult::Smaller);

  // Strictly greater / smaller with symbolic tokens.
  EXPECT_EQ(compare_expressions("a+1", "a").result, CompareResult::Greater);
  EXPECT_EQ(compare_expressions("a", "a+1").result, CompareResult::Smaller);
  EXPECT_EQ(compare_expressions("a+b+1", "a").result, CompareResult::Greater);

  // Unknown: mixed-sign coefficients; difference is expr2 - expr1.
  auto un = compare_expressions("a", "b");
  EXPECT_EQ(un.result, CompareResult::Unknown);
  EXPECT_EQ(simplify_result_to_string(un.difference), "b-a");

  // Unknown: same token sign but zero constant (equal when the token is null).
  EXPECT_EQ(compare_expressions("2*a", "a").result, CompareResult::Unknown);

  // Malformed input propagates a parse error.
  EXPECT_THROW(compare_expressions("a", "b +"), std::runtime_error);
}

TEST(SimplifyExpressions, SimplifyExpression_bracket) {
  EXPECT_EQ(get_str(simplify_expression("2*x//2")), "x");
  EXPECT_EQ(get_str(simplify_expression("(2*x)//2")), "x");
  EXPECT_EQ(get_str(simplify_expression("(x*y)//y")), "x");
  EXPECT_EQ(get_str(simplify_expression("(x*(y+1))//(y+1)")), "x");
  EXPECT_EQ(get_str(simplify_expression("((c)//(2))")), "c//2");
}

TEST(SimplifyExpressions, SimplifyExpression_distribute_floordiv) {
  EXPECT_EQ(get_str(simplify_expression("(2*b+2*c)//2")), "b+c");
  EXPECT_EQ(get_str(simplify_expression("(4*a+2*b)//2")), "2*a+b");
  EXPECT_EQ(get_str(simplify_expression("(2*a-2*b)//2")), "a-b");
  EXPECT_EQ(get_str(simplify_expression("(2*a+4*b+6*c)//2")), "2*b+3*c+a");
  // Constant residual smaller than divisor: 1//2 == 0, so (1+2b+2c)//2 → b+c.
  EXPECT_EQ(get_str(simplify_expression("(2*b+2*c+1)//2")), "b+c");
  EXPECT_EQ(get_str(simplify_expression("(1+2*b+2*c)//2")), "b+c");
  // Constant residual equal to or greater than divisor folds into an
  // additive integer constant.
  EXPECT_EQ(get_str(simplify_expression("(2*b+2*c+3)//2")), "b+c+1");
  EXPECT_EQ(get_str(simplify_expression("(sequence-10)//5")), "sequence//5-2");
  EXPECT_EQ(get_str(simplify_expression("(sequence+10)//5")), "sequence//5+2");
  EXPECT_EQ(get_str(simplify_expression("(2*b+3*c)//2")), "(2*b+3*c)//2");
}

TEST(SimplifyExpressions, SimplifyExpression_floordiv_not_exact) {
  // `//` is floor division, so a factor cannot cross the division boundary:
  // a*(x//a) == x only when x is a multiple of a, whereas (a*x)//a == x always.
  EXPECT_EQ(get_str(simplify_expression("2*(H//2)")), "2*(H//2)");
  EXPECT_EQ(get_str(simplify_expression("(2*H)//2")), "H");
  EXPECT_EQ(get_str(simplify_expression("3*(H//3)")), "3*(H//3)");
  EXPECT_EQ(get_str(simplify_expression("(3*H)//3")), "H");
  EXPECT_EQ(get_str(simplify_expression("2*(n//2)+1")), "2*(n//2)+1");
  // Concrete counter-example: 2*(3//2) == 2, not 3.
  EXPECT_EQ(evaluate_expression("2*(H//2)", {{"H", 3}}), 2);
  EXPECT_EQ(evaluate_expression("(2*H)//2", {{"H", 3}}), 3);
}

TEST(SimplifyExpressions, SimplifyExpression_bracket_max) {
  EXPECT_EQ(get_str(simplify_expression("(x)^(y+1)")), "x^1+y");
  EXPECT_EQ(get_str(simplify_expression("(x+1)^(y)")), "1+x^y");
}

TEST(SimplifyExpressions, SimplifyAddSub) {
  EXPECT_EQ(get_str(simplify_expression("b+c-CeilToInt(b+c,2)+CeilToInt(b+c,2)")), "b+c");
}

TEST(SimplifyExpressions, SimplifyExpression_floordiv_add_ring) {
  // floor(y/n) + floor((y+1)/n) + ... + floor((y+n-1)/n) == y (for integer y).
  EXPECT_EQ(get_str(simplify_expression("(1+b+c)//2+(b+c)//2")), "b+c");
  EXPECT_EQ(get_str(simplify_expression("(b+c+1)//2+(b+c)//2")), "b+c");
  EXPECT_EQ(get_str(simplify_expression("a//2+(a+1)//2")), "a");
  EXPECT_EQ(get_str(simplify_expression("x//3+(x+1)//3+(x+2)//3")), "x");
  EXPECT_EQ(get_str(simplify_expression("(x+5)//3+(x+6)//3+(x+7)//3")), "x+5");
  EXPECT_EQ(get_str(simplify_expression("a//2+(a+1)//2+b")), "a+b");
  EXPECT_EQ(get_str(simplify_expression("CeilToInt(b+c, 2)+(b+c)//2")), "b+c");
  // Negative cases: not enough terms, or offsets do not span all residues.
  EXPECT_EQ(get_str(simplify_expression("x//3+(x+1)//3")), "(1+x)//3+x//3");
  EXPECT_EQ(get_str(simplify_expression("a//2+(a+2)//2")), "(2+a)//2+a//2");
}

TEST(SimplifyExpressions, SimplifyFunction) {
  EXPECT_EQ(get_str(simplify_expression("CeilToInt(b+c,2)")), "(1+b+c)//2");
}

TEST(SimplifyExpressions, SimplifyFunctionOrder) {
  EXPECT_EQ(get_str(simplify_expression("b+a")), "a+b");
}

TEST(SimplifyExpressions, SimplifyFunctionOrder3) {
  EXPECT_EQ(get_str(simplify_expression("c+b+a")), "a+b+c");
  EXPECT_EQ(get_str(simplify_expression("b+c+a")), "a+b+c");
  EXPECT_EQ(get_str(simplify_expression("a+c+b")), "a+b+c");
}

TEST(SimplifyExpressions, SimplifyFunctionFloorDivInt) {
  EXPECT_EQ(get_str(simplify_expression("1024*a//2")), "512*a");
  EXPECT_EQ(get_str(simplify_expression("1024*a//1024")), "a");
  EXPECT_EQ(get_str(simplify_expression("1024*(a+b)//1024")), "a+b");
  EXPECT_EQ(get_str(simplify_expression("1024*(a+b)//1024*2")), "2*a+2*b");
}

TEST(SimplifyExpressions, SimplifyExpressionNegation) {
  EXPECT_EQ(get_str(simplify_expression("-1+1+length")), "length");
  EXPECT_EQ(get_str(simplify_expression("-1+x")), "x-1");
  EXPECT_EQ(get_str(simplify_expression("-x+2*x")), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// test_evaluate_expressions.py (adapted)
// ═══════════════════════════════════════════════════════════════════════════

TEST(EvaluateExpressions, EvaluateExpression) {
  EXPECT_EQ(evaluate_expression("x - y", {{"x", 5}, {"y", 6}}), -1);
  EXPECT_EQ(evaluate_expression("- x", {{"x", 5}}), -5);
}

TEST(EvaluateExpressions, EvaluateExpression_SyntaxError) {
  EXPECT_THROW(evaluate_expression("x +", {}), std::runtime_error);
}

// ═══════════════════════════════════════════════════════════════════════════
// test_rename_expressions.py (adapted)
// ═══════════════════════════════════════════════════════════════════════════

TEST(RenameExpressions, RenameExpression2) {
  EXPECT_EQ(rename_expression("s52+seq_length", {{"s52", "B"}}), "B+seq_length");
}

TEST(RenameExpressions, RenameExpression_Max) {
  std::unordered_map<std::string, std::string> cst{
      {"A", "s77"}, {"s77", "A"}, {"B", "s27"}, {"s27", "B"},           {"D", "s3"},
      {"s3", "D"},  {"E", "s10"}, {"s10", "E"}, {"E^D", "Max(s10,s3)"}, {"Max(s10,s3)", "E^D"},
  };
  EXPECT_EQ(rename_expression("s10^s3", cst), "E^D");
  EXPECT_EQ(rename_expression("Max(s10,s3)", cst), "E^D");
  EXPECT_EQ(rename_dynamic_expression("Max(s10,s3)", cst), "E^D");
}

TEST(RenameExpressions, RenameExpression) {
  EXPECT_EQ(rename_expression("s52+seq_length", {{"s52", "B"}}), "B+seq_length");
}

TEST(RenameExpressions, ParseExpressionTokens_SyntaxError) {
  std::string inv = "a +";
  auto result = parse_expression_tokens(inv);
  EXPECT_EQ(result, std::unordered_set<std::string>{inv});
}

TEST(RenameExpressions, RenameDynamicExpression_SyntaxError) {
  std::string inv = "a +";
  auto result = rename_dynamic_expression(inv, {{"a", "b"}});
  EXPECT_EQ(result, inv);
}

TEST(RenameExpressions, RenameDynamicExpression) {
  std::unordered_map<std::string, std::string> replacements{
      {"DYN0", "DYN0"},        {"batch", "batch"},      {"cache_length", "cache_length"},
      {"s0", "batch"},         {"s1", "seq_length"},    {"s10", "batch"},
      {"s12", "batch"},        {"s14", "batch"},        {"s2", "batch"},
      {"s3", "DYN0"},          {"s8", "batch"},         {"seq_length", "seq_length"},
      {"s11", "cache_length"}, {"s15", "cache_length"}, {"s9", "cache_length"},
      {"s13", "cache_length"},
  };
  EXPECT_EQ(rename_dynamic_expression("s9+seq_length", replacements), "cache_length+seq_length");
}

TEST(RenameExpressions, RenameDynamicExpression_CompoundSubexpression) {
  // A compound subexpression that is a replacement key collapses to its target,
  // even when nested inside a function call such as ``broadcast``.
  std::unordered_map<std::string, std::string> replacements{
      {"past_seq+seq", "total_seq"},
      {"total_seq", "total_seq"},
  };
  EXPECT_EQ(rename_dynamic_expression("past_seq+seq", replacements), "total_seq");
  EXPECT_EQ(rename_dynamic_expression("broadcast(past_seq+seq,total_seq)", replacements),
            "total_seq");
}

TEST(RenameExpressions, RenameDynamicExpression_BroadcastIdenticalCollapses) {
  // ``broadcast(x, x)`` is a no-op and collapses to ``x``.
  EXPECT_EQ(
      rename_dynamic_expression("broadcast(total_seq,total_seq)", {{"total_seq", "total_seq"}}),
      "total_seq");
}

// ═══════════════════════════════════════════════════════════════════════════
// test_operations.py (adapted)
// ═══════════════════════════════════════════════════════════════════════════

TEST(DimOperations, DimMul_int_int) {
  EXPECT_EQ(std::get<int64_t>(dim_mul(DimType{int64_t{3}}, DimType{int64_t{4}})), 12);
}

TEST(DimOperations, DimMul_zero) {
  EXPECT_EQ(std::get<int64_t>(dim_mul(DimType{int64_t{0}}, DimType{int64_t{5}})), 0);
}

TEST(DimOperations, DimMul_symbolic) {
  auto r = dim_mul(DimType{std::string{"n"}}, DimType{int64_t{2}});
  ASSERT_TRUE(std::holds_alternative<std::string>(r));
  const std::string &s = std::get<std::string>(r);
  EXPECT_NE(s.find('n'), std::string::npos);
  EXPECT_NE(s.find('2'), std::string::npos);
}

TEST(DimOperations, DimMul_both_symbolic) {
  auto r = dim_mul(DimType{std::string{"a"}}, DimType{std::string{"b"}});
  ASSERT_TRUE(std::holds_alternative<std::string>(r));
  const std::string &s = std::get<std::string>(r);
  EXPECT_NE(s.find('a'), std::string::npos);
  EXPECT_NE(s.find('b'), std::string::npos);
}

TEST(DimOperations, DimMultiMul_all_int) {
  EXPECT_EQ(std::get<int64_t>(
                dim_multi_mul({DimType{int64_t{2}}, DimType{int64_t{3}}, DimType{int64_t{4}}})),
            24);
}

TEST(DimOperations, DimMultiMul_single_int) {
  EXPECT_EQ(std::get<int64_t>(dim_multi_mul({DimType{int64_t{7}}})), 7);
}

TEST(DimOperations, DimMultiMul_with_symbolic) {
  auto r = dim_multi_mul({DimType{int64_t{2}}, DimType{std::string{"n"}}, DimType{int64_t{3}}});
  ASSERT_TRUE(std::holds_alternative<std::string>(r));
  EXPECT_NE(std::get<std::string>(r).find('n'), std::string::npos);
}

TEST(DimOperations, DimAdd_int_int) {
  EXPECT_EQ(std::get<int64_t>(dim_add(DimType{int64_t{3}}, DimType{int64_t{4}})), 7);
}

TEST(DimOperations, DimAdd_symbolic) {
  auto r = dim_add(DimType{std::string{"n"}}, DimType{int64_t{1}});
  ASSERT_TRUE(std::holds_alternative<std::string>(r));
  const std::string &s = std::get<std::string>(r);
  EXPECT_NE(s.find('n'), std::string::npos);
  EXPECT_NE(s.find('1'), std::string::npos);
}

TEST(DimOperations, DimSub_int_int) {
  EXPECT_EQ(std::get<int64_t>(dim_sub(DimType{int64_t{10}}, DimType{int64_t{3}})), 7);
}

TEST(DimOperations, DimSub_same_symbol) {
  auto r = dim_sub(DimType{std::string{"n"}}, DimType{std::string{"n"}});
  EXPECT_EQ(simplify_result_to_string(SimplifyResult{simplify_result_to_string(r)}), "0");
}

TEST(DimOperations, DimDiv_int_int_exact) {
  EXPECT_EQ(std::get<int64_t>(dim_div(DimType{int64_t{12}}, DimType{int64_t{4}})), 3);
}

TEST(DimOperations, DimDiv_int_int_floor) {
  EXPECT_EQ(std::get<int64_t>(dim_div(DimType{int64_t{7}}, DimType{int64_t{2}})), 3);
}

TEST(DimOperations, DimDiv_symbolic) {
  auto r = dim_div(DimType{std::string{"2*n"}}, DimType{int64_t{2}});
  EXPECT_EQ(dim_to_string(r), "n");
}

TEST(DimOperations, DimMod_int_int) {
  EXPECT_EQ(std::get<int64_t>(dim_mod(DimType{int64_t{10}}, DimType{int64_t{3}})), 1);
}

TEST(DimOperations, DimMod_exact) {
  EXPECT_EQ(std::get<int64_t>(dim_mod(DimType{int64_t{12}}, DimType{int64_t{4}})), 0);
}

TEST(DimOperations, DimMax_int_int) {
  EXPECT_EQ(std::get<int64_t>(dim_max(DimType{int64_t{7}}, DimType{int64_t{3}})), 7);
  EXPECT_EQ(std::get<int64_t>(dim_max(DimType{int64_t{2}}, DimType{int64_t{9}})), 9);
  EXPECT_EQ(std::get<int64_t>(dim_max(DimType{int64_t{5}}, DimType{int64_t{5}})), 5);
}

TEST(DimOperations, DimMax_same_symbol) {
  auto r = dim_max(DimType{std::string{"n"}}, DimType{std::string{"n"}});
  EXPECT_EQ(dim_to_string(r), "n");
}

TEST(DimOperations, DimMin_int_int) {
  EXPECT_EQ(std::get<int64_t>(dim_min(DimType{int64_t{2}}, DimType{int64_t{9}})), 2);
  EXPECT_EQ(std::get<int64_t>(dim_min(DimType{int64_t{8}}, DimType{int64_t{3}})), 3);
  EXPECT_EQ(std::get<int64_t>(dim_min(DimType{int64_t{4}}, DimType{int64_t{4}})), 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional parse / unparse round-trip tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(ParseUnparse, RoundTrip_basics) {
  // parse then unparse should give back a canonical form
  auto node = parse("x+y");
  EXPECT_EQ(unparse(*node), "x+y");

  node = parse("a*b//c");
  EXPECT_EQ(unparse(*node), "a*b//c");

  node = parse("(a+b)*c");
  EXPECT_EQ(unparse(*node), "(a+b)*c");
}

TEST(ParseUnparse, RoundTrip_precedence) {
  // a^(b+c) — ^ lower than +, so no parens needed on right
  auto node = parse("a^b+c");
  EXPECT_EQ(unparse(*node), "a^b+c");

  // (a+b)^c — left of ^ is Add (higher prec), no parens
  node = parse("(a+b)^c");
  EXPECT_EQ(unparse(*node), "a+b^c");
}

TEST(ParseUnparse, SyntaxError) { EXPECT_THROW(parse("x +"), std::runtime_error); }

// ═══════════════════════════════════════════════════════════════════════════
// Exact division (/: ) tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(ExactDiv, ParseUnparse) {
  auto node = parse("a/:b");
  EXPECT_EQ(unparse(*node), "a/:b");
  node = parse("(a*b)/:c");
  EXPECT_EQ(unparse(*node), "a*b/:c");
}

TEST(ExactDiv, Simplify_concrete) {
  // Both concrete: simplifies to integer.
  EXPECT_EQ(get_int(simplify_expression("12/:4")), 3);
  EXPECT_EQ(get_int(simplify_expression("(a*b)/:(a*b)")), 1);
}

TEST(ExactDiv, Simplify_cancel_common_factor) {
  // Exact division allows the simplifier to cancel a factor in the numerator.
  EXPECT_EQ(get_str(simplify_expression("(2*H)/:2")), "H");
  EXPECT_EQ(get_str(simplify_expression("(3*H)/:3")), "H");
  EXPECT_EQ(get_str(simplify_expression("(batch*seq)/:seq")), "batch");
  EXPECT_EQ(get_str(simplify_expression("(1024*a)/:2")), "512*a");
}

TEST(ExactDiv, Simplify_exact_commutes_with_mult) {
  // Unlike floor division (//), exact division (/: ) commutes with
  // multiplication: c*(a/:b) == (c*a)/:b is valid for exact division.
  // The simplifier may therefore cancel the common factor even when it
  // appears outside the division.
  EXPECT_EQ(get_str(simplify_expression("2*(H/:2)")), "H");
}

TEST(ExactDiv, Simplify_floordiv_still_blocked) {
  // Floor division still does NOT commute: 2*(H//2) must remain unchanged.
  EXPECT_EQ(get_str(simplify_expression("2*(H//2)")), "2*(H//2)");
}

TEST(ExactDiv, Evaluate) {
  EXPECT_EQ(evaluate_expression("12/:4", {}), 3);
  EXPECT_EQ(evaluate_expression("(batch*4)/:4", {{"batch", 5}}), 5);
}

TEST(ExactDiv, Evaluate_error_on_remainder) {
  // /: must be exact: evaluating 7/:2 should throw.
  EXPECT_THROW(evaluate_expression("7/:2", {}), std::runtime_error);
}

TEST(DimOperations, DimExactDiv_int_int) {
  EXPECT_EQ(std::get<int64_t>(dim_exact_div(DimType{int64_t{12}}, DimType{int64_t{4}})), 3);
}

TEST(DimOperations, DimExactDiv_int_int_not_exact_throws) {
  EXPECT_THROW(dim_exact_div(DimType{int64_t{7}}, DimType{int64_t{2}}), std::runtime_error);
}

TEST(DimOperations, DimExactDiv_symbolic_simplifies) {
  // dim_exact_div("2*n", 2) should simplify to "n".
  auto r = dim_exact_div(DimType{std::string{"2*n"}}, DimType{int64_t{2}});
  EXPECT_EQ(dim_to_string(r), "n");
}

TEST(DimOperations, DimExactDiv_symbolic_mult_outside) {
  // c*(a/:b) simplifies when c == b.
  auto r = dim_exact_div(DimType{std::string{"batch*4"}}, DimType{int64_t{2}});
  EXPECT_EQ(dim_to_string(r), "2*batch");
}
