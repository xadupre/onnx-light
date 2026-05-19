// SPDX-License-Identifier: Apache-2.0
//
// C++ unit tests for onnx_light::expressions.
// These test cases are translated from
// https://github.com/xadupre/yet-another-onnx-builder/tree/main/unittests/xexpressions

#include "cc_onnx_expressions/expressions.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

using namespace onnx_light::expressions;

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

TEST(SimplifyExpressions, SimplifyExpression_bracket) {
  EXPECT_EQ(get_str(simplify_expression("2*x//2")), "x");
  EXPECT_EQ(get_str(simplify_expression("(2*x)//2")), "x");
  EXPECT_EQ(get_str(simplify_expression("(x*y)//y")), "x");
  EXPECT_EQ(get_str(simplify_expression("(x*(y+1))//(y+1)")), "x");
  EXPECT_EQ(get_str(simplify_expression("((c)//(2))")), "c//2");
}

TEST(SimplifyExpressions, SimplifyExpression_bracket_max) {
  EXPECT_EQ(get_str(simplify_expression("(x)^(y+1)")), "x^1+y");
  EXPECT_EQ(get_str(simplify_expression("(x+1)^(y)")), "1+x^y");
}

TEST(SimplifyExpressions, SimplifyAddSub) {
  EXPECT_EQ(get_str(simplify_expression("b+c-CeilToInt(b+c,2)+CeilToInt(b+c,2)")), "b+c");
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
