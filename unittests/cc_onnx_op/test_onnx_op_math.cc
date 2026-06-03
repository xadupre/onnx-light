// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
// onnx_lib headers define ONNX_LIGHT_NAMESPACE as a macro alias (onnx_light),
// while onnx_op headers in this target use the literal ONNX_LIGHT_NAMESPACE namespace.
// Undefining keeps this test bound to onnx_op symbols while still using onnx_lib APIs explicitly.
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpMathRegistrationTest, ReturnsSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> add_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Add");
  const std::vector<onnx_op::LightOpSchema> mul_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Mul");
  const std::vector<onnx_op::LightOpSchema> div_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Div");
  const std::vector<onnx_op::LightOpSchema> sub_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Sub");
  const std::vector<onnx_op::LightOpSchema> mod_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Mod");
  const std::vector<onnx_op::LightOpSchema> pow_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Pow");
  const std::vector<onnx_op::LightOpSchema> abs_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Abs");
  const std::vector<onnx_op::LightOpSchema> sin_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Sin");
  const std::vector<onnx_op::LightOpSchema> cos_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Cos");
  const std::vector<onnx_op::LightOpSchema> cosh_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Cosh");
  const std::vector<onnx_op::LightOpSchema> sinh_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Sinh");
  const std::vector<onnx_op::LightOpSchema> asin_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Asin");
  const std::vector<onnx_op::LightOpSchema> acos_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Acos");
  const std::vector<onnx_op::LightOpSchema> asinh_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Asinh");
  const std::vector<onnx_op::LightOpSchema> acosh_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Acosh");
  const std::vector<onnx_op::LightOpSchema> atan_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Atan");
  const std::vector<onnx_op::LightOpSchema> atanh_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Atanh");
  const std::vector<onnx_op::LightOpSchema> tan_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Tan");
  const std::vector<onnx_op::LightOpSchema> tanh_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Tanh");
  const std::vector<onnx_op::LightOpSchema> blackman_window_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("BlackmanWindow");
  const std::vector<onnx_op::LightOpSchema> hann_window_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("HannWindow");
  const std::vector<onnx_op::LightOpSchema> hamming_window_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("HammingWindow");
  const std::vector<onnx_op::LightOpSchema> sigmoid_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Sigmoid");
  const std::vector<onnx_op::LightOpSchema> softmax_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Softmax");
  const std::vector<onnx_op::LightOpSchema> sqrt_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Sqrt");
  const std::vector<onnx_op::LightOpSchema> exp_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Exp");
  const std::vector<onnx_op::LightOpSchema> erf_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Erf");
  const std::vector<onnx_op::LightOpSchema> log_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Log");
  const std::vector<onnx_op::LightOpSchema> mat_mul_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("MatMul");
  const std::vector<onnx_op::LightOpSchema> gemm_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Gemm");
  const std::vector<onnx_op::LightOpSchema> sum_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Sum");
  const std::vector<onnx_op::LightOpSchema> einsum_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Einsum");
  const std::vector<onnx_op::LightOpSchema> topk_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("TopK");
  const std::vector<onnx_op::LightOpSchema> swish_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Swish");

  EXPECT_EQ(schemas.size(), 118u);

  // Swish was introduced at v24 and has had a single schema since then.
  ASSERT_EQ(swish_schemas.size(), 1u);
  const onnx_op::LightOpSchema *const swish_v24 = FindByVersion(swish_schemas, 24);
  ASSERT_NE(nullptr, swish_v24);
  EXPECT_EQ(swish_v24->domain(), "ai.onnx");
  EXPECT_EQ(swish_v24->since_version(), 24);
  EXPECT_TRUE(swish_v24->has_function_implementation());
  ASSERT_EQ(swish_v24->inputs().size(), 1u);
  EXPECT_EQ(swish_v24->inputs()[0].name, "X");
  ASSERT_EQ(swish_v24->outputs().size(), 1u);
  EXPECT_EQ(swish_v24->outputs()[0].name, "Y");
  ASSERT_EQ(swish_v24->attributes().size(), 1u);
  EXPECT_EQ(swish_v24->attributes()[0].name, "alpha");
  EXPECT_EQ(swish_v24->attributes()[0].type, onnx_op::AttributeType::FLOAT);
  EXPECT_FALSE(swish_v24->attributes()[0].required);

  // Einsum was introduced at v12 and has had a single schema since then.
  ASSERT_EQ(einsum_schemas.size(), 1u);
  const onnx_op::LightOpSchema *const einsum_v12 = FindByVersion(einsum_schemas, 12);
  ASSERT_NE(nullptr, einsum_v12);
  EXPECT_EQ(einsum_v12->domain(), "ai.onnx");
  EXPECT_EQ(einsum_v12->since_version(), 12);
  EXPECT_FALSE(einsum_v12->has_function_implementation());
  EXPECT_EQ(einsum_v12->inputs().size(), 1u);
  EXPECT_EQ(einsum_v12->inputs()[0].name, "Inputs");
  EXPECT_EQ(einsum_v12->outputs().size(), 1u);
  EXPECT_EQ(einsum_v12->outputs()[0].name, "Output");
  EXPECT_EQ(einsum_v12->type_constraints().size(), 1u);
  EXPECT_EQ(einsum_v12->type_constraints()[0].type_param_str, "T");
  ASSERT_EQ(einsum_v12->attributes().size(), 1u);
  EXPECT_EQ(einsum_v12->attributes()[0].name, "equation");
  EXPECT_EQ(einsum_v12->attributes()[0].type, onnx_op::AttributeType::STRING);
  EXPECT_TRUE(einsum_v12->attributes()[0].required);

  // TopK has three versioned schemas (v1, v10, v11).
  ASSERT_EQ(topk_schemas.size(), 3u);
  const onnx_op::LightOpSchema *const topk_v1 = FindByVersion(topk_schemas, 1);
  const onnx_op::LightOpSchema *const topk_v10 = FindByVersion(topk_schemas, 10);
  const onnx_op::LightOpSchema *const topk_v11 = FindByVersion(topk_schemas, 11);
  ASSERT_NE(nullptr, topk_v1);
  ASSERT_NE(nullptr, topk_v10);
  ASSERT_NE(nullptr, topk_v11);
  EXPECT_EQ(topk_v1->domain(), "ai.onnx");
  EXPECT_EQ(topk_v1->inputs().size(), 1u);
  EXPECT_EQ(topk_v1->inputs()[0].name, "X");
  EXPECT_EQ(topk_v1->outputs().size(), 2u);
  EXPECT_EQ(topk_v1->outputs()[0].name, "Values");
  EXPECT_EQ(topk_v1->outputs()[1].name, "Indices");
  ASSERT_EQ(topk_v1->attributes().size(), 2u);
  EXPECT_EQ(topk_v10->inputs().size(), 2u);
  EXPECT_EQ(topk_v10->inputs()[1].name, "K");
  EXPECT_EQ(topk_v11->inputs().size(), 2u);
  ASSERT_EQ(topk_v11->attributes().size(), 3u);

  // Sum has four versioned schemas (v1, v6, v8, v13).
  EXPECT_EQ(sum_schemas.size(), 4u);
  const onnx_op::LightOpSchema *const sum_v13 = FindByVersion(sum_schemas, 13);
  const onnx_op::LightOpSchema *const sum_v8 = FindByVersion(sum_schemas, 8);
  const onnx_op::LightOpSchema *const sum_v6 = FindByVersion(sum_schemas, 6);
  const onnx_op::LightOpSchema *const sum_v1 = FindByVersion(sum_schemas, 1);
  ASSERT_NE(nullptr, sum_v13);
  ASSERT_NE(nullptr, sum_v8);
  ASSERT_NE(nullptr, sum_v6);
  ASSERT_NE(nullptr, sum_v1);
  EXPECT_EQ(sum_v13->domain(), "ai.onnx");
  EXPECT_EQ(sum_v13->since_version(), 13);
  EXPECT_FALSE(sum_v13->has_function_implementation());
  EXPECT_EQ(sum_v13->inputs().size(), 1u);
  EXPECT_EQ(sum_v13->inputs()[0].name, "data_0");
  EXPECT_EQ(sum_v13->outputs().size(), 1u);
  EXPECT_EQ(sum_v13->outputs()[0].name, "sum");
  EXPECT_EQ(sum_v13->type_constraints().size(), 1u);
  // v13 widens T to include bfloat16; v8 does not.
  EXPECT_NE(sum_v13->type_constraints()[0].allowed_type_strs,
            sum_v8->type_constraints()[0].allowed_type_strs);
  // v8 and v6 share the same float type set (float16/float/double) but
  // their docs differ (v8+ documents broadcasting).
  EXPECT_EQ(sum_v8->type_constraints()[0].allowed_type_strs,
            sum_v6->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(sum_v8->doc(), sum_v6->doc());
  // v1 carries the legacy ``consumed_inputs`` attribute; later opsets drop it.
  EXPECT_EQ(sum_v1->attributes().size(), 1u);
  EXPECT_EQ(sum_v1->attributes()[0].name, "consumed_inputs");
  EXPECT_TRUE(sum_v6->attributes().empty());
  EXPECT_TRUE(sum_v13->attributes().empty());

  const onnx_op::LightOpSchema *const add = FindByVersion(add_schemas, 14);
  const onnx_op::LightOpSchema *const add_v1 = FindByVersion(add_schemas, 1);
  const onnx_op::LightOpSchema *const mul_v13 = FindByVersion(mul_schemas, 13);
  const onnx_op::LightOpSchema *const div_v7 = FindByVersion(div_schemas, 7);
  const onnx_op::LightOpSchema *const sub_v6 = FindByVersion(sub_schemas, 6);
  const onnx_op::LightOpSchema *const mod_v13 = FindByVersion(mod_schemas, 13);
  const onnx_op::LightOpSchema *const mod_v10 = FindByVersion(mod_schemas, 10);
  const onnx_op::LightOpSchema *const pow_v7 = FindByVersion(pow_schemas, 7);
  const onnx_op::LightOpSchema *const pow_v1 = FindByVersion(pow_schemas, 1);
  const onnx_op::LightOpSchema *const abs_v13 = FindByVersion(abs_schemas, 13);
  const onnx_op::LightOpSchema *const abs_v6 = FindByVersion(abs_schemas, 6);
  const onnx_op::LightOpSchema *const abs_v1 = FindByVersion(abs_schemas, 1);
  const onnx_op::LightOpSchema *const sin_v22 = FindByVersion(sin_schemas, 22);
  const onnx_op::LightOpSchema *const sin_v7 = FindByVersion(sin_schemas, 7);
  const onnx_op::LightOpSchema *const cos_v22 = FindByVersion(cos_schemas, 22);
  const onnx_op::LightOpSchema *const cosh_v22 = FindByVersion(cosh_schemas, 22);
  const onnx_op::LightOpSchema *const sinh_v22 = FindByVersion(sinh_schemas, 22);
  const onnx_op::LightOpSchema *const sinh_v9 = FindByVersion(sinh_schemas, 9);
  const onnx_op::LightOpSchema *const asin_v22 = FindByVersion(asin_schemas, 22);
  const onnx_op::LightOpSchema *const asin_v7 = FindByVersion(asin_schemas, 7);
  const onnx_op::LightOpSchema *const acos_v22 = FindByVersion(acos_schemas, 22);
  const onnx_op::LightOpSchema *const acos_v7 = FindByVersion(acos_schemas, 7);
  const onnx_op::LightOpSchema *const asinh_v22 = FindByVersion(asinh_schemas, 22);
  const onnx_op::LightOpSchema *const asinh_v9 = FindByVersion(asinh_schemas, 9);
  const onnx_op::LightOpSchema *const acosh_v22 = FindByVersion(acosh_schemas, 22);
  const onnx_op::LightOpSchema *const acosh_v9 = FindByVersion(acosh_schemas, 9);
  const onnx_op::LightOpSchema *const atan_v22 = FindByVersion(atan_schemas, 22);
  const onnx_op::LightOpSchema *const atan_v7 = FindByVersion(atan_schemas, 7);
  const onnx_op::LightOpSchema *const atanh_v22 = FindByVersion(atanh_schemas, 22);
  const onnx_op::LightOpSchema *const atanh_v9 = FindByVersion(atanh_schemas, 9);
  const onnx_op::LightOpSchema *const tan_v22 = FindByVersion(tan_schemas, 22);
  const onnx_op::LightOpSchema *const tan_v7 = FindByVersion(tan_schemas, 7);
  const onnx_op::LightOpSchema *const tanh_v13 = FindByVersion(tanh_schemas, 13);
  const onnx_op::LightOpSchema *const tanh_v6 = FindByVersion(tanh_schemas, 6);
  const onnx_op::LightOpSchema *const tanh_v1 = FindByVersion(tanh_schemas, 1);
  const onnx_op::LightOpSchema *const blackman_window_v17 =
      FindByVersion(blackman_window_schemas, 17);
  const onnx_op::LightOpSchema *const hann_window_v17 = FindByVersion(hann_window_schemas, 17);
  const onnx_op::LightOpSchema *const hamming_window_v17 =
      FindByVersion(hamming_window_schemas, 17);
  const onnx_op::LightOpSchema *const sigmoid_v13 = FindByVersion(sigmoid_schemas, 13);
  const onnx_op::LightOpSchema *const sigmoid_v6 = FindByVersion(sigmoid_schemas, 6);
  const onnx_op::LightOpSchema *const sigmoid_v1 = FindByVersion(sigmoid_schemas, 1);
  const onnx_op::LightOpSchema *const softmax_v13 = FindByVersion(softmax_schemas, 13);
  const onnx_op::LightOpSchema *const softmax_v11 = FindByVersion(softmax_schemas, 11);
  const onnx_op::LightOpSchema *const softmax_v1 = FindByVersion(softmax_schemas, 1);
  const onnx_op::LightOpSchema *const sqrt_v13 = FindByVersion(sqrt_schemas, 13);
  const onnx_op::LightOpSchema *const sqrt_v6 = FindByVersion(sqrt_schemas, 6);
  const onnx_op::LightOpSchema *const sqrt_v1 = FindByVersion(sqrt_schemas, 1);
  const onnx_op::LightOpSchema *const exp_v13 = FindByVersion(exp_schemas, 13);
  const onnx_op::LightOpSchema *const exp_v6 = FindByVersion(exp_schemas, 6);
  const onnx_op::LightOpSchema *const exp_v1 = FindByVersion(exp_schemas, 1);
  const onnx_op::LightOpSchema *const erf_v13 = FindByVersion(erf_schemas, 13);
  const onnx_op::LightOpSchema *const erf_v9 = FindByVersion(erf_schemas, 9);
  const onnx_op::LightOpSchema *const log_v13 = FindByVersion(log_schemas, 13);
  const onnx_op::LightOpSchema *const log_v6 = FindByVersion(log_schemas, 6);
  const onnx_op::LightOpSchema *const log_v1 = FindByVersion(log_schemas, 1);
  const onnx_op::LightOpSchema *const matmul_v13 = FindByVersion(mat_mul_schemas, 13);
  const onnx_op::LightOpSchema *const matmul_v9 = FindByVersion(mat_mul_schemas, 9);
  const onnx_op::LightOpSchema *const matmul_v1 = FindByVersion(mat_mul_schemas, 1);
  const onnx_op::LightOpSchema *const gemm_v13 = FindByVersion(gemm_schemas, 13);
  const onnx_op::LightOpSchema *const gemm_v11 = FindByVersion(gemm_schemas, 11);
  const onnx_op::LightOpSchema *const gemm_v9 = FindByVersion(gemm_schemas, 9);
  const onnx_op::LightOpSchema *const gemm_v7 = FindByVersion(gemm_schemas, 7);
  const onnx_op::LightOpSchema *const gemm_v6 = FindByVersion(gemm_schemas, 6);
  const onnx_op::LightOpSchema *const gemm_v1 = FindByVersion(gemm_schemas, 1);
  ASSERT_NE(nullptr, add);
  ASSERT_NE(nullptr, add_v1);
  ASSERT_NE(nullptr, mul_v13);
  ASSERT_NE(nullptr, div_v7);
  ASSERT_NE(nullptr, sub_v6);
  ASSERT_NE(nullptr, mod_v13);
  ASSERT_NE(nullptr, mod_v10);
  ASSERT_NE(nullptr, pow_v7);
  ASSERT_NE(nullptr, pow_v1);
  ASSERT_NE(nullptr, abs_v13);
  ASSERT_NE(nullptr, abs_v6);
  ASSERT_NE(nullptr, abs_v1);
  ASSERT_NE(nullptr, sin_v22);
  ASSERT_NE(nullptr, sin_v7);
  ASSERT_NE(nullptr, cos_v22);
  ASSERT_NE(nullptr, cosh_v22);
  ASSERT_NE(nullptr, sinh_v22);
  ASSERT_NE(nullptr, sinh_v9);
  ASSERT_NE(nullptr, asin_v22);
  ASSERT_NE(nullptr, asin_v7);
  ASSERT_NE(nullptr, acos_v22);
  ASSERT_NE(nullptr, acos_v7);
  ASSERT_NE(nullptr, asinh_v22);
  ASSERT_NE(nullptr, asinh_v9);
  ASSERT_NE(nullptr, acosh_v22);
  ASSERT_NE(nullptr, acosh_v9);
  ASSERT_NE(nullptr, atan_v22);
  ASSERT_NE(nullptr, atan_v7);
  ASSERT_NE(nullptr, atanh_v22);
  ASSERT_NE(nullptr, atanh_v9);
  ASSERT_NE(nullptr, tan_v22);
  ASSERT_NE(nullptr, tan_v7);
  ASSERT_NE(nullptr, tanh_v13);
  ASSERT_NE(nullptr, tanh_v6);
  ASSERT_NE(nullptr, tanh_v1);
  ASSERT_NE(nullptr, blackman_window_v17);
  ASSERT_NE(nullptr, sigmoid_v13);
  ASSERT_NE(nullptr, sigmoid_v6);
  ASSERT_NE(nullptr, sigmoid_v1);
  ASSERT_NE(nullptr, softmax_v13);
  ASSERT_NE(nullptr, softmax_v11);
  ASSERT_NE(nullptr, softmax_v1);
  ASSERT_NE(nullptr, sqrt_v13);
  ASSERT_NE(nullptr, sqrt_v6);
  ASSERT_NE(nullptr, sqrt_v1);
  ASSERT_NE(nullptr, exp_v13);
  ASSERT_NE(nullptr, exp_v6);
  ASSERT_NE(nullptr, exp_v1);
  ASSERT_NE(nullptr, erf_v13);
  ASSERT_NE(nullptr, erf_v9);
  ASSERT_NE(nullptr, log_v13);
  ASSERT_NE(nullptr, log_v6);
  ASSERT_NE(nullptr, log_v1);
  ASSERT_NE(nullptr, matmul_v13);
  ASSERT_NE(nullptr, matmul_v9);
  ASSERT_NE(nullptr, matmul_v1);
  ASSERT_NE(nullptr, gemm_v13);
  ASSERT_NE(nullptr, gemm_v11);
  ASSERT_NE(nullptr, gemm_v9);
  ASSERT_NE(nullptr, gemm_v7);
  ASSERT_NE(nullptr, gemm_v6);
  ASSERT_NE(nullptr, gemm_v1);
  EXPECT_EQ(add->domain(), "ai.onnx");
  EXPECT_EQ(add->since_version(), 14);
  EXPECT_FALSE(add->has_function_implementation());
  EXPECT_EQ(add->inputs().size(), 2u);
  EXPECT_EQ(add->outputs().size(), 1u);
  EXPECT_EQ(add->type_constraints().size(), 1u);
  EXPECT_NE(add_v1->inputs()[0].description, add->inputs()[0].description);
  EXPECT_NE(add_v1->type_constraints()[0].allowed_type_strs,
            add->type_constraints()[0].allowed_type_strs);
  const std::vector<onnx_op::TensorType> expected_v22_float_types = {
      onnx_op::TensorType::kBfloat16, onnx_op::TensorType::kFloat16, onnx_op::TensorType::kFloat,
      onnx_op::TensorType::kDouble};
  const std::vector<onnx_op::TensorType> expected_log_v13_float_types = {
      onnx_op::TensorType::kFloat16, onnx_op::TensorType::kFloat, onnx_op::TensorType::kDouble,
      onnx_op::TensorType::kBfloat16};
  EXPECT_EQ(mod_v13->inputs()[0].description, "Dividend tensor");
  EXPECT_EQ(mod_v13->outputs()[0].description, "Remainder tensor");
  EXPECT_NE(mod_v10->type_constraints()[0].allowed_type_strs,
            mod_v13->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(pow_v7->inputs()[0].description, "First operand, base of the exponent.");
  EXPECT_EQ(pow_v1->outputs()[0].description, "Output tensor (same size as X)");
  EXPECT_NE(abs_v1->type_constraints()[0].allowed_type_strs,
            abs_v6->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(abs_v6->type_constraints()[0].allowed_type_strs,
            abs_v13->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(abs_v13->inputs()[0].name, "X");
  EXPECT_EQ(abs_v13->outputs()[0].name, "Y");
  EXPECT_EQ(sin_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_EQ(cos_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_EQ(cosh_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_EQ(sinh_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_NE(sin_v7->type_constraints()[0].allowed_type_strs,
            sin_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(sinh_v9->type_constraints()[0].allowed_type_strs,
            sinh_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(cosh_v22->outputs()[0].description,
            "The hyperbolic cosine values of the input tensor computed element-wise");
  EXPECT_EQ(sinh_v9->outputs()[0].description,
            "The hyperbolic sine values of the input tensor computed element-wise");
  EXPECT_EQ(asin_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_EQ(acos_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_EQ(asinh_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_EQ(acosh_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_NE(asin_v7->type_constraints()[0].allowed_type_strs,
            asin_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(acos_v7->type_constraints()[0].allowed_type_strs,
            acos_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(asinh_v9->type_constraints()[0].allowed_type_strs,
            asinh_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(acosh_v9->type_constraints()[0].allowed_type_strs,
            acosh_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(asin_v22->outputs()[0].description,
            "The arcsine of the input tensor computed element-wise");
  EXPECT_EQ(acos_v22->outputs()[0].description,
            "The arccosine of the input tensor computed element-wise");
  EXPECT_EQ(asinh_v9->outputs()[0].description,
            "The hyperbolic arcsine values of the input tensor computed element-wise");
  EXPECT_EQ(acosh_v9->outputs()[0].description,
            "The hyperbolic arccosine values of the input tensor computed element-wise");
  EXPECT_EQ(atan_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_EQ(atanh_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_NE(atan_v7->type_constraints()[0].allowed_type_strs,
            atan_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(atanh_v9->type_constraints()[0].allowed_type_strs,
            atanh_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(atan_v22->outputs()[0].description,
            "The arctangent of the input tensor computed element-wise");
  EXPECT_EQ(atanh_v9->outputs()[0].description,
            "The hyperbolic arctangent values of the input tensor computed element-wise");
  EXPECT_EQ(tan_v22->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_NE(tan_v7->type_constraints()[0].allowed_type_strs,
            tan_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(tan_v22->outputs()[0].description,
            "The tangent of the input tensor computed element-wise");
  EXPECT_EQ(exp_v13->type_constraints()[0].allowed_type_strs, expected_v22_float_types);
  EXPECT_NE(exp_v6->type_constraints()[0].allowed_type_strs,
            exp_v13->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(exp_v1->outputs()[0].description,
            "The exponential of the input tensor computed element-wise");
  EXPECT_EQ(log_v13->type_constraints()[0].allowed_type_strs, expected_log_v13_float_types);
  EXPECT_NE(log_v6->type_constraints()[0].allowed_type_strs,
            log_v13->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(log_v1->outputs()[0].description,
            "The natural log of the input tensor computed element-wise");
  EXPECT_EQ(tanh_v13->inputs()[0].name, "input");
  EXPECT_EQ(tanh_v13->outputs()[0].name, "output");
  EXPECT_NE(tanh_v6->type_constraints()[0].allowed_type_strs,
            tanh_v13->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(tanh_v1->type_constraints()[0].allowed_type_strs,
            tanh_v6->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(tanh_v13->outputs()[0].description,
            "The hyperbolic tangent values of the input tensor computed element-wise");
  EXPECT_EQ(tanh_v1->inputs()[0].description, "1-D input tensor");
  EXPECT_EQ(sigmoid_v13->inputs()[0].name, "X");
  EXPECT_EQ(sigmoid_v13->outputs()[0].name, "Y");
  EXPECT_NE(sigmoid_v6->type_constraints()[0].allowed_type_strs,
            sigmoid_v13->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(sigmoid_v1->type_constraints()[0].allowed_type_strs,
            sigmoid_v6->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(sqrt_v13->inputs()[0].name, "X");
  EXPECT_EQ(sqrt_v13->outputs()[0].name, "Y");
  EXPECT_EQ(sqrt_v13->type_constraints()[0].allowed_type_strs, expected_log_v13_float_types);
  EXPECT_NE(sqrt_v6->type_constraints()[0].allowed_type_strs,
            sqrt_v13->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(sqrt_v1->type_constraints()[0].allowed_type_strs,
            sqrt_v6->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(sqrt_v1->inputs()[0].description, "Input tensor");
  EXPECT_EQ(sqrt_v1->outputs()[0].description, "Output tensor");
  EXPECT_EQ(softmax_v13->inputs()[0].name, "input");
  EXPECT_EQ(softmax_v13->outputs()[0].name, "output");
  ASSERT_EQ(softmax_v13->attributes().size(), 1u);
  ASSERT_EQ(softmax_v11->attributes().size(), 1u);
  ASSERT_EQ(softmax_v1->attributes().size(), 1u);
  EXPECT_EQ(softmax_v13->attributes()[0].name, "axis");
  EXPECT_EQ(softmax_v13->attributes()[0].default_value, onnx_op::AttributeDefault(int64_t{-1}));
  EXPECT_EQ(softmax_v11->attributes()[0].default_value, onnx_op::AttributeDefault(int64_t{1}));
  EXPECT_EQ(softmax_v1->attributes()[0].default_value, onnx_op::AttributeDefault(int64_t{1}));
  EXPECT_EQ(blackman_window_v17->inputs().size(), 1u);
  EXPECT_EQ(blackman_window_v17->inputs()[0].name, "size");
  EXPECT_EQ(blackman_window_v17->inputs()[0].type, "T1");
  EXPECT_EQ(blackman_window_v17->outputs().size(), 1u);
  EXPECT_EQ(blackman_window_v17->outputs()[0].name, "output");
  EXPECT_EQ(blackman_window_v17->outputs()[0].type, "T2");
  ASSERT_EQ(blackman_window_v17->type_constraints().size(), 2u);
  EXPECT_EQ(
      blackman_window_v17->type_constraints()[0].allowed_type_strs,
      (std::vector<onnx_op::TensorType>{onnx_op::TensorType::kInt32, onnx_op::TensorType::kInt64}));
  EXPECT_EQ(blackman_window_v17->type_constraints()[1].allowed_type_strs,
            onnx_op::AllNumericTypesIr4());

  ASSERT_NE(hann_window_v17, nullptr);
  EXPECT_EQ(hann_window_v17->inputs().size(), 1u);
  EXPECT_EQ(hann_window_v17->inputs()[0].name, "size");
  EXPECT_EQ(hann_window_v17->inputs()[0].type, "T1");
  EXPECT_EQ(hann_window_v17->outputs().size(), 1u);
  EXPECT_EQ(hann_window_v17->outputs()[0].name, "output");
  EXPECT_EQ(hann_window_v17->outputs()[0].type, "T2");
  ASSERT_EQ(hann_window_v17->type_constraints().size(), 2u);
  EXPECT_EQ(
      hann_window_v17->type_constraints()[0].allowed_type_strs,
      (std::vector<onnx_op::TensorType>{onnx_op::TensorType::kInt32, onnx_op::TensorType::kInt64}));
  EXPECT_EQ(hann_window_v17->type_constraints()[1].allowed_type_strs,
            onnx_op::AllNumericTypesIr4());

  ASSERT_NE(hamming_window_v17, nullptr);
  EXPECT_EQ(hamming_window_v17->inputs().size(), 1u);
  EXPECT_EQ(hamming_window_v17->inputs()[0].name, "size");
  EXPECT_EQ(hamming_window_v17->inputs()[0].type, "T1");
  EXPECT_EQ(hamming_window_v17->outputs().size(), 1u);
  EXPECT_EQ(hamming_window_v17->outputs()[0].name, "output");
  EXPECT_EQ(hamming_window_v17->outputs()[0].type, "T2");
  ASSERT_EQ(hamming_window_v17->type_constraints().size(), 2u);
  EXPECT_EQ(
      hamming_window_v17->type_constraints()[0].allowed_type_strs,
      (std::vector<onnx_op::TensorType>{onnx_op::TensorType::kInt32, onnx_op::TensorType::kInt64}));
  EXPECT_EQ(hamming_window_v17->type_constraints()[1].allowed_type_strs,
            onnx_op::AllNumericTypesIr4());

  // MatMul
  EXPECT_EQ(matmul_v1->inputs().size(), 2u);
  EXPECT_EQ(matmul_v1->inputs()[0].name, "A");
  EXPECT_EQ(matmul_v1->inputs()[1].name, "B");
  EXPECT_EQ(matmul_v1->outputs().size(), 1u);
  EXPECT_EQ(matmul_v1->outputs()[0].name, "Y");
  EXPECT_EQ(matmul_v1->type_constraints()[0].allowed_type_strs, onnx_op::FloatTypes());
  EXPECT_NE(matmul_v9->type_constraints()[0].allowed_type_strs,
            matmul_v1->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(matmul_v13->type_constraints()[0].allowed_type_strs,
            matmul_v9->type_constraints()[0].allowed_type_strs);

  // Gemm
  EXPECT_EQ(gemm_v1->inputs().size(), 3u);
  EXPECT_EQ(gemm_v1->inputs()[2].description, "Input tensor C, can be inplace.");
  EXPECT_EQ(gemm_v6->inputs()[2].description, "Input tensor C");
  EXPECT_EQ(gemm_v7->inputs()[2].description,
            "Input tensor C. The shape of C should be unidirectional broadcastable to (M, N).");
  EXPECT_EQ(gemm_v11->inputs()[2].description,
            "Optional input tensor C. If not specified, the computation is done as if C is a "
            "scalar 0. The shape of C should be unidirectional broadcastable to (M, N).");
  EXPECT_EQ(gemm_v1->type_constraints()[0].allowed_type_strs, onnx_op::FloatTypes());
  EXPECT_EQ(gemm_v6->type_constraints()[0].allowed_type_strs, onnx_op::FloatTypes());
  EXPECT_EQ(gemm_v7->type_constraints()[0].allowed_type_strs, onnx_op::FloatTypes());
  EXPECT_NE(gemm_v9->type_constraints()[0].allowed_type_strs,
            gemm_v7->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(gemm_v13->type_constraints()[0].allowed_type_strs,
            gemm_v11->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(gemm_v11->outputs()[0].name, "Y");
  EXPECT_EQ(gemm_v11->outputs()[0].description, "Output tensor of shape (M, N).");
}

TEST(OnnxOpMathRegistrationTest, DetHistoryHasExpectedVersions) {
  const std::vector<onnx_op::LightOpSchema> det_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Det");
  ASSERT_EQ(det_schemas.size(), 2u);
  const onnx_op::LightOpSchema *const det_v22 = FindByVersion(det_schemas, 22);
  const onnx_op::LightOpSchema *const det_v11 = FindByVersion(det_schemas, 11);
  ASSERT_NE(nullptr, det_v22);
  ASSERT_NE(nullptr, det_v11);
  EXPECT_EQ(det_v22->domain(), "ai.onnx");
  EXPECT_EQ(det_v22->name(), "Det");
  EXPECT_EQ(det_v22->since_version(), 22);
  EXPECT_FALSE(det_v22->has_function_implementation());
  ASSERT_EQ(det_v22->inputs().size(), 1u);
  EXPECT_EQ(det_v22->inputs()[0].name, "X");
  ASSERT_EQ(det_v22->outputs().size(), 1u);
  EXPECT_EQ(det_v22->outputs()[0].name, "Y");
  ASSERT_EQ(det_v22->type_constraints().size(), 1u);
  EXPECT_EQ(det_v22->type_constraints()[0].type_param_str, "T");
  // v22 widens T to include bfloat16; v11 does not.
  EXPECT_EQ(det_v11->type_constraints()[0].allowed_type_strs, onnx_op::FloatTypes());
  EXPECT_NE(det_v22->type_constraints()[0].allowed_type_strs,
            det_v11->type_constraints()[0].allowed_type_strs);
  EXPECT_TRUE(det_v22->attributes().empty());
  EXPECT_TRUE(det_v11->attributes().empty());
}

TEST(OnnxOpMathRegistrationTest, OpTypeFilterReturnsOnlyMatchingSchemas) {
  const std::vector<onnx_op::LightOpSchema> abs_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Abs");
  EXPECT_EQ(abs_schemas.size(), 3u);
  for (const auto &schema : abs_schemas) {
    EXPECT_EQ(schema.name(), "Abs");
  }

  const std::vector<onnx_op::LightOpSchema> empty_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("DoesNotExist");
  EXPECT_TRUE(empty_schemas.empty());

  const std::vector<onnx_op::LightOpSchema> all_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> default_filter =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("");
  EXPECT_EQ(default_filter.size(), all_schemas.size());
}

TEST(OnnxOpMathRegistrationTest, FloorCeilRoundHistoryHasExpectedVersions) {
  const std::vector<onnx_op::LightOpSchema> floor_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Floor");
  const std::vector<onnx_op::LightOpSchema> ceil_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Ceil");
  const std::vector<onnx_op::LightOpSchema> round_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Round");

  ASSERT_EQ(floor_schemas.size(), 3u);
  ASSERT_EQ(ceil_schemas.size(), 3u);
  ASSERT_EQ(round_schemas.size(), 2u);

  const onnx_op::LightOpSchema *const floor_v13 = FindByVersion(floor_schemas, 13);
  const onnx_op::LightOpSchema *const floor_v6 = FindByVersion(floor_schemas, 6);
  const onnx_op::LightOpSchema *const floor_v1 = FindByVersion(floor_schemas, 1);
  const onnx_op::LightOpSchema *const ceil_v13 = FindByVersion(ceil_schemas, 13);
  const onnx_op::LightOpSchema *const ceil_v6 = FindByVersion(ceil_schemas, 6);
  const onnx_op::LightOpSchema *const ceil_v1 = FindByVersion(ceil_schemas, 1);
  const onnx_op::LightOpSchema *const round_v22 = FindByVersion(round_schemas, 22);
  const onnx_op::LightOpSchema *const round_v11 = FindByVersion(round_schemas, 11);

  ASSERT_NE(nullptr, floor_v13);
  ASSERT_NE(nullptr, floor_v6);
  ASSERT_NE(nullptr, floor_v1);
  ASSERT_NE(nullptr, ceil_v13);
  ASSERT_NE(nullptr, ceil_v6);
  ASSERT_NE(nullptr, ceil_v1);
  ASSERT_NE(nullptr, round_v22);
  ASSERT_NE(nullptr, round_v11);

  // v13 widens dtypes to include bfloat16; earlier versions don't.
  const std::vector<onnx_op::TensorType> expected_v13_float_types{
      onnx_op::TensorType::kFloat16, onnx_op::TensorType::kFloat, onnx_op::TensorType::kDouble,
      onnx_op::TensorType::kBfloat16};
  const std::vector<onnx_op::TensorType> expected_float_types{
      onnx_op::TensorType::kFloat16, onnx_op::TensorType::kFloat, onnx_op::TensorType::kDouble};
  EXPECT_EQ(floor_v13->type_constraints()[0].allowed_type_strs, expected_v13_float_types);
  EXPECT_EQ(floor_v6->type_constraints()[0].allowed_type_strs, expected_float_types);
  EXPECT_EQ(floor_v1->type_constraints()[0].allowed_type_strs, expected_float_types);
  EXPECT_EQ(ceil_v13->type_constraints()[0].allowed_type_strs, expected_v13_float_types);
  EXPECT_EQ(ceil_v6->type_constraints()[0].allowed_type_strs, expected_float_types);
  EXPECT_EQ(ceil_v1->type_constraints()[0].allowed_type_strs, expected_float_types);

  // Round v22 uses bfloat16-first ordering (all_float_types_ir4); v11 has the
  // narrower float16/float/double set.
  const std::vector<onnx_op::TensorType> expected_round_v22_float_types{
      onnx_op::TensorType::kBfloat16, onnx_op::TensorType::kFloat16, onnx_op::TensorType::kFloat,
      onnx_op::TensorType::kDouble};
  EXPECT_EQ(round_v22->type_constraints()[0].allowed_type_strs, expected_round_v22_float_types);
  EXPECT_EQ(round_v11->type_constraints()[0].allowed_type_strs, expected_float_types);

  // All three operators use X/Y input/output names (matching ONNX).
  EXPECT_EQ(floor_v13->inputs()[0].name, "X");
  EXPECT_EQ(floor_v13->outputs()[0].name, "Y");
  EXPECT_EQ(ceil_v13->inputs()[0].name, "X");
  EXPECT_EQ(ceil_v13->outputs()[0].name, "Y");
  EXPECT_EQ(round_v22->inputs()[0].name, "X");
  EXPECT_EQ(round_v22->outputs()[0].name, "Y");
}

TEST(OnnxOpMathRegistrationTest, ReturnsDFTSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> dft_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("DFT");

  const onnx_op::LightOpSchema *const dft_v17 = FindByVersion(dft_schemas, 17);
  const onnx_op::LightOpSchema *const dft_v20 = FindByVersion(dft_schemas, 20);
  ASSERT_NE(nullptr, dft_v17);
  ASSERT_NE(nullptr, dft_v20);

  // --- v17: axis is an INT attribute; inputs are (input, dft_length?).
  EXPECT_EQ(dft_v17->name(), "DFT");
  EXPECT_EQ(dft_v17->domain(), onnx_op::kOnnxDomain);
  EXPECT_EQ(dft_v17->since_version(), 17);
  ASSERT_EQ(dft_v17->inputs().size(), 2u);
  EXPECT_EQ(dft_v17->inputs()[0].name, "input");
  EXPECT_EQ(dft_v17->inputs()[0].type, "T1");
  EXPECT_EQ(dft_v17->inputs()[1].name, "dft_length");
  EXPECT_EQ(dft_v17->inputs()[1].type, "T2");
  ASSERT_EQ(dft_v17->outputs().size(), 1u);
  EXPECT_EQ(dft_v17->outputs()[0].name, "output");
  EXPECT_EQ(dft_v17->outputs()[0].type, "T1");
  ASSERT_EQ(dft_v17->attributes().size(), 3u);
  // axis / inverse / onesided.
  bool has_axis = false, has_inverse = false, has_onesided = false;
  for (const auto &attr : dft_v17->attributes()) {
    if (attr.name == "axis") {
      has_axis = true;
      EXPECT_EQ(attr.type, onnx_op::AttributeType::INT);
    } else if (attr.name == "inverse") {
      has_inverse = true;
      EXPECT_EQ(attr.type, onnx_op::AttributeType::INT);
    } else if (attr.name == "onesided") {
      has_onesided = true;
      EXPECT_EQ(attr.type, onnx_op::AttributeType::INT);
    }
  }
  EXPECT_TRUE(has_axis);
  EXPECT_TRUE(has_inverse);
  EXPECT_TRUE(has_onesided);

  // --- v20: axis is the third input; only inverse/onesided remain as attrs.
  EXPECT_EQ(dft_v20->since_version(), 20);
  ASSERT_EQ(dft_v20->inputs().size(), 3u);
  EXPECT_EQ(dft_v20->inputs()[0].name, "input");
  EXPECT_EQ(dft_v20->inputs()[0].type, "T1");
  EXPECT_EQ(dft_v20->inputs()[1].name, "dft_length");
  EXPECT_EQ(dft_v20->inputs()[1].type, "T2");
  EXPECT_EQ(dft_v20->inputs()[2].name, "axis");
  EXPECT_EQ(dft_v20->inputs()[2].type, "tensor(int64)");
  ASSERT_EQ(dft_v20->outputs().size(), 1u);
  EXPECT_EQ(dft_v20->outputs()[0].name, "output");
  ASSERT_EQ(dft_v20->attributes().size(), 2u);
  for (const auto &attr : dft_v20->attributes()) {
    EXPECT_TRUE(attr.name == "inverse" || attr.name == "onesided");
    EXPECT_EQ(attr.type, onnx_op::AttributeType::INT);
  }

  EXPECT_FALSE(dft_v17->doc().empty());
  EXPECT_FALSE(dft_v20->doc().empty());
}

} // namespace Test
