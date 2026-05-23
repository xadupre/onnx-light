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

const onnx_op::LightOpSchema *FindSchema(const std::vector<onnx_op::LightOpSchema> &schemas,
                                         const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpMathRegistrationTest, ReturnsSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 53u);

  const onnx_op::LightOpSchema *const add = FindSchema(schemas, "Add", 14);
  const onnx_op::LightOpSchema *const add_v1 = FindSchema(schemas, "Add", 1);
  const onnx_op::LightOpSchema *const mul_v13 = FindSchema(schemas, "Mul", 13);
  const onnx_op::LightOpSchema *const div_v7 = FindSchema(schemas, "Div", 7);
  const onnx_op::LightOpSchema *const sub_v6 = FindSchema(schemas, "Sub", 6);
  const onnx_op::LightOpSchema *const mod_v13 = FindSchema(schemas, "Mod", 13);
  const onnx_op::LightOpSchema *const mod_v10 = FindSchema(schemas, "Mod", 10);
  const onnx_op::LightOpSchema *const pow_v7 = FindSchema(schemas, "Pow", 7);
  const onnx_op::LightOpSchema *const pow_v1 = FindSchema(schemas, "Pow", 1);
  const onnx_op::LightOpSchema *const abs_v13 = FindSchema(schemas, "Abs", 13);
  const onnx_op::LightOpSchema *const abs_v6 = FindSchema(schemas, "Abs", 6);
  const onnx_op::LightOpSchema *const abs_v1 = FindSchema(schemas, "Abs", 1);
  const onnx_op::LightOpSchema *const sin_v22 = FindSchema(schemas, "Sin", 22);
  const onnx_op::LightOpSchema *const sin_v7 = FindSchema(schemas, "Sin", 7);
  const onnx_op::LightOpSchema *const cos_v22 = FindSchema(schemas, "Cos", 22);
  const onnx_op::LightOpSchema *const cosh_v22 = FindSchema(schemas, "Cosh", 22);
  const onnx_op::LightOpSchema *const sinh_v22 = FindSchema(schemas, "Sinh", 22);
  const onnx_op::LightOpSchema *const sinh_v9 = FindSchema(schemas, "Sinh", 9);
  const onnx_op::LightOpSchema *const asin_v22 = FindSchema(schemas, "Asin", 22);
  const onnx_op::LightOpSchema *const asin_v7 = FindSchema(schemas, "Asin", 7);
  const onnx_op::LightOpSchema *const acos_v22 = FindSchema(schemas, "Acos", 22);
  const onnx_op::LightOpSchema *const acos_v7 = FindSchema(schemas, "Acos", 7);
  const onnx_op::LightOpSchema *const asinh_v22 = FindSchema(schemas, "Asinh", 22);
  const onnx_op::LightOpSchema *const asinh_v9 = FindSchema(schemas, "Asinh", 9);
  const onnx_op::LightOpSchema *const acosh_v22 = FindSchema(schemas, "Acosh", 22);
  const onnx_op::LightOpSchema *const acosh_v9 = FindSchema(schemas, "Acosh", 9);
  const onnx_op::LightOpSchema *const blackman_window_v17 =
      FindSchema(schemas, "BlackmanWindow", 17);
  const onnx_op::LightOpSchema *const matmul_v13 = FindSchema(schemas, "MatMul", 13);
  const onnx_op::LightOpSchema *const matmul_v9 = FindSchema(schemas, "MatMul", 9);
  const onnx_op::LightOpSchema *const matmul_v1 = FindSchema(schemas, "MatMul", 1);
  const onnx_op::LightOpSchema *const gemm_v13 = FindSchema(schemas, "Gemm", 13);
  const onnx_op::LightOpSchema *const gemm_v11 = FindSchema(schemas, "Gemm", 11);
  const onnx_op::LightOpSchema *const gemm_v9 = FindSchema(schemas, "Gemm", 9);
  const onnx_op::LightOpSchema *const gemm_v7 = FindSchema(schemas, "Gemm", 7);
  const onnx_op::LightOpSchema *const gemm_v6 = FindSchema(schemas, "Gemm", 6);
  const onnx_op::LightOpSchema *const gemm_v1 = FindSchema(schemas, "Gemm", 1);
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
  ASSERT_NE(nullptr, blackman_window_v17);
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

} // namespace Test
