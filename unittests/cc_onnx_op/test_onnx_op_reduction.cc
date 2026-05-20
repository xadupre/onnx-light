// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_reduction.h"

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

const onnx_op::LightOpSchema *
FindReductionSchema(const std::vector<onnx_op::LightOpSchema> &schemas, const std::string &op_type,
                    int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpReductionRegistrationTest, ReturnsSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::reduction::GetAllOnnxOpReductionSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 3u);

  const onnx_op::LightOpSchema *const reduce_sum_v13 =
      FindReductionSchema(schemas, "ReduceSum", 13);
  const onnx_op::LightOpSchema *const reduce_sum_v11 =
      FindReductionSchema(schemas, "ReduceSum", 11);
  const onnx_op::LightOpSchema *const reduce_sum_v1 = FindReductionSchema(schemas, "ReduceSum", 1);

  ASSERT_NE(nullptr, reduce_sum_v13);
  ASSERT_NE(nullptr, reduce_sum_v11);
  ASSERT_NE(nullptr, reduce_sum_v1);

  // Check v13 schema: two inputs (data + optional axes), one output.
  EXPECT_EQ(reduce_sum_v13->domain(), "ai.onnx");
  EXPECT_EQ(reduce_sum_v13->since_version(), 13);
  EXPECT_EQ(reduce_sum_v13->inputs().size(), 2u);
  EXPECT_EQ(reduce_sum_v13->inputs()[0].name, "data");
  EXPECT_EQ(reduce_sum_v13->inputs()[1].name, "axes");
  EXPECT_EQ(reduce_sum_v13->outputs().size(), 1u);
  EXPECT_EQ(reduce_sum_v13->outputs()[0].name, "reduced");
  EXPECT_EQ(reduce_sum_v13->type_constraints().size(), 1u);
  EXPECT_EQ(reduce_sum_v13->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(reduce_sum_v13->type_constraints()[0].allowed_type_strs,
            onnx_op::NumericTypesForMathReductionIr4());
  EXPECT_FALSE(reduce_sum_v13->has_function_implementation());

  // Check v11 schema: one input (axes remain an attribute), one output.
  EXPECT_EQ(reduce_sum_v11->since_version(), 11);
  EXPECT_EQ(reduce_sum_v11->inputs().size(), 1u);
  EXPECT_EQ(reduce_sum_v11->inputs()[0].name, "data");
  EXPECT_EQ(reduce_sum_v11->outputs().size(), 1u);
  EXPECT_EQ(reduce_sum_v11->type_constraints().size(), 1u);
  EXPECT_EQ(reduce_sum_v11->type_constraints()[0].allowed_type_strs,
            onnx_op::NumericTypesForMathReduction());

  // Check v1 schema: one input, one output.
  EXPECT_EQ(reduce_sum_v1->since_version(), 1);
  EXPECT_EQ(reduce_sum_v1->inputs().size(), 1u);
  EXPECT_EQ(reduce_sum_v1->outputs().size(), 1u);
  EXPECT_EQ(reduce_sum_v1->type_constraints().size(), 1u);
  EXPECT_EQ(reduce_sum_v1->type_constraints()[0].allowed_type_strs,
            onnx_op::NumericTypesForMathReduction());

  // v13 has bfloat16 while v11 and v1 do not.
  EXPECT_NE(reduce_sum_v13->type_constraints()[0].allowed_type_strs,
            reduce_sum_v11->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(reduce_sum_v1->type_constraints()[0].allowed_type_strs,
            reduce_sum_v11->type_constraints()[0].allowed_type_strs);

  // Docs differ across versions.
  EXPECT_NE(reduce_sum_v13->doc(), reduce_sum_v11->doc());
  EXPECT_NE(reduce_sum_v13->doc(), reduce_sum_v1->doc());
  EXPECT_NE(reduce_sum_v11->doc(), reduce_sum_v1->doc());
}

} // namespace Test
