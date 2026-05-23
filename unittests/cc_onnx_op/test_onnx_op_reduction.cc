// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_reduction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
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

  // Counts per op: ReduceSum=3 (v1,v11,v13); ReduceMax/ReduceMin=6 each
  // (v1,v11,v12,v13,v18,v20); ArgMax/ArgMin=4 each (v1,v11,v12,v13); all other
  // reduce ops (Mean, Prod, SumSquare, LogSum, LogSumExp, L1, L2)=4 each
  // (v1,v11,v13,v18).
  EXPECT_EQ(schemas.size(), 51u);

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

TEST(OnnxOpReductionRegistrationTest, SimpleReduceOpsShareVersionStructure) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::reduction::GetAllOnnxOpReductionSchemasWithHistory();

  const std::vector<std::string> simple_ops = {
      "ReduceMean", "ReduceProd",      "ReduceSumSquare", "ReduceLogSum",
      "ReduceL1",   "ReduceLogSumExp", "ReduceL2",
  };
  for (const std::string &op : simple_ops) {
    const onnx_op::LightOpSchema *const v18 = FindReductionSchema(schemas, op, 18);
    const onnx_op::LightOpSchema *const v13 = FindReductionSchema(schemas, op, 13);
    const onnx_op::LightOpSchema *const v11 = FindReductionSchema(schemas, op, 11);
    const onnx_op::LightOpSchema *const v1 = FindReductionSchema(schemas, op, 1);
    ASSERT_NE(nullptr, v18) << op;
    ASSERT_NE(nullptr, v13) << op;
    ASSERT_NE(nullptr, v11) << op;
    ASSERT_NE(nullptr, v1) << op;

    // v18: axes moved to optional second input.
    EXPECT_EQ(v18->inputs().size(), 2u) << op;
    EXPECT_EQ(v18->inputs()[0].name, "data") << op;
    EXPECT_EQ(v18->inputs()[1].name, "axes") << op;
    EXPECT_EQ(v18->outputs().size(), 1u) << op;
    EXPECT_EQ(v18->type_constraints()[0].allowed_type_strs,
              onnx_op::NumericTypesForMathReductionIr4())
        << op;

    // v13: axes remain an attribute; bfloat16 supported.
    EXPECT_EQ(v13->inputs().size(), 1u) << op;
    EXPECT_EQ(v13->type_constraints()[0].allowed_type_strs,
              onnx_op::NumericTypesForMathReductionIr4())
        << op;

    // v11 / v1: axes attribute, no bfloat16.
    EXPECT_EQ(v11->inputs().size(), 1u) << op;
    EXPECT_EQ(v11->type_constraints()[0].allowed_type_strs, onnx_op::NumericTypesForMathReduction())
        << op;
    EXPECT_EQ(v1->inputs().size(), 1u) << op;
    EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs, onnx_op::NumericTypesForMathReduction())
        << op;

    // Docs differ across the v18 -> v1 boundary (and other transitions).
    EXPECT_NE(v18->doc(), v13->doc()) << op;
    EXPECT_NE(v13->doc(), v11->doc()) << op;
    EXPECT_NE(v11->doc(), v1->doc()) << op;
  }
}

TEST(OnnxOpReductionRegistrationTest, ReduceMaxMinSupports8BitAndBool) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::reduction::GetAllOnnxOpReductionSchemasWithHistory();

  for (const std::string &op : {"ReduceMax", "ReduceMin"}) {
    const onnx_op::LightOpSchema *const v20 = FindReductionSchema(schemas, op, 20);
    const onnx_op::LightOpSchema *const v18 = FindReductionSchema(schemas, op, 18);
    const onnx_op::LightOpSchema *const v13 = FindReductionSchema(schemas, op, 13);
    const onnx_op::LightOpSchema *const v12 = FindReductionSchema(schemas, op, 12);
    const onnx_op::LightOpSchema *const v11 = FindReductionSchema(schemas, op, 11);
    const onnx_op::LightOpSchema *const v1 = FindReductionSchema(schemas, op, 1);
    ASSERT_NE(nullptr, v20) << op;
    ASSERT_NE(nullptr, v18) << op;
    ASSERT_NE(nullptr, v13) << op;
    ASSERT_NE(nullptr, v12) << op;
    ASSERT_NE(nullptr, v11) << op;
    ASSERT_NE(nullptr, v1) << op;

    // v20: axes input, includes bool type.
    EXPECT_EQ(v20->inputs().size(), 2u) << op;
    const auto &types_v20 = v20->type_constraints()[0].allowed_type_strs;
    EXPECT_NE(std::find(types_v20.begin(), types_v20.end(), onnx_op::TensorType::kBool),
              types_v20.end())
        << op;

    // v18: axes input, includes 8-bit but not bool.
    EXPECT_EQ(v18->inputs().size(), 2u) << op;
    const auto &types_v18 = v18->type_constraints()[0].allowed_type_strs;
    EXPECT_NE(std::find(types_v18.begin(), types_v18.end(), onnx_op::TensorType::kUint8),
              types_v18.end())
        << op;
    EXPECT_NE(std::find(types_v18.begin(), types_v18.end(), onnx_op::TensorType::kInt8),
              types_v18.end())
        << op;
    EXPECT_EQ(std::find(types_v18.begin(), types_v18.end(), onnx_op::TensorType::kBool),
              types_v18.end())
        << op;

    // v13/v12 keep axes as an attribute and include 8-bit numeric tensors.
    EXPECT_EQ(v13->inputs().size(), 1u) << op;
    EXPECT_EQ(v12->inputs().size(), 1u) << op;
    const auto &types_v12 = v12->type_constraints()[0].allowed_type_strs;
    EXPECT_NE(std::find(types_v12.begin(), types_v12.end(), onnx_op::TensorType::kInt8),
              types_v12.end())
        << op;

    // v11 / v1 are the pre-8-bit baseline.
    EXPECT_EQ(v11->type_constraints()[0].allowed_type_strs, onnx_op::NumericTypesForMathReduction())
        << op;
    EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs, onnx_op::NumericTypesForMathReduction())
        << op;
  }
}

TEST(OnnxOpReductionRegistrationTest, ArgReduceSchemas) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::reduction::GetAllOnnxOpReductionSchemasWithHistory();

  for (const std::string &op : {"ArgMax", "ArgMin"}) {
    const onnx_op::LightOpSchema *const v13 = FindReductionSchema(schemas, op, 13);
    const onnx_op::LightOpSchema *const v12 = FindReductionSchema(schemas, op, 12);
    const onnx_op::LightOpSchema *const v11 = FindReductionSchema(schemas, op, 11);
    const onnx_op::LightOpSchema *const v1 = FindReductionSchema(schemas, op, 1);
    ASSERT_NE(nullptr, v13) << op;
    ASSERT_NE(nullptr, v12) << op;
    ASSERT_NE(nullptr, v11) << op;
    ASSERT_NE(nullptr, v1) << op;

    for (const onnx_op::LightOpSchema *s : {v13, v12, v11, v1}) {
      EXPECT_EQ(s->inputs().size(), 1u) << op;
      EXPECT_EQ(s->inputs()[0].name, "data") << op;
      EXPECT_EQ(s->outputs().size(), 1u) << op;
      EXPECT_EQ(s->outputs()[0].name, "reduced") << op;
      // Output is int64.
      EXPECT_EQ(s->outputs()[0].type, "tensor(int64)") << op;
    }

    // v13 includes bfloat16; earlier versions do not.
    EXPECT_EQ(v13->type_constraints()[0].allowed_type_strs, onnx_op::AllNumericTypesIr4());
    EXPECT_EQ(v12->type_constraints()[0].allowed_type_strs, onnx_op::AllNumericTypes());
    EXPECT_EQ(v11->type_constraints()[0].allowed_type_strs, onnx_op::AllNumericTypes());
    EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs, onnx_op::AllNumericTypes());

    // Docs differ between v12 (select_last_index) and v11 / v1.
    EXPECT_NE(v12->doc(), v11->doc()) << op;
    EXPECT_NE(v11->doc(), v1->doc()) << op;
  }
}

} // namespace Test
