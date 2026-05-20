// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_controlflow.h"

#include <gtest/gtest.h>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

const onnx_op::controlflow::LightOpSchema *
FindControlFlowSchema(const std::vector<onnx_op::controlflow::LightOpSchema> &schemas,
                      const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpControlFlowRegistrationTest, ReturnsIfSchemasWithoutShapeInference) {
  const std::vector<onnx_op::controlflow::LightOpSchema> schemas =
      onnx_op::controlflow::GetAllOnnxOpControlFlowSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 6u);

  const onnx_op::controlflow::LightOpSchema *const if_v25 =
      FindControlFlowSchema(schemas, "If", 25);
  const onnx_op::controlflow::LightOpSchema *const if_v24 =
      FindControlFlowSchema(schemas, "If", 24);
  const onnx_op::controlflow::LightOpSchema *const if_v23 =
      FindControlFlowSchema(schemas, "If", 23);
  const onnx_op::controlflow::LightOpSchema *const if_v21 =
      FindControlFlowSchema(schemas, "If", 21);
  const onnx_op::controlflow::LightOpSchema *const if_v19 =
      FindControlFlowSchema(schemas, "If", 19);
  const onnx_op::controlflow::LightOpSchema *const if_v16 =
      FindControlFlowSchema(schemas, "If", 16);
  ASSERT_NE(nullptr, if_v25);
  ASSERT_NE(nullptr, if_v24);
  ASSERT_NE(nullptr, if_v23);
  ASSERT_NE(nullptr, if_v21);
  ASSERT_NE(nullptr, if_v19);
  ASSERT_NE(nullptr, if_v16);

  EXPECT_EQ(if_v25->domain(), "ai.onnx");
  EXPECT_EQ(if_v25->doc(), "If conditional");
  EXPECT_EQ(if_v25->inputs().size(), 1u);
  EXPECT_EQ(if_v25->outputs().size(), 1u);
  EXPECT_EQ(if_v25->type_constraints().size(), 2u);
  EXPECT_EQ(if_v25->inputs()[0].name, "cond");
  EXPECT_EQ(if_v25->outputs()[0].name, "outputs");
  EXPECT_EQ(if_v25->type_constraints()[0].type_param_str, "V");
  EXPECT_EQ(if_v25->type_constraints()[1].type_param_str, "B");
  EXPECT_EQ(if_v25->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(if_v25->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kBool);
  EXPECT_EQ(if_v16->type_constraints()[0].allowed_type_strs.size(), 64u);
  EXPECT_EQ(if_v19->type_constraints()[0].allowed_type_strs.size(), 76u);
  EXPECT_EQ(if_v21->type_constraints()[0].allowed_type_strs.size(), 82u);
  EXPECT_EQ(if_v23->type_constraints()[0].allowed_type_strs.size(), 85u);
  EXPECT_EQ(if_v24->type_constraints()[0].allowed_type_strs.size(), 88u);
  EXPECT_EQ(if_v25->type_constraints()[0].allowed_type_strs.size(), 94u);
  EXPECT_EQ(if_v25->type_constraints()[0].allowed_type_strs.front(), onnx_op::TensorType::kUint8);
  EXPECT_EQ(if_v25->type_constraints()[0].allowed_type_strs[25], onnx_op::TensorType::kInt2);
  EXPECT_EQ(if_v25->type_constraints()[0].allowed_type_strs[26].type_str, "seq(tensor(uint8))");
  EXPECT_EQ(if_v25->type_constraints()[0].allowed_type_strs[51].type_str, "seq(tensor(int2))");
  EXPECT_EQ(if_v25->type_constraints()[0].allowed_type_strs[52].type_str,
            "optional(seq(tensor(uint8)))");
  EXPECT_EQ(if_v25->type_constraints()[0].allowed_type_strs.back().type_str,
            "optional(tensor(int2))");
  EXPECT_EQ(if_v24->type_constraints()[0].allowed_type_strs.back().type_str,
            "optional(tensor(float8e8m0))");
  EXPECT_EQ(if_v23->type_constraints()[0].allowed_type_strs.back().type_str,
            "optional(tensor(float4e2m1))");
  EXPECT_EQ(if_v21->type_constraints()[0].allowed_type_strs.back().type_str,
            "optional(tensor(int4))");
  EXPECT_EQ(if_v19->type_constraints()[0].allowed_type_strs.back().type_str,
            "optional(tensor(float8e5m2fnuz))");
  EXPECT_EQ(if_v16->type_constraints()[0].allowed_type_strs.back().type_str,
            "optional(tensor(complex128))");
}

} // namespace Test
