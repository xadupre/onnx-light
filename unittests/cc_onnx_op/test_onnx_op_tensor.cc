// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"

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

const onnx_op::tensor::LightOpSchema *
FindTensorSchema(const std::vector<onnx_op::tensor::LightOpSchema> &schemas,
                 const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpTensorRegistrationTest, ReturnsCastSchemasWithoutShapeInference) {
  const std::vector<onnx_op::tensor::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 9u);

  const onnx_op::tensor::LightOpSchema *const cast_v1 = FindTensorSchema(schemas, "Cast", 1);
  const onnx_op::tensor::LightOpSchema *const cast_v9 = FindTensorSchema(schemas, "Cast", 9);
  const onnx_op::tensor::LightOpSchema *const cast_v21 = FindTensorSchema(schemas, "Cast", 21);
  const onnx_op::tensor::LightOpSchema *const cast_v25 = FindTensorSchema(schemas, "Cast", 25);
  ASSERT_NE(nullptr, cast_v1);
  ASSERT_NE(nullptr, cast_v9);
  ASSERT_NE(nullptr, cast_v21);
  ASSERT_NE(nullptr, cast_v25);
  EXPECT_EQ(cast_v1->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(cast_v9->type_constraints()[0].allowed_type_strs.size(), 13u);
  EXPECT_EQ(cast_v21->type_constraints()[0].allowed_type_strs.size(), 20u);
  EXPECT_EQ(cast_v25->type_constraints()[0].allowed_type_strs.size(), 24u);
  EXPECT_EQ(cast_v25->inputs()[0].description, "Input tensor to be cast.");
  EXPECT_EQ(cast_v25->outputs()[0].name, "output");
  EXPECT_NE(cast_v1->type_constraints()[0].allowed_type_strs,
            cast_v25->type_constraints()[0].allowed_type_strs);
}

} // namespace Test
