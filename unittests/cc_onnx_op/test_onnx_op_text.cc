// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_text.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedTextSchemaCount = 1;

const onnx_op::LightOpSchema *FindTextSchema(const std::vector<onnx_op::LightOpSchema> &schemas,
                                             const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpTextRegistrationTest, ReturnsStringConcatSchema) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::text::GetAllOnnxOpTextSchemasWithHistory();

  EXPECT_EQ(schemas.size(), kExpectedTextSchemaCount);

  const onnx_op::LightOpSchema *const string_concat_v20 =
      FindTextSchema(schemas, "StringConcat", 20);
  ASSERT_NE(nullptr, string_concat_v20);

  EXPECT_EQ(string_concat_v20->domain(), "ai.onnx");
  EXPECT_EQ(string_concat_v20->inputs().size(), 2u);
  EXPECT_EQ(string_concat_v20->outputs().size(), 1u);
  EXPECT_EQ(string_concat_v20->type_constraints().size(), 1u);

  EXPECT_EQ(string_concat_v20->inputs()[0].name, "X");
  EXPECT_EQ(string_concat_v20->inputs()[0].description, "Tensor to prepend in concatenation");
  EXPECT_EQ(string_concat_v20->inputs()[0].type, "T");

  EXPECT_EQ(string_concat_v20->inputs()[1].name, "Y");
  EXPECT_EQ(string_concat_v20->inputs()[1].description, "Tensor to append in concatenation");
  EXPECT_EQ(string_concat_v20->inputs()[1].type, "T");

  EXPECT_EQ(string_concat_v20->outputs()[0].name, "Z");
  EXPECT_EQ(string_concat_v20->outputs()[0].description, "Concatenated string tensor");
  EXPECT_EQ(string_concat_v20->outputs()[0].type, "T");

  EXPECT_EQ(string_concat_v20->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(string_concat_v20->type_constraints()[0].allowed_type_strs,
            std::vector<onnx_op::TensorType>{onnx_op::TensorType::kString});
  EXPECT_EQ(string_concat_v20->type_constraints()[0].description,
            "Inputs and outputs must be UTF-8 strings");

  EXPECT_NE(string_concat_v20->doc().find("StringConcat concatenates string tensors elementwise"),
            std::string::npos);
}

} // namespace Test
