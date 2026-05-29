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

constexpr size_t kExpectedTextSchemaCount = 4;

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

TEST(OnnxOpTextRegistrationTest, ReturnsStringSplitSchema) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::text::GetAllOnnxOpTextSchemasWithHistory();

  const onnx_op::LightOpSchema *const string_split_v20 = FindTextSchema(schemas, "StringSplit", 20);
  ASSERT_NE(nullptr, string_split_v20);

  EXPECT_EQ(string_split_v20->domain(), "ai.onnx");
  EXPECT_EQ(string_split_v20->inputs().size(), 1u);
  EXPECT_EQ(string_split_v20->outputs().size(), 2u);
  EXPECT_EQ(string_split_v20->type_constraints().size(), 3u);

  EXPECT_EQ(string_split_v20->inputs()[0].name, "X");
  EXPECT_EQ(string_split_v20->inputs()[0].type, "T1");

  EXPECT_EQ(string_split_v20->outputs()[0].name, "Y");
  EXPECT_EQ(string_split_v20->outputs()[0].type, "T2");
  EXPECT_EQ(string_split_v20->outputs()[1].name, "Z");
  EXPECT_EQ(string_split_v20->outputs()[1].type, "T3");

  EXPECT_EQ(string_split_v20->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(string_split_v20->type_constraints()[0].allowed_type_strs,
            std::vector<onnx_op::TensorType>{onnx_op::TensorType::kString});
  EXPECT_EQ(string_split_v20->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(string_split_v20->type_constraints()[1].allowed_type_strs,
            std::vector<onnx_op::TensorType>{onnx_op::TensorType::kString});
  EXPECT_EQ(string_split_v20->type_constraints()[2].type_param_str, "T3");
  EXPECT_EQ(string_split_v20->type_constraints()[2].allowed_type_strs,
            std::vector<onnx_op::TensorType>{onnx_op::TensorType::kInt64});

  EXPECT_NE(string_split_v20->doc().find("StringSplit splits a string tensor"), std::string::npos);
}

TEST(OnnxOpTextRegistrationTest, ReturnsRegexFullMatchSchema) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::text::GetAllOnnxOpTextSchemasWithHistory();

  const onnx_op::LightOpSchema *const regex_v20 = FindTextSchema(schemas, "RegexFullMatch", 20);
  ASSERT_NE(nullptr, regex_v20);

  EXPECT_EQ(regex_v20->domain(), "ai.onnx");
  EXPECT_EQ(regex_v20->inputs().size(), 1u);
  EXPECT_EQ(regex_v20->outputs().size(), 1u);
  EXPECT_EQ(regex_v20->type_constraints().size(), 2u);

  EXPECT_EQ(regex_v20->inputs()[0].name, "X");
  EXPECT_EQ(regex_v20->inputs()[0].type, "T1");
  EXPECT_EQ(regex_v20->outputs()[0].name, "Y");
  EXPECT_EQ(regex_v20->outputs()[0].type, "T2");

  EXPECT_EQ(regex_v20->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(regex_v20->type_constraints()[0].allowed_type_strs,
            std::vector<onnx_op::TensorType>{onnx_op::TensorType::kString});
  EXPECT_EQ(regex_v20->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(regex_v20->type_constraints()[1].allowed_type_strs,
            std::vector<onnx_op::TensorType>{onnx_op::TensorType::kBool});

  EXPECT_NE(regex_v20->doc().find("RegexFullMatch performs a full regex match"), std::string::npos);
}

TEST(OnnxOpTextRegistrationTest, ReturnsStringNormalizerSchema) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::text::GetAllOnnxOpTextSchemasWithHistory();

  const onnx_op::LightOpSchema *const normalizer_v10 =
      FindTextSchema(schemas, "StringNormalizer", 10);
  ASSERT_NE(nullptr, normalizer_v10);

  EXPECT_EQ(normalizer_v10->domain(), "ai.onnx");
  EXPECT_EQ(normalizer_v10->inputs().size(), 1u);
  EXPECT_EQ(normalizer_v10->outputs().size(), 1u);
  EXPECT_EQ(normalizer_v10->type_constraints().size(), 0u);

  EXPECT_EQ(normalizer_v10->inputs()[0].name, "X");
  EXPECT_EQ(normalizer_v10->inputs()[0].type, "tensor(string)");
  EXPECT_EQ(normalizer_v10->outputs()[0].name, "Y");
  EXPECT_EQ(normalizer_v10->outputs()[0].type, "tensor(string)");

  EXPECT_NE(normalizer_v10->doc().find("StringNormalization performs string operations"),
            std::string::npos);
}

TEST(OnnxOpTextRegistrationTest, StripDocsRemovesDocumentation) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::text::GetAllOnnxOpTextSchemasWithHistory(/*op_type=*/"", /*init_doc=*/false);
  EXPECT_EQ(schemas.size(), kExpectedTextSchemaCount);
  for (const auto &schema : schemas) {
    EXPECT_EQ(schema.doc(), "");
  }
}

} // namespace Test
