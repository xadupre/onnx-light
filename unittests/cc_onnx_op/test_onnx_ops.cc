// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets.h"

#include "onnx_lib/defs/operator_sets.h"
#include "onnx_lib/defs/schema.h"
#include <gtest/gtest.h>

#ifdef ONNX_LIGHT_NAMESPACE
// onnx_lib headers define ONNX_LIGHT_NAMESPACE as a macro alias (onnx_light),
// while onnx_op headers in this target use the literal ONNX_LIGHT_NAMESPACE namespace.
// Undefining keeps this test bound to onnx_op symbols while still using onnx_lib APIs explicitly.
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

void ExpectSameFormalParameters(
    const std::vector<onnx_op::FormalParameter> &light_params,
    const std::vector<onnx_light::OpSchema::FormalParameter> &onnx_params) {
  ASSERT_EQ(light_params.size(), onnx_params.size());
  for (size_t i = 0; i < light_params.size(); ++i) {
    SCOPED_TRACE("FormalParameter index " + std::to_string(i));
    ASSERT_EQ(light_params[i].name, onnx_params[i].GetName());
    ASSERT_EQ(light_params[i].description, onnx_params[i].GetDescription());
    ASSERT_EQ(light_params[i].type, onnx_params[i].GetTypeStr());
  }
}

void ExpectSameTypeConstraints(
    const std::vector<onnx_op::TypeConstraintParam> &light_constraints,
    const std::vector<onnx_light::OpSchema::TypeConstraintParam> &onnx_constraints) {
  ASSERT_EQ(light_constraints.size(), onnx_constraints.size());
  for (size_t i = 0; i < light_constraints.size(); ++i) {
    SCOPED_TRACE("TypeConstraint index " + std::to_string(i));
    ASSERT_EQ(light_constraints[i].type_param_str, onnx_constraints[i].type_param_str);
    ASSERT_EQ(light_constraints[i].description, onnx_constraints[i].description);
    ASSERT_EQ(light_constraints[i].allowed_type_strs.size(),
              onnx_constraints[i].allowed_type_strs.size());
    for (size_t j = 0; j < light_constraints[i].allowed_type_strs.size(); ++j) {
      ASSERT_EQ(onnx_op::ToTypeString(light_constraints[i].allowed_type_strs[j]),
                onnx_constraints[i].allowed_type_strs[j]);
    }
  }
}

TEST(OnnxOpSchemaParityTest, MatchesOnnxLibDefinitionsForAllOnnxOpSchemas) {
  onnx_light::RegisterOnnxOperatorSetSchema(0, false);
  const std::vector<onnx_op::LightOpSchema> all_schemas = onnx_op::GetAllOnnxOpSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> math_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory();
  const std::vector<onnx_op::controlflow::LightOpSchema> controlflow_schemas =
      onnx_op::controlflow::GetAllOnnxOpControlflowSchemasWithHistory();
  const std::vector<onnx_op::generator::LightOpSchema> generator_schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory();
  const std::vector<onnx_op::image::LightOpSchema> image_schemas =
      onnx_op::image::GetAllOnnxOpImageSchemasWithHistory();
  const std::vector<onnx_op::logical::LightOpSchema> logical_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory();
  const std::vector<onnx_op::nn::LightOpSchema> nn_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();
  const std::vector<onnx_op::object_detection::LightOpSchema> object_detection_schemas =
      onnx_op::object_detection::GetAllOnnxOpObjectDetectionSchemasWithHistory();
  const std::vector<onnx_op::optional::LightOpSchema> optional_schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory();
  const std::vector<onnx_op::preview::LightOpSchema> preview_schemas =
      onnx_op::preview::GetAllOnnxOpPreviewSchemasWithHistory();
  const std::vector<onnx_op::quantization::LightOpSchema> quantization_schemas =
      onnx_op::quantization::GetAllOnnxOpQuantizationSchemasWithHistory();
  const std::vector<onnx_op::reduction::LightOpSchema> reduction_schemas =
      onnx_op::reduction::GetAllOnnxOpReductionSchemasWithHistory();
  const std::vector<onnx_op::sequence::LightOpSchema> sequence_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory();
  const std::vector<onnx_op::tensor::LightOpSchema> tensor_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();
  const std::vector<onnx_op::text::LightOpSchema> text_schemas =
      onnx_op::text::GetAllOnnxOpTextSchemasWithHistory();
  const std::vector<onnx_op::traditionalml::LightOpSchema> traditionalml_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory();
  const std::vector<onnx_op::training::LightOpSchema> training_schemas =
      onnx_op::training::GetAllOnnxOpTrainingSchemasWithHistory();

  const size_t expected_total =
      controlflow_schemas.size() + generator_schemas.size() + image_schemas.size() +
      math_schemas.size() + logical_schemas.size() + nn_schemas.size() +
      object_detection_schemas.size() + optional_schemas.size() + preview_schemas.size() +
      quantization_schemas.size() + reduction_schemas.size() + sequence_schemas.size() +
      tensor_schemas.size() + text_schemas.size() + traditionalml_schemas.size() +
      training_schemas.size();
  ASSERT_EQ(all_schemas.size(), expected_total);

  for (const onnx_op::reduction::LightOpSchema &reduction_schema : reduction_schemas) {
    bool found = false;
    for (const onnx_op::LightOpSchema &schema : all_schemas) {
      if (schema.name() == reduction_schema.name() &&
          schema.since_version() == reduction_schema.since_version() &&
          schema.domain() == reduction_schema.domain()) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }

  std::vector<onnx_op::LightOpSchema> parity_schemas;
  parity_schemas.reserve(all_schemas.size() - reduction_schemas.size());
  parity_schemas.insert(parity_schemas.end(), math_schemas.begin(), math_schemas.end());
  parity_schemas.insert(parity_schemas.end(), controlflow_schemas.begin(),
                        controlflow_schemas.end());
  parity_schemas.insert(parity_schemas.end(), generator_schemas.begin(), generator_schemas.end());
  parity_schemas.insert(parity_schemas.end(), image_schemas.begin(), image_schemas.end());
  parity_schemas.insert(parity_schemas.end(), logical_schemas.begin(), logical_schemas.end());
  parity_schemas.insert(parity_schemas.end(), nn_schemas.begin(), nn_schemas.end());
  parity_schemas.insert(parity_schemas.end(), object_detection_schemas.begin(),
                        object_detection_schemas.end());
  parity_schemas.insert(parity_schemas.end(), optional_schemas.begin(), optional_schemas.end());
  parity_schemas.insert(parity_schemas.end(), preview_schemas.begin(), preview_schemas.end());
  parity_schemas.insert(parity_schemas.end(), quantization_schemas.begin(),
                        quantization_schemas.end());
  parity_schemas.insert(parity_schemas.end(), sequence_schemas.begin(), sequence_schemas.end());
  parity_schemas.insert(parity_schemas.end(), tensor_schemas.begin(), tensor_schemas.end());
  parity_schemas.insert(parity_schemas.end(), text_schemas.begin(), text_schemas.end());
  parity_schemas.insert(parity_schemas.end(), traditionalml_schemas.begin(),
                        traditionalml_schemas.end());
  parity_schemas.insert(parity_schemas.end(), training_schemas.begin(), training_schemas.end());

  for (const onnx_op::LightOpSchema &schema : parity_schemas) {
    SCOPED_TRACE(schema.name() + "@" + std::to_string(schema.since_version()));
    const std::string domain =
        onnx_light::IsOnnxDomain(schema.domain()) ? onnx_light::ONNX_DOMAIN : schema.domain();
    const onnx_light::OpSchema *const onnx_lib_schema =
        onnx_light::OpSchemaRegistry::Schema(schema.name(), schema.since_version(), domain);
    ASSERT_NE(onnx_lib_schema, nullptr);
    ASSERT_EQ(onnx_lib_schema->Name(), schema.name());
    ASSERT_EQ(onnx_lib_schema->SinceVersion(), schema.since_version());
    ExpectSameFormalParameters(schema.inputs(), onnx_lib_schema->inputs());
    ExpectSameFormalParameters(schema.outputs(), onnx_lib_schema->outputs());
    ExpectSameTypeConstraints(schema.type_constraints(), onnx_lib_schema->typeConstraintParams());
  }
}

TEST(LightOpSchemaTest, InitDocDefaultStoresDoc) {
  onnx_op::LightOpSchema schema("Op", onnx_op::kOnnxDomain, 1, "some documentation", {}, {}, {});
  EXPECT_EQ(schema.doc(), "some documentation");
}

TEST(LightOpSchemaTest, InitDocFalseDiscardsDoc) {
  onnx_op::LightOpSchema schema("Op", onnx_op::kOnnxDomain, 1, "some documentation", {}, {}, {},
                                /*has_function_implementation=*/false, /*init_doc=*/false);
  EXPECT_TRUE(schema.doc().empty());
}

TEST(LightOpSchemaTest, GetAllOnnxOpSchemasInitDocFalseStripsDocs) {
  const std::vector<onnx_op::LightOpSchema> with_docs =
      onnx_op::GetAllOnnxOpSchemasWithHistory(/*init_doc=*/true);
  const std::vector<onnx_op::LightOpSchema> without_docs =
      onnx_op::GetAllOnnxOpSchemasWithHistory(/*init_doc=*/false);

  ASSERT_EQ(with_docs.size(), without_docs.size());
  ASSERT_FALSE(with_docs.empty());
  for (const onnx_op::LightOpSchema &schema : without_docs) {
    EXPECT_TRUE(schema.doc().empty());
  }
  bool any_doc_populated = false;
  for (const onnx_op::LightOpSchema &schema : with_docs) {
    if (!schema.doc().empty()) {
      any_doc_populated = true;
      break;
    }
  }
  EXPECT_TRUE(any_doc_populated);
}

TEST(LightOpSchemaTest, GetAllOnnxOpSchemasOpNameFiltersByOperator) {
  const std::vector<onnx_op::LightOpSchema> all_schemas =
      onnx_op::GetAllOnnxOpSchemasWithHistory(/*init_doc=*/false);
  ASSERT_FALSE(all_schemas.empty());

  const std::vector<onnx_op::LightOpSchema> abs_schemas =
      onnx_op::GetAllOnnxOpSchemasWithHistory(/*init_doc=*/false, /*op_type=*/"Abs");
  ASSERT_FALSE(abs_schemas.empty());
  for (const onnx_op::LightOpSchema &schema : abs_schemas) {
    EXPECT_EQ(schema.name(), "Abs");
  }

  size_t expected_count = 0;
  for (const onnx_op::LightOpSchema &schema : all_schemas) {
    if (schema.name() == "Abs") {
      ++expected_count;
    }
  }
  EXPECT_EQ(abs_schemas.size(), expected_count);

  const std::vector<onnx_op::LightOpSchema> unknown_schemas =
      onnx_op::GetAllOnnxOpSchemasWithHistory(/*init_doc=*/false,
                                              /*op_type=*/"ThisOpDoesNotExist");
  EXPECT_TRUE(unknown_schemas.empty());
}

} // namespace Test
