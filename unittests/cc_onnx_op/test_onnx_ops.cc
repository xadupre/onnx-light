// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_controlflow.h"
#include "onnx_op/operator_sets_generator.h"
#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_sequence.h"
#include "onnx_op/operator_sets_tensor.h"
#include "onnx_op/operator_sets_traditionalml.h"

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
  const std::vector<onnx_op::LightOpSchema> math_schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory();
  const std::vector<onnx_op::controlflow::LightOpSchema> controlflow_schemas =
      onnx_op::controlflow::GetAllOnnxOpControlflowSchemasWithHistory();
  const std::vector<onnx_op::generator::LightOpSchema> generator_schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory();
  const std::vector<onnx_op::logical::LightOpSchema> logical_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory();
  const std::vector<onnx_op::sequence::LightOpSchema> sequence_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory();
  const std::vector<onnx_op::tensor::LightOpSchema> tensor_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();
  const std::vector<onnx_op::traditionalml::LightOpSchema> traditionalml_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory();

  for (const onnx_op::controlflow::LightOpSchema &schema : controlflow_schemas) {
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

  for (const onnx_op::generator::LightOpSchema &schema : generator_schemas) {
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

  for (const onnx_op::LightOpSchema &schema : math_schemas) {
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

  for (const onnx_op::logical::LightOpSchema &schema : logical_schemas) {
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

  for (const onnx_op::sequence::LightOpSchema &schema : sequence_schemas) {
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

  for (const onnx_op::tensor::LightOpSchema &schema : tensor_schemas) {
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

  for (const onnx_op::traditionalml::LightOpSchema &schema : traditionalml_schemas) {
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

} // namespace Test
