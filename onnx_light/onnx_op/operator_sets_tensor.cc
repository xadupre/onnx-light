// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"
#include "onnx_op/operator_sets_tensor_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

void AppendCastSchema(std::vector<LightOpSchema> &schemas, int version,
                      const std::vector<std::string> &types,
                      const char *input_constraint_description,
                      const char *output_constraint_description) {
  schemas.push_back(LightOpSchema("Cast", kOnnxDomain, version, MakeCastDoc(),
                                  {
                                      {"input", MakeCastInputDescription(), "T1"},
                                  },
                                  {
                                      {"output", MakeCastOutputDescription(), "T2"},
                                  },
                                  {
                                      {"T1", types, input_constraint_description},
                                      {"T2", types, output_constraint_description},
                                  }));
}

std::vector<LightOpSchema> GetAllOnnxOpTensorSchemasWithHistory() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(9);
  const char *legacy_input_constraint = MakeCastLegacyInputConstraintDescription();
  const char *legacy_output_constraint = MakeCastLegacyOutputConstraintDescription();
  const char *input_constraint = MakeCastInputConstraintDescription();
  const char *output_constraint = MakeCastOutputConstraintDescription();
  AppendCastSchema(schemas, 1, CastTypesV1V6Strings(), legacy_input_constraint,
                   legacy_output_constraint);
  AppendCastSchema(schemas, 6, CastTypesV1V6Strings(), legacy_input_constraint,
                   legacy_output_constraint);
  AppendCastSchema(schemas, 9, CastTypesV9Strings(), input_constraint, output_constraint);
  AppendCastSchema(schemas, 13, CastTypesV13Strings(), input_constraint, output_constraint);
  AppendCastSchema(schemas, 19, CastTypesV19Strings(), input_constraint, output_constraint);
  AppendCastSchema(schemas, 21, CastTypesV21Strings(), input_constraint, output_constraint);
  AppendCastSchema(schemas, 23, CastTypesV23Strings(), input_constraint, output_constraint);
  AppendCastSchema(schemas, 24, CastTypesV24Strings(), input_constraint, output_constraint);
  AppendCastSchema(schemas, 25, CastTypesV25Strings(), input_constraint, output_constraint);
  return schemas;
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
