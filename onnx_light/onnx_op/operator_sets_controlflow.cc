// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_controlflow.h"
#include "onnx_op/operator_sets_controlflow_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace controlflow {

namespace {

std::vector<TensorType> IfTypes(int since_version) {
  if (since_version == 13) {
    std::vector<TensorType> types = AllTensorTypes();
    const std::vector<TensorType> seq_types = AllTensorSequenceTypes();
    types.insert(types.end(), seq_types.begin(), seq_types.end());
    return types;
  }
  if (since_version == 11 || since_version == 1) {
    return AllTensorTypes();
  }
  throw SchemaError("Unsupported If since_version: " + std::to_string(since_version));
}

std::string IfTypeConstraintDescription(int since_version) {
  return since_version == 13 ? "All Tensor and Sequence types" : "All Tensor types";
}

LightOpSchema MakeIfSchema(int since_version) {
  return LightOpSchema(
      "If", kOnnxDomain, since_version, MakeIfDoc(),
      {
          {"cond", "Condition for the if. The tensor must contain a single element.", "B"},
      },
      {
          {"outputs", MakeIfOutputDescription(since_version), "V"},
      },
      {
          {"V", IfTypes(since_version), IfTypeConstraintDescription(since_version)},
          {"B", {TensorType::kBool}, "Only bool"},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpControlflowSchemasWithHistory(bool init_doc,
                                                                     const std::string &op_type) {
  std::vector<LightOpSchema> schemas{
      MakeIfSchema(13),
      MakeIfSchema(11),
      MakeIfSchema(1),
  };
  return FilterSchemasByOpType(init_doc ? schemas : StripDocs(schemas), op_type);
}

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
