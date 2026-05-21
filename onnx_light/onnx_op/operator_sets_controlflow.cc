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

std::vector<TensorType> AllTensorTypes() {
  return {
      TensorType::kUint8,   TensorType::kUint16,    TensorType::kUint32,     TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,     TensorType::kInt32,      TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,     TensorType::kString,
      TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<TensorType> AllTensorSequenceTypes() {
  return {
      TensorType::kSeqUint8,  TensorType::kSeqUint16,    TensorType::kSeqUint32,
      TensorType::kSeqUint64, TensorType::kSeqInt8,      TensorType::kSeqInt16,
      TensorType::kSeqInt32,  TensorType::kSeqInt64,     TensorType::kSeqFloat16,
      TensorType::kSeqFloat,  TensorType::kSeqDouble,    TensorType::kSeqString,
      TensorType::kSeqBool,   TensorType::kSeqComplex64, TensorType::kSeqComplex128,
  };
}

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

std::vector<LightOpSchema> GetAllOnnxOpControlflowSchemasWithHistory() {
  return std::vector<LightOpSchema>{
      MakeIfSchema(13),
      MakeIfSchema(11),
      MakeIfSchema(1),
  };
}

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
