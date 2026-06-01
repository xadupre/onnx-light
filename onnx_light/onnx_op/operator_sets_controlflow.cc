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
      },
      {
          AttributeParam{"then_branch", MakeIfThenBranchAttributeDescription(),
                         AttributeType::GRAPH,
                         /*required=*/true, std::monostate{}},
          AttributeParam{"else_branch", MakeIfElseBranchAttributeDescription(),
                         AttributeType::GRAPH,
                         /*required=*/true, std::monostate{}},
      });
}

std::vector<TensorType> LoopTypes(int since_version) {
  if (since_version >= 13) {
    std::vector<TensorType> types = AllTensorTypes();
    const std::vector<TensorType> seq_types = AllTensorSequenceTypes();
    types.insert(types.end(), seq_types.begin(), seq_types.end());
    return types;
  }
  if (since_version == 11 || since_version == 1) {
    return AllTensorTypes();
  }
  throw SchemaError("Unsupported Loop since_version: " + std::to_string(since_version));
}

std::string LoopTypeConstraintDescription(int since_version) {
  return since_version >= 13 ? "All Tensor and Sequence types" : "All Tensor types";
}

LightOpSchema MakeLoopSchema(int since_version) {
  return LightOpSchema(
      "Loop", kOnnxDomain, since_version, MakeLoopDoc(since_version),
      {
          {"M",
           "A maximum trip-count for the loop specified at runtime. Optional. "
           "Pass empty string to skip.",
           "I"},
          {"cond", "A boolean termination condition. Optional. Pass empty string to skip.", "B"},
          {"v_initial",
           "The initial values of any loop-carried dependencies (values that "
           "change across loop iterations)",
           "V"},
      },
      {
          {"v_final_and_scan_outputs", MakeLoopOutputDescription(since_version), "V"},
      },
      {
          {"V", LoopTypes(since_version), LoopTypeConstraintDescription(since_version)},
          {"I", {TensorType::kInt64}, "tensor of int64, which should be a scalar."},
          {"B", {TensorType::kBool}, "tensor of bool, which should be a scalar."},
      },
      {
          AttributeParam{"body", MakeLoopBodyAttributeDescription(), AttributeType::GRAPH,
                         /*required=*/true, std::monostate{}},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpControlflowSchemasWithHistory(const std::string &op_type,
                                                                     bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"If",
       [] {
         return std::vector<LightOpSchema>{
             MakeIfSchema(13),
             MakeIfSchema(11),
             MakeIfSchema(1),
         };
       }},
      {"Loop",
       [] {
         return std::vector<LightOpSchema>{
             MakeLoopSchema(13),
             MakeLoopSchema(11),
             MakeLoopSchema(1),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
