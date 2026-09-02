// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_optional.h"
#include "onnx_op/operator_sets_optional_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_op::optional {

namespace {

std::vector<TensorType> OptionalTensorTypesIr14() {
  return {
      TensorType::kUint8,      TensorType::kUint16,         TensorType::kUint32,
      TensorType::kUint64,     TensorType::kInt8,           TensorType::kInt16,
      TensorType::kInt32,      TensorType::kInt64,          TensorType::kBfloat16,
      TensorType::kFloat16,    TensorType::kFloat,          TensorType::kDouble,
      TensorType::kString,     TensorType::kBool,           TensorType::kComplex64,
      TensorType::kComplex128, TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz,
      TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz, TensorType::kUint4,
      TensorType::kInt4,       TensorType::kFloat4e2m1,     TensorType::kFloat8e8m0,
      TensorType::kUint2,      TensorType::kInt2,           TensorType::kFloat6e2m3,
      TensorType::kFloat6e3m2,
  };
}

std::vector<TensorType> OptionalInputTypes() {
  std::vector<TensorType> types = AllTensorTypes();
  const std::vector<TensorType> seq_types = AllTensorSequenceTypes();
  types.insert(types.end(), seq_types.begin(), seq_types.end());
  return types;
}

std::vector<TensorType> OptionalInputTypesIr14() {
  std::vector<TensorType> types = OptionalTensorTypesIr14();
  const auto tensor_types = types;
  types.reserve(2 * tensor_types.size());
  for (const auto type : tensor_types) {
    types.push_back(onnx_proto::SeqTypeOf(type));
  }
  return types;
}

std::vector<TensorType> OptionalTypesIr14() {
  std::vector<TensorType> types;
  types.reserve(2 * OptionalTensorTypesIr14().size());
  for (const auto type : OptionalTensorTypesIr14()) {
    types.push_back(onnx_proto::OptTypeOf(type));
  }
  for (const auto type : OptionalTensorTypesIr14()) {
    types.push_back(onnx_proto::OptSeqTypeOf(type));
  }
  return types;
}

std::vector<TensorType> OptionalAndTensorTypes(int since_version) {
  auto types = since_version == 28 ? OptionalTypesIr14() : AllOptionalTypes();
  const auto element_types = since_version == 28 ? OptionalInputTypesIr14() : OptionalInputTypes();
  types.insert(types.end(), element_types.begin(), element_types.end());
  return types;
}

LightOpSchema MakeOptionalSchema(int since_version) {
  const auto input_types = since_version == 28 ? OptionalInputTypesIr14() : OptionalInputTypes();
  return LightOpSchema(
      "Optional", kOnnxDomain, since_version, MakeOptionalDoc(since_version),
      {
          {"input", "The input element.", "V"},
      },
      {
          {"output", "The optional output enclosing the input element.", "O"},
      },
      {
          {"V", input_types, "Constrain input type to all tensor and sequence types."},
          {"O", since_version == 28 ? OptionalTypesIr14() : AllOptionalTypes(),
           "Constrain output type to all optional tensor or optional sequence types."},
      });
}

LightOpSchema MakeOptionalHasElementSchema(int since_version) {
  const std::vector<TensorType> input_types =
      since_version >= 18 ? OptionalAndTensorTypes(since_version) : AllOptionalTypes();
  return LightOpSchema(
      "OptionalHasElement", kOnnxDomain, since_version, MakeOptionalHasElementDoc(since_version),
      {
          {"input", "The optional input.", "O"},
      },
      {
          {"output",
           "A scalar boolean tensor. If true, it indicates that optional-type input contains "
           "an element. Otherwise, it is empty.",
           "B"},
      },
      {
          {"O", input_types,
           since_version >= 18 ? "Constrain input type to optional, tensor and sequence types."
                               : "Constrain input type to optional tensor and optional sequence "
                                 "types."},
          {"B", {TensorType::kBool}, "Constrain output to a boolean tensor."},
      });
}

LightOpSchema MakeOptionalGetElementSchema(int since_version) {
  const std::vector<TensorType> input_types =
      since_version >= 18 ? OptionalAndTensorTypes(since_version) : AllOptionalTypes();
  const std::vector<TensorType> output_types =
      since_version == 28 ? OptionalInputTypesIr14() : OptionalInputTypes();
  return LightOpSchema(
      "OptionalGetElement", kOnnxDomain, since_version, MakeOptionalGetElementDoc(since_version),
      {
          {"input", "The optional input.", "O"},
      },
      {
          {"output", "Output element in the optional input.", "V"},
      },
      {
          {"O", input_types,
           since_version >= 18 ? "Constrain input type to optional, tensor and sequence types."
                               : "Constrain input type to optional tensor and optional sequence "
                                 "types."},
          {"V", output_types, "Constrain output type to all tensor or sequence types."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpOptionalSchemasWithHistory(const std::string &op_type,
                                                                  bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"Optional",
       [] {
         return std::vector<LightOpSchema>{
             MakeOptionalSchema(28),
             MakeOptionalSchema(15),
         };
       }},
      {"OptionalGetElement",
       [] {
         return std::vector<LightOpSchema>{
             MakeOptionalGetElementSchema(28),
             MakeOptionalGetElementSchema(18),
             MakeOptionalGetElementSchema(15),
         };
       }},
      {"OptionalHasElement",
       [] {
         return std::vector<LightOpSchema>{
             MakeOptionalHasElementSchema(28),
             MakeOptionalHasElementSchema(18),
             MakeOptionalHasElementSchema(15),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::optional
