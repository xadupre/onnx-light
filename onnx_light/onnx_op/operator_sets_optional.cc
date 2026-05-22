// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_optional.h"
#include "onnx_op/operator_sets_optional_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace optional {

namespace {

std::vector<TensorType> OptionalInputTypes() {
  std::vector<TensorType> types = AllTensorTypes();
  const std::vector<TensorType> seq_types = AllTensorSequenceTypes();
  types.insert(types.end(), seq_types.begin(), seq_types.end());
  return types;
}

std::vector<TensorType> OptionalAndTensorTypes() {
  std::vector<TensorType> types = AllOptionalTypes();
  const std::vector<TensorType> tensor_types = AllTensorTypes();
  const std::vector<TensorType> seq_types = AllTensorSequenceTypes();
  types.insert(types.end(), tensor_types.begin(), tensor_types.end());
  types.insert(types.end(), seq_types.begin(), seq_types.end());
  return types;
}

LightOpSchema MakeOptionalSchema(int since_version) {
  return LightOpSchema(
      "Optional", kOnnxDomain, since_version, MakeOptionalDoc(since_version),
      {
          {"input", "The input element.", "V"},
      },
      {
          {"output", "The optional output enclosing the input element.", "O"},
      },
      {
          {"V", OptionalInputTypes(), "Constrain input type to all tensor and sequence types."},
          {"O", AllOptionalTypes(),
           "Constrain output type to all optional tensor or optional sequence types."},
      });
}

LightOpSchema MakeOptionalHasElementSchema(int since_version) {
  const std::vector<TensorType> input_types =
      since_version == 18 ? OptionalAndTensorTypes() : AllOptionalTypes();
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
           "Constrain input type to optional tensor and optional sequence types."},
          {"B", {TensorType::kBool}, "Constrain output to a boolean tensor."},
      });
}

LightOpSchema MakeOptionalGetElementSchema(int since_version) {
  const std::vector<TensorType> input_types =
      since_version == 18 ? OptionalAndTensorTypes() : AllOptionalTypes();
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
           "Constrain input type to optional tensor and optional sequence types."},
          {"V", OptionalInputTypes(), "Constrain output type to all tensor or sequence types."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpOptionalSchemasWithHistory() {
  return std::vector<LightOpSchema>{
      MakeOptionalSchema(15),           MakeOptionalGetElementSchema(18),
      MakeOptionalGetElementSchema(15), MakeOptionalHasElementSchema(18),
      MakeOptionalHasElementSchema(15),
  };
}

} // namespace optional
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
