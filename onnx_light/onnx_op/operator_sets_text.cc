// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_text.h"
#include "onnx_op/operator_sets_text_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace text {

namespace {

LightOpSchema MakeStringConcatSchema(int since_version) {
  return LightOpSchema("StringConcat", kOnnxDomain, since_version,
                       MakeStringConcatDoc(since_version),
                       {
                           {"X", "Tensor to prepend in concatenation", "T"},
                           {"Y", "Tensor to append in concatenation", "T"},
                       },
                       {
                           {"Z", "Concatenated string tensor", "T"},
                       },
                       {
                           {"T", {TensorType::kString}, "Inputs and outputs must be UTF-8 strings"},
                       });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpTextSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas{
      MakeStringConcatSchema(20),
  };
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace text
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
