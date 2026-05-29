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

LightOpSchema MakeStringSplitSchema(int since_version) {
  return LightOpSchema(
      "StringSplit", kOnnxDomain, since_version, MakeStringSplitDoc(since_version),
      {
          {"X", "Tensor of strings to split.", "T1"},
      },
      {
          {"Y",
           "Tensor of substrings representing the outcome of splitting the strings in the "
           "input on the delimiter. Note that to ensure the same number of elements are "
           "present in the final rank, this tensor will pad any necessary empty strings.",
           "T2"},
          {"Z", "The number of substrings generated for each input element.", "T3"},
      },
      {
          {"T1", {TensorType::kString}, "The input must be a UTF-8 string tensor"},
          {"T2", {TensorType::kString}, "Tensor of substrings."},
          {"T3", {TensorType::kInt64}, "The number of substrings generated."},
      });
}

LightOpSchema MakeRegexFullMatchSchema(int since_version) {
  return LightOpSchema(
      "RegexFullMatch", kOnnxDomain, since_version, MakeRegexFullMatchDoc(since_version),
      {
          {"X", "Tensor with strings to match on.", "T1"},
      },
      {
          {"Y",
           "Tensor of bools indicating if each input string fully matches the regex pattern "
           "specified.",
           "T2"},
      },
      {
          {"T1", {TensorType::kString}, "Inputs must be UTF-8 strings"},
          {"T2",
           {TensorType::kBool},
           "Outputs are bools and are True where there is a full regex match and False otherwise."},
      });
}

LightOpSchema MakeStringNormalizerSchema(int since_version) {
  return LightOpSchema("StringNormalizer", kOnnxDomain, since_version,
                       MakeStringNormalizerDoc(since_version),
                       {
                           {"X", "UTF-8 strings to normalize", "tensor(string)"},
                       },
                       {
                           {"Y", "UTF-8 Normalized strings", "tensor(string)"},
                       },
                       {});
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpTextSchemasWithHistory(const std::string &op_type,
                                                              bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"RegexFullMatch", [] { return std::vector<LightOpSchema>{MakeRegexFullMatchSchema(20)}; }},
      {"StringConcat", [] { return std::vector<LightOpSchema>{MakeStringConcatSchema(20)}; }},
      {"StringNormalizer",
       [] { return std::vector<LightOpSchema>{MakeStringNormalizerSchema(10)}; }},
      {"StringSplit", [] { return std::vector<LightOpSchema>{MakeStringSplitSchema(20)}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace text
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
