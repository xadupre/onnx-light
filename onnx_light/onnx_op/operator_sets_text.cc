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

LightOpSchema MakeTfIdfVectorizerSchema(int since_version) {
  return LightOpSchema(
      "TfIdfVectorizer", kOnnxDomain, since_version, MakeTfIdfVectorizerDoc(since_version),
      {
          {"X", "Input for n-gram extraction", "T"},
      },
      {
          {"Y", "Ngram results", "T1"},
      },
      {
          {"T",
           {TensorType::kString, TensorType::kInt32, TensorType::kInt64},
           "Input is ether string UTF-8 or int32/int64"},
          {"T1", {TensorType::kFloat}, "1-D tensor of floats"},
      },
      {
          AttributeParam{"max_gram_length",
                         "Maximum n-gram length. If this value is 3, 3-grams will be used to "
                         "generate the output.",
                         AttributeType::INT, /*required=*/true, std::monostate{}},
          AttributeParam{"min_gram_length",
                         "Minimum n-gram length. If this value is 2 and max_gram_length is 3, "
                         "output may contain counts of 2-grams and 3-grams.",
                         AttributeType::INT, /*required=*/true, std::monostate{}},
          AttributeParam{"max_skip_count",
                         "Maximum number of items (integers/strings) to be skipped when "
                         "constructing an n-gram from X. "
                         "If max_skip_count=1, min_gram_length=2, max_gram_length=3, this "
                         "operator may generate 2-grams with skip_count=0 and skip_count=1, "
                         "and 3-grams with skip_count=0 and skip_count=1",
                         AttributeType::INT, /*required=*/true, std::monostate{}},
          AttributeParam{"pool_strings",
                         "List of strings n-grams learned from the training set. Either this "
                         "or pool_int64s attributes must be present but not both. "
                         "It's an 1-D tensor starting with the collections of all 1-grams and "
                         "ending with the collections of n-grams. "
                         "The i-th element in pool stores the n-gram that should be mapped to "
                         "coordinate ngram_indexes[i] in the output vector.",
                         AttributeType::STRINGS, /*required=*/false, std::monostate{}},
          AttributeParam{"pool_int64s",
                         "List of int64 n-grams learned from the training set. Either this or "
                         "pool_strings attributes must be present but not both. "
                         "It's an 1-D tensor starting with the collections of all 1-grams and "
                         "ending with the collections of n-grams. "
                         "The i-th element in pool stores the n-gram that should be mapped to "
                         "coordinate ngram_indexes[i] in the output vector.",
                         AttributeType::INTS, /*required=*/false, std::monostate{}},
          AttributeParam{"ngram_counts",
                         "The starting indexes of 1-grams, 2-grams, and so on in pool. "
                         "It is useful when determining the boundary between two consecutive "
                         "collections of n-grams. "
                         "For example, if ngram_counts is [0, 17, 36], the first index "
                         "(zero-based) of 1-gram/2-gram/3-gram in pool are 0/17/36. This "
                         "format is essentially identical to CSR (or CSC) sparse matrix "
                         "format, and we choose to use this due to its popularity.",
                         AttributeType::INTS, /*required=*/true, std::monostate{}},
          AttributeParam{"ngram_indexes",
                         "list of int64s (type: AttributeProto::INTS). This list is parallel "
                         "to the specified 'pool_*' attribute. The i-th element in "
                         "ngram_indexes indicate the coordinate of the i-th n-gram in the "
                         "output tensor.",
                         AttributeType::INTS, /*required=*/true, std::monostate{}},
          AttributeParam{"weights",
                         "list of floats. This attribute stores the weight of each n-gram in "
                         "pool. The i-th element in weights is the weight of the i-th n-gram "
                         "in pool. Its length equals to the size of ngram_indexes. By "
                         "default, weights is an all-one tensor.This attribute is used when "
                         "mode is \"IDF\" or \"TFIDF\" to scale the associated word counts.",
                         AttributeType::FLOATS, /*required=*/false, std::monostate{}},
          AttributeParam{"mode",
                         "The weighting criteria. It can be one of \"TF\" (term frequency), "
                         "\"IDF\" (inverse document frequency), and \"TFIDF\" (the "
                         "combination of TF and IDF)",
                         AttributeType::STRING, /*required=*/true, std::monostate{}},
      });
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
      {"TfIdfVectorizer", [] { return std::vector<LightOpSchema>{MakeTfIdfVectorizerSchema(9)}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace text
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
