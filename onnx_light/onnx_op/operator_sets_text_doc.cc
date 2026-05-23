// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_text_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace text {

std::string MakeStringConcatDoc(int since_version) {
  if (since_version == 20) {
    return R"DOC(StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support))DOC";
  }
  return "";
}

std::string MakeStringSplitDoc(int since_version) {
  if (since_version == 20) {
    return R"DOC(StringSplit splits a string tensor's elements into substrings based on a delimiter attribute and a maxsplit attribute.

The first output of this operator is a tensor of strings representing the substrings from splitting each input string on the `delimiter` substring. This tensor has one additional rank compared to the input tensor in order to store the substrings for each input element (where the input tensor is not empty). Note that, in order to ensure the same number of elements are present in the final dimension, this tensor will pad empty strings as illustrated in the examples below. Consecutive delimiters are not grouped together and are deemed to delimit empty strings, except if the `delimiter` is unspecified or is the empty string (""). In the case where the `delimiter` is unspecified or the empty string, consecutive whitespace characters are regarded as a single separator and leading or trailing whitespace is removed in the output.

The second output tensor represents the number of substrings generated. `maxsplit` can be used to limit the number of splits performed - after the `maxsplit`\ th split if the string is not fully split, the trailing suffix of input string after the final split point is also added. For elements where fewer splits are possible than specified in `maxsplit`, it has no effect.)DOC";
  }
  return "";
}

std::string MakeRegexFullMatchDoc(int since_version) {
  if (since_version == 20) {
    return R"DOC(RegexFullMatch performs a full regex match on each element of the input tensor. If an element fully matches the regex pattern specified as an attribute, the corresponding element in the output is True and it is False otherwise. [RE2](https://github.com/google/re2/wiki/Syntax) regex syntax is used.)DOC";
  }
  return "";
}

std::string MakeStringNormalizerDoc(int since_version) {
  if (since_version == 10) {
    return R"DOC(
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
)DOC";
  }
  return "";
}

} // namespace text
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
