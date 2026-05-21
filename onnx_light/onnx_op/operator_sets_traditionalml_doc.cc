// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_traditionalml_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace traditionalml {

std::string MakeLabelEncoderDoc() {
  return R"DOC(
Maps each element in the input tensor to another value.<br>
The mapping is determined by the two parallel attributes, 'keys_*' and
'values_*' attribute. The i-th value in the specified 'keys_*' attribute
would be mapped to the i-th value in the specified 'values_*' attribute. It
implies that input's element type and the element type of the specified
'keys_*' should be identical while the output type is identical to the
specified 'values_*' attribute. Note that the 'keys_*' and 'values_*' attributes
must have the same length. If an input element can not be found in the
specified 'keys_*' attribute, the 'default_*' that matches the specified
'values_*' attribute may be used as its output value. The type of the 'default_*'
attribute must match the 'values_*' attribute chosen. <br>
Let's consider an example which maps a string tensor to an integer tensor.
Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
"Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
Since this operator is an one-to-one mapping, its input and output shapes
are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
Float keys with value 'NaN' match any input 'NaN' value regardless of bit
value. If a key is repeated, the last key takes precedence.
)DOC";
}

std::string MakeZipMapDoc() {
  return R"DOC(
Creates a map from the input and the attributes.<br>
The values are provided by the input tensor, while the keys are specified by the attributes.
Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
)DOC";
}

} // namespace traditionalml
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
