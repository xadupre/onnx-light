// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

std::string MakeCastDoc(int since_version) {
  if (since_version == 1 || since_version == 6) {
    return R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
)DOC";
  }
  return R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
)DOC";
}

std::string MakeCastInputTypeConstraintDescription(int since_version) {
  if (since_version == 1 || since_version == 6) {
    return "Constrain input types. Casting from strings and complex are not supported.";
  }
  return "Constrain input types. Casting from complex is not supported.";
}

std::string MakeCastOutputTypeConstraintDescription(int since_version) {
  if (since_version == 1 || since_version == 6) {
    return "Constrain output types. Casting to strings and complex are not supported.";
  }
  return "Constrain output types. Casting to complex is not supported.";
}

std::string MakeConcatDoc(int since_version) {
  if (since_version == 1) {
    return R"DOC(Concatenate a list of tensors into a single tensor)DOC";
  }
  if (since_version == 4) {
    return R"DOC(Concatenate a list of tensors into a single tensor)DOC";
  }
  return R"DOC(Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.)DOC";
}

std::string MakeConcatTypeConstraintDescription(int since_version) {
  if (since_version == 1) {
    return "Constrain output types to float tensors.";
  }
  return "Constrain output types to any tensor type.";
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
