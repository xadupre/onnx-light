// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

const char *MakeCastDoc() {
  return "Casts the elements of an input tensor to a specified data type.";
}

const char *MakeCastInputDescription() { return "Input tensor to be cast."; }

const char *MakeCastOutputDescription() {
  return "Output tensor with the same shape as input with type specified by the 'to' argument";
}

const char *MakeCastLegacyInputConstraintDescription() {
  return "Constrain input types. Casting from strings and complex are not supported.";
}

const char *MakeCastLegacyOutputConstraintDescription() {
  return "Constrain output types. Casting to strings and complex are not supported.";
}

const char *MakeCastInputConstraintDescription() {
  return "Constrain input types. Casting from complex is not supported.";
}

const char *MakeCastOutputConstraintDescription() {
  return "Constrain output types. Casting to complex is not supported.";
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
