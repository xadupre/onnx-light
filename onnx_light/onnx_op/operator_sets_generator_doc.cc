// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_generator_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace generator {

std::string MakeConstantDoc(int since_version) {
  if (since_version == 1 || since_version == 9) {
    return "A constant tensor.";
  }
  if (since_version == 11) {
    return R"DOC(
A constant tensor. Exactly one of the two attributes, either value or sparse_value,
must be specified.
)DOC";
  }
  return R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC";
}

std::string MakeConstantOfShapeDoc(int /*since_version*/) {
  return R"DOC(
Generate a tensor with given value and shape.
)DOC";
}

} // namespace generator
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
