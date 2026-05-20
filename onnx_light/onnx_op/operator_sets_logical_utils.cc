// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical_utils.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {
namespace detail {

std::string BuildLogicalOperatorDoc(const char *name, int since_version) {
  if (since_version == 1) {
    std::string doc = "\nReturns the tensor resulted from performing the `";
    doc += name;
    doc += R"DOC(` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
    return doc;
  }

  std::string doc = "\nReturns the tensor resulted from performing the `";
  doc += name;
  doc += R"DOC(` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
)DOC";
  return doc;
}

} // namespace detail
} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
