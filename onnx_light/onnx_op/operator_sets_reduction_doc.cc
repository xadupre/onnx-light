// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_reduction_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace reduction {

std::string MakeReduceSumDoc(int since_version) {
  if (since_version <= 1) {
    return R"DOC(Computes the sum of the input tensor's elements along the provided axes.
The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
The axes attribute specifies which dimensions to reduce. If axes is not provided, all dimensions are reduced.)DOC";
  }

  if (since_version <= 11) {
    return R"DOC(Computes the sum of the input tensor's elements along the provided axes.
The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
The axes attribute specifies which dimensions to reduce. Negative axes are supported.
If axes is not provided, all dimensions are reduced.
If noop_with_empty_axes is set and axes is empty, the input tensor is returned unchanged.)DOC";
  }

  return R"DOC(Computes the sum of the input tensor's elements along the provided axes.
The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
The axes are provided as an optional second input tensor. Negative axes are supported.
If axes is not provided or is empty and noop_with_empty_axes is set to 1, the input tensor is returned unchanged.
If axes is not provided or is empty and noop_with_empty_axes is not set, all dimensions are reduced.)DOC";
}

} // namespace reduction
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
