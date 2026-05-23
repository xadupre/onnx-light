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

std::string MakeReduceOpDoc(const std::string &op_name, const std::string &empty_value,
                            int since_version) {
  if (since_version <= 1) {
    return "Computes the " + op_name +
           " of the input tensor's element along the provided axes.\n"
           "The resulting tensor has the same rank as the input if keepdims equals 1.\n"
           "If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.";
  }

  if (since_version <= 11) {
    return "Computes the " + op_name +
           " of the input tensor's element along the provided axes.\n"
           "The resulting tensor has the same rank as the input if keepdims equals 1.\n"
           "If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.\n"
           "Negative axes are supported in the axes attribute.";
  }

  if (since_version <= 13) {
    return "Computes the " + op_name +
           " of the input tensor's elements along the provided axes.\n"
           "The resulting tensor has the same rank as the input if keepdims equals 1.\n"
           "If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.\n"
           "The axes attribute specifies which dimensions to reduce. Negative axes are "
           "supported.\n"
           "If axes is not provided, all dimensions are reduced.\n"
           "Reduction over an empty set of values yields " +
           empty_value + ".";
  }

  // since_version >= 18 (axes moved to optional input, noop_with_empty_axes introduced)
  return "Computes the " + op_name +
         " of the input tensor's elements along the provided axes.\n"
         "The resulting tensor has the same rank as the input if keepdims equals 1.\n"
         "If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.\n"
         "The axes are provided as an optional second input tensor. Negative axes are "
         "supported.\n"
         "If axes is not provided or is empty and noop_with_empty_axes is set to 1, the input "
         "tensor is returned unchanged.\n"
         "If axes is not provided or is empty and noop_with_empty_axes is not set, all "
         "dimensions are reduced.\n"
         "Reduction over an empty set of values yields " +
         empty_value + ".";
}

std::string MakeArgReduceDoc(const std::string &op_name, int since_version) {
  if (since_version <= 1) {
    return "Computes the indices of the " + op_name +
           " elements of the input tensor's element along the provided axis.\n"
           "The resulting tensor has the same rank as the input if keepdims equals 1.\n"
           "If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.\n"
           "The type of the output tensor is integer.";
  }

  if (since_version <= 11) {
    return "Computes the indices of the " + op_name +
           " elements of the input tensor's element along the provided axis.\n"
           "The resulting tensor has the same rank as the input if keepdims equals 1.\n"
           "If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.\n"
           "The input tensor must not be empty.\n"
           "The type of the output tensor is integer.";
  }

  // since_version >= 12 introduced the select_last_index attribute.
  return "Computes the indices of the " + op_name +
         " elements of the input tensor's element along the provided axis.\n"
         "The resulting tensor has the same rank as the input if keepdims equals 1.\n"
         "If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.\n"
         "If select_last_index is True (default False), the index of the last occurrence of "
         "the " +
         op_name + " is selected if the " + op_name +
         " appears more than once in the input. "
         "Otherwise the index of the first occurrence is selected.\n"
         "The type of the output tensor is integer.";
}

} // namespace reduction
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
