// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <optional>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

NodeProto MakeCompressNode(std::optional<int64_t> axis) {
  NodeProto node;
  node.set_op_type("Compress");
  node.add_input("input");
  node.add_input("condition");
  node.add_output("output");
  if (axis.has_value()) {
    AddAttribute<int64_t>(node, "axis", *axis);
  }
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// Compress — backend test cases on tensors with empty shapes. Covers
// scenarios where either the input has a zero-sized dimension, the condition
// selects no element (producing an output with a zero-sized dimension), or
// both. ``kernel::Compress`` is used to compute the expected output so the
// case is self-consistent with the in-tree kernel.
// ---------------------------------------------------------------------------
void RegisterCompressEmptyShapeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};
  const kernel::Compress compress_kernel{ctx};

  // test_cc_compress_empty_shape_no_axis_all_false — flatten mode, condition
  // selects nothing; output is a 1-D empty tensor with shape {0}.
  {
    Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor condition = Tensor::FromBool("condition", {6}, {0, 0, 0, 0, 0, 0});
    Tensor output = compress_kernel(input, condition, std::nullopt);
    Expect(MakeCompressNode(std::nullopt), {input, condition}, {output},
           "test_cc_compress_empty_shape_no_axis_all_false", {opset}, "backend-test", registry,
           "empty_shape");
  }

  // test_cc_compress_empty_shape_axis0_all_false — axis mode, condition
  // selects no row; output has shape {0, 2}.
  {
    Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor condition = Tensor::FromBool("condition", {3}, {0, 0, 0});
    Tensor output = compress_kernel(input, condition, 0);
    Expect(MakeCompressNode(0), {input, condition}, {output},
           "test_cc_compress_empty_shape_axis0_all_false", {opset}, "backend-test", registry,
           "empty_shape");
  }

  // test_cc_compress_empty_shape_input_zero_dim — input itself already has a
  // zero-sized axis; result must also be a zero-element tensor with shape
  // {0, 2}.
  {
    Tensor input = Tensor::FromFloat("input", {0, 2}, {});
    Tensor condition = Tensor::FromBool("condition", {0}, {});
    Tensor output = compress_kernel(input, condition, 0);
    Expect(MakeCompressNode(0), {input, condition}, {output},
           "test_cc_compress_empty_shape_input_zero_dim", {opset}, "backend-test", registry,
           "empty_shape");
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
