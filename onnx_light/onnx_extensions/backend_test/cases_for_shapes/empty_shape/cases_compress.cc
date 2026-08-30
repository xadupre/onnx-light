// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <optional>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
void RegisterCompressEmptyShapeCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(11);
  const auto compress_kernel = MakeReferenceKernel<onnx_kernels::kernel::Compress>(opset);

  // test_cc_compress_empty_shape_no_axis_all_false — flatten mode, condition
  // selects nothing; output is a 1-D empty tensor with shape {0}.
  {
    Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor condition = Tensor::FromBool("condition", {6}, {0, 0, 0, 0, 0, 0});
    Expect(
        registry, MakeCompressNode(std::nullopt), "test_cc_compress_empty_shape_no_axis_all_false",
        {opset},
        [=]() -> IoData {
          Tensor output = compress_kernel.Invoke(
              [&](const auto &kernel) { return kernel(input, condition, std::nullopt); });
          return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
        },
        "backend-test", TestCaseTag::EMPTY_SHAPE);
  }

  // test_cc_compress_empty_shape_axis0_all_false — axis mode, condition
  // selects no row; output has shape {0, 2}.
  {
    Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor condition = Tensor::FromBool("condition", {3}, {0, 0, 0});
    Expect(
        registry, MakeCompressNode(0), "test_cc_compress_empty_shape_axis0_all_false", {opset},
        [=]() -> IoData {
          Tensor output = compress_kernel.Invoke(
              [&](const auto &kernel) { return kernel(input, condition, 0); });
          return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
        },
        "backend-test", TestCaseTag::EMPTY_SHAPE);
  }

  // test_cc_compress_empty_shape_input_zero_dim — input itself already has a
  // zero-sized axis; result must also be a zero-element tensor with shape
  // {0, 2}.
  {
    Tensor input = Tensor::FromFloat("input", {0, 2}, {});
    Tensor condition = Tensor::FromBool("condition", {0}, {});
    Expect(
        registry, MakeCompressNode(0), "test_cc_compress_empty_shape_input_zero_dim", {opset},
        [=]() -> IoData {
          Tensor output = compress_kernel.Invoke(
              [&](const auto &kernel) { return kernel(input, condition, 0); });
          return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
        },
        "backend-test", TestCaseTag::EMPTY_SHAPE);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
