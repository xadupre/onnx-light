// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Add — backend test cases on tensors with empty shapes. Covers both:
//   * rank-0 (``shape == {}``) tensors, and
//   * zero-element tensors with a 0-sized dimension (``shape == {0}`` or
//     ``shape == {0, 3}``).
// Expected outputs are computed by ``kernel::Add`` so any future kernel
// regression is caught at runtime.
// ---------------------------------------------------------------------------
void RegisterAddEmptyShapeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Add add_kernel{ctx};

  // test_cc_add_empty_shape_scalars — rank-0 + rank-0.
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {}, {2.5f});
    Tensor y = Tensor::FromFloat("", {}, {3.5f});
    Tensor z = add_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_add_empty_shape_scalars", {opset}, "backend-test", registry);
  }

  // test_cc_add_empty_shape_zero_dim — zero-element 1-D tensors.
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {0}, {});
    Tensor y = Tensor::FromFloat("", {0}, {});
    Tensor z = add_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_add_empty_shape_zero_dim", {opset}, "backend-test",
           registry);
  }

  // test_cc_add_empty_shape_zero_dim_2d — zero-element 2-D tensors.
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {0, 3}, {});
    Tensor y = Tensor::FromFloat("", {0, 3}, {});
    Tensor z = add_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_add_empty_shape_zero_dim_2d", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
