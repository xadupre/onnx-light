// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// BlackmanWindow — generates a Blackman window of length ``size`` (since opset
// 17). Registers both the default periodic case and the symmetric variant
// (``periodic = 0``).
// ---------------------------------------------------------------------------
void RegisterBlackmanWindowCases(std::vector<TestCase> &registry) {
  constexpr int32_t kSize = 10;
  const OpsetId opset = DefaultOpset(17);
  const kernel::KernelContext ctx{opset};
  const kernel::BlackmanWindow blackman_kernel{ctx};

  // Default periodic variant.
  {
    NodeProto node;
    node.set_op_type("BlackmanWindow");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromInt32("", {}, {kSize});
    Tensor y = blackman_kernel(x, /*periodic=*/true);

    Expect(node, {x}, {y}, "test_cc_blackmanwindow", {opset}, "backend-test", registry);
  }

  // Symmetric variant (periodic = 0).
  {
    NodeProto node;
    node.set_op_type("BlackmanWindow");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("periodic");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(0);

    Tensor x = Tensor::FromInt32("", {}, {kSize});
    Tensor y = blackman_kernel(x, /*periodic=*/false);

    Expect(node, {x}, {y}, "test_cc_blackmanwindow_symmetric", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
