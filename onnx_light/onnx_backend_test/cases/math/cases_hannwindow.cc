// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// HannWindow — generates a Hann window of length ``size`` (since opset 17).
// Registers both the default periodic case and the symmetric variant
// (``periodic = 0``).
// ---------------------------------------------------------------------------
void RegisterHannWindowCases(std::vector<TestCase> &registry) {
  constexpr int32_t kSize = 10;
  const OpsetId opset = DefaultOpset(17);
  const kernel::KernelContext ctx{opset};
  const kernel::HannWindow hann_kernel{ctx};

  // Default periodic variant.
  {
    NodeProto node;
    node.set_op_type("HannWindow");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromInt32("", {}, {kSize});
    Tensor y = hann_kernel(x, /*periodic=*/true);

    Expect(node, {x}, {y}, "test_cc_hannwindow", {opset}, "backend-test", registry);
  }

  // Symmetric variant (periodic = 0).
  {
    NodeProto node;
    node.set_op_type("HannWindow");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("periodic");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(0);

    Tensor x = Tensor::FromInt32("", {}, {kSize});
    Tensor y = hann_kernel(x, /*periodic=*/false);

    Expect(node, {x}, {y}, "test_cc_hannwindow_symmetric", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
