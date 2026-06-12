// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Sign — y = sign(x) (since opset 9, widened at opset 13).
// Registers a small deterministic ``test_cc_sign`` case and the upstream
// ONNX backend test case ``test_sign``.
// ---------------------------------------------------------------------------
void RegisterSignCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Sign sign_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.5f, -1.0f, 0.0f, 0.5f, 1.0f, 3.25f});
    Tensor y = sign_kernel(x);
    Expect(node, {x}, {y}, "test_cc_sign", {opset}, "backend-test", registry);
  }

  // From Sign.export(): ``test_sign`` uses x = np.array(range(-5, 6)).
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat(
        "", {11}, {-5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor y = sign_kernel(x);
    Expect(node, {x}, {y}, "test_sign", {opset}, "backend-test", registry);
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");

    Tensor x = kernel::MakeFloat16Tensor("", {2, 3}, {-3.0f, -0.5f, 0.0f, 0.5f, 2.0f, -1.0f});
    Tensor y = sign_kernel(x);
    Expect(node, {x}, {y}, "test_cc_sign_float16", {opset}, "backend-test", registry);
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");

    std::vector<float> vals = {-3.0f, -0.5f, 0.0f, 0.5f, 2.0f, -1.0f};
    std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
    auto *dst = reinterpret_cast<uint16_t *>(raw.data());
    for (size_t i = 0; i < vals.size(); ++i)
      dst[i] = kernel::FloatToBfloat16Bits(vals[i]);
    Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
    Tensor y = sign_kernel(x);
    Expect(node, {x}, {y}, "test_cc_sign_bfloat16", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
