// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeIdentityNode() {
  NodeProto node;
  node.set_op_type("Identity");
  node.add_input("x");
  node.add_output("y");
  return node;
}

// Returns a copy of ``t`` with a new name; lets us rename kernel outputs to
// match the ``y`` output name in :func:`MakeIdentityNode`.
Tensor Rename(Tensor t, const std::string &name) {
  t.name = name;
  return t;
}

} // namespace

void RegisterIdentityCases(std::vector<TestCase> &registry) {
  // Identity has been available since opset 1; opset 14 broadened the
  // ``V`` type constraint to also cover sequence types and opset 16 added
  // optional types. The default opset chosen here matches the most common
  // tensor-only usage exercised by these reference cases.
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Identity identity_kernel{ctx};

  // test_cc_identity — 4-D float tensor with non-trivial shape; mirrors the
  // ONNX upstream ``test_identity`` case shape (1, 3, 2, 2).
  {
    const Tensor x = Tensor::FromFloat(
        "x", {1, 3, 2, 2},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    const Tensor y = Rename(identity_kernel(x), "y");
    Expect(MakeIdentityNode(), {x}, {y}, "test_cc_identity", {opset}, "backend-test", registry);
  }

  // test_cc_identity_scalar — 0-D input is propagated verbatim.
  {
    const Tensor x = Tensor::FromFloat("x", {}, {42.0f});
    const Tensor y = Rename(identity_kernel(x), "y");
    Expect(MakeIdentityNode(), {x}, {y}, "test_cc_identity_scalar", {opset}, "backend-test",
           registry);
  }

  // test_cc_identity_int64 — integer dtype is preserved.
  {
    const Tensor x = Tensor::FromInt64("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor y = Rename(identity_kernel(x), "y");
    Expect(MakeIdentityNode(), {x}, {y}, "test_cc_identity_int64", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
