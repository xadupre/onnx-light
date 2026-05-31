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

NodeProto MakeNonZeroNode() {
  NodeProto node;
  node.set_op_type("NonZero");
  node.add_input("X");
  node.add_output("Y");
  return node;
}

} // namespace

void RegisterNonZeroCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::NonZero nonzero_kernel{ctx};

  // test_cc_nonzero_2d
  {
    const Tensor x = Tensor::FromFloat("X", {2, 2}, {1.0f, 0.0f, 1.0f, 1.0f});
    const Tensor y = nonzero_kernel(x);
    Expect(MakeNonZeroNode(), {x}, {y}, "test_cc_nonzero_2d", {opset}, "backend-test", registry);
  }

  // test_cc_nonzero_1d
  {
    const Tensor x = Tensor::FromFloat("X", {5}, {0.0f, 1.0f, 0.0f, -1.0f, 2.0f});
    const Tensor y = nonzero_kernel(x);
    Expect(MakeNonZeroNode(), {x}, {y}, "test_cc_nonzero_1d", {opset}, "backend-test", registry);
  }

  // test_cc_nonzero_bool
  {
    const Tensor x = Tensor::FromBool("X", {2, 3}, {1, 0, 1, 0, 1, 0});
    const Tensor y = nonzero_kernel(x);
    Expect(MakeNonZeroNode(), {x}, {y}, "test_cc_nonzero_bool", {opset}, "backend-test", registry);
  }

  // test_cc_nonzero_int64
  {
    const Tensor x = Tensor::FromInt64("X", {2, 3}, {0, 1, 2, 0, 0, 3});
    const Tensor y = nonzero_kernel(x);
    Expect(MakeNonZeroNode(), {x}, {y}, "test_cc_nonzero_int64", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
