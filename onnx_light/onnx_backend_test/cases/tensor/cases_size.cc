// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeSizeNode() {
  NodeProto node;
  node.set_op_type("Size");
  node.add_input("x");
  node.add_output("y");
  return node;
}

// Renames a tensor (copy with a new ``name``) — used so kernel-produced
// expected outputs match the ``y`` output name in :func:`MakeSizeNode`.
Tensor Rename(Tensor t, const std::string &name) {
  t.name = name;
  return t;
}

} // namespace

void RegisterSizeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Size size_kernel{ctx};

  // test_cc_size_example — mirrors upstream ``test_size_example`` (2-D
  // float input of shape [2, 3]).
  {
    const Tensor x = Tensor::FromFloat("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Tensor y = Rename(size_kernel(x), "y");
    Expect(MakeSizeNode(), {x}, {y}, "test_cc_size_example", {opset}, "backend-test", registry);
  }

  // test_cc_size — mirrors upstream ``test_size`` (3-D float input of shape
  // [3, 4, 5]). Only the shape (and hence the element count) is observed by
  // the op.
  {
    const Tensor x = Tensor::FromFloat("x", {3, 4, 5}, std::vector<float>(3 * 4 * 5, 0.0f));
    const Tensor y = Rename(size_kernel(x), "y");
    Expect(MakeSizeNode(), {x}, {y}, "test_cc_size", {opset}, "backend-test", registry);
  }

  // test_cc_size_scalar — 0-D (scalar) input has exactly one element.
  {
    const Tensor x = Tensor::FromFloat("x", {}, {42.0f});
    const Tensor y = Rename(size_kernel(x), "y");
    Expect(MakeSizeNode(), {x}, {y}, "test_cc_size_scalar", {opset}, "backend-test", registry);
  }

  // test_cc_size_empty — input with a zero dimension has zero elements.
  {
    const Tensor x = Tensor::FromFloat("x", {2, 0, 3}, {});
    const Tensor y = Rename(size_kernel(x), "y");
    Expect(MakeSizeNode(), {x}, {y}, "test_cc_size_empty", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
