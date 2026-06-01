// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeConvTransposeNode(const std::vector<std::string> &inputs,
                                const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("ConvTranspose");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

void RegisterConvTransposeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::ConvTranspose ct{ctx};

  // -------------------------------------------------------------------
  // Case 1: basic 3x3 ConvTranspose (mirrors ``test_convtranspose``).
  {
    std::vector<float> Xv(9);
    for (int i = 0; i < 9; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 2, 3, 3}, std::vector<float>(18, 1.0f));
    Tensor B;
    kernel::ConvTranspose::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    Expect(node, {X, W}, {Y}, "test_cc_convtranspose", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 2: ConvTranspose with explicit pads (mirrors ``test_convtranspose_pads``).
  {
    std::vector<float> Xv(9);
    for (int i = 0; i < 9; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 2, 3, 3}, std::vector<float>(18, 1.0f));
    Tensor B;
    kernel::ConvTranspose::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.pads = {1, 2, 1, 2};
    attrs.strides = {3, 2};
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 2, 1, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {3, 2});
    Expect(node, {X, W}, {Y}, "test_cc_convtranspose_pads", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 3: ConvTranspose with explicit kernel_shape, bias, dilations.
  {
    std::vector<float> Xv(9);
    for (int i = 0; i < 9; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor B = Tensor::FromFloat("B", {1}, {0.5f});
    kernel::ConvTranspose::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W", "B"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    Expect(node, {X, W, B}, {Y}, "test_cc_convtranspose_with_kernel", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
