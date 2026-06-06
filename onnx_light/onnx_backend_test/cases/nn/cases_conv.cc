// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeConvNode(const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("Conv");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

void RegisterConvCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Conv conv{ctx};

  // -------------------------------------------------------------------
  // Case 1: basic 3x3 kernel without padding
  // (mirrors ``test_basic_conv_without_padding``).
  {
    std::vector<float> Xv(25);
    for (int i = 0; i < 25; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 5, 5}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor B;
    kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    Expect(node, {X, W}, {Y}, "test_cc_basic_conv_without_padding", {opset}, "backend-test",
           registry);
  }

  // -------------------------------------------------------------------
  // Case 2: 3x3 kernel with padding=1 (mirrors ``test_basic_conv_with_padding``).
  {
    std::vector<float> Xv(25);
    for (int i = 0; i < 25; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 5, 5}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor B;
    kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.pads = {1, 1, 1, 1};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    Expect(node, {X, W}, {Y}, "test_cc_basic_conv_with_padding", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 3: 3x3 kernel with strides=2 and explicit pads
  // (mirrors ``test_conv_with_strides_padding``).
  {
    std::vector<float> Xv(35);
    for (int i = 0; i < 35; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 7, 5}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor B;
    kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.pads = {1, 1, 1, 1};
    attrs.strides = {2, 2};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(node, {X, W}, {Y}, "test_cc_conv_with_strides_padding", {opset}, "backend-test",
           registry);
  }

  // -------------------------------------------------------------------
  // Case 4: SAME_UPPER auto_pad with bias.
  {
    std::vector<float> Xv(16);
    for (int i = 0; i < 16; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 4, 4}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor B = Tensor::FromFloat("B", {1}, {0.5f});
    kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.auto_pad = "SAME_UPPER";
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W", "B"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::string>(node, "auto_pad", "SAME_UPPER");
    Expect(node, {X, W, B}, {Y}, "test_cc_conv_with_autopad_same", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
