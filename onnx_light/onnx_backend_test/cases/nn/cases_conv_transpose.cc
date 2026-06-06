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

  // -------------------------------------------------------------------
  // Case 4: 1D ConvTranspose (mirrors ``test_convtranspose_1d``).
  {
    Tensor X = Tensor::FromFloat("X", {1, 1, 3}, {0.0f, 1.0f, 2.0f});
    Tensor W = Tensor::FromFloat("W", {1, 2, 3}, std::vector<float>(6, 1.0f));
    Tensor B;
    kernel::ConvTranspose::Attributes attrs;
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    Expect(node, {X, W}, {Y}, "test_cc_convtranspose_1d", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 5: 3D ConvTranspose (mirrors ``test_convtranspose_3d``).
  {
    std::vector<float> Xv(60);
    for (int i = 0; i < 60; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 4, 5}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 2, 3, 3, 3}, std::vector<float>(54, 1.0f));
    Tensor B;
    kernel::ConvTranspose::Attributes attrs;
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    Expect(node, {X, W}, {Y}, "test_cc_convtranspose_3d", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 6: ConvTranspose with auto_pad=SAME_UPPER and strides
  // (mirrors ``test_convtranspose_autopad_same``).
  {
    std::vector<float> Xv(9);
    for (int i = 0; i < 9; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 2, 3, 3}, std::vector<float>(18, 1.0f));
    Tensor B;
    kernel::ConvTranspose::Attributes attrs;
    attrs.auto_pad = "SAME_UPPER";
    attrs.strides = {2, 2};
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::string>(node, "auto_pad", "SAME_UPPER");
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(node, {X, W}, {Y}, "test_cc_convtranspose_autopad_same", {opset}, "backend-test",
           registry);
  }

  // -------------------------------------------------------------------
  // Case 7: ConvTranspose with dilations (mirrors ``test_convtranspose_dilations``).
  {
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3},
                                 {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f});
    Tensor W = Tensor::FromFloat("W", {1, 1, 2, 2}, {7.0f, 2.0f, 1.0f, 9.0f});
    Tensor B;
    kernel::ConvTranspose::Attributes attrs;
    attrs.dilations = {2, 2};
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2});
    Expect(node, {X, W}, {Y}, "test_cc_convtranspose_dilations", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 8: ConvTranspose with group=2 (mirrors ``test_convtranspose_group_2``).
  {
    std::vector<float> Xv(18);
    for (int i = 0; i < 18; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 2, 3, 3}, Xv);
    Tensor W = Tensor::FromFloat("W", {2, 1, 3, 3}, std::vector<float>(18, 1.0f));
    Tensor B;
    kernel::ConvTranspose::Attributes attrs;
    attrs.group = 2;
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<int64_t>(node, "group", 2);
    Expect(node, {X, W}, {Y}, "test_cc_convtranspose_group_2", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 9: ConvTranspose with group=2 and batch=3
  // (mirrors ``test_convtranspose_group_2_image_3``).
  {
    Tensor X = Tensor::FromFloat("X", {3, 2, 3, 3},
                                 {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                                  18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f,
                                  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                                  0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f});
    Tensor W = Tensor::FromFloat("W", {2, 1, 3, 3}, std::vector<float>(18, 1.0f));
    Tensor B;
    kernel::ConvTranspose::Attributes attrs;
    attrs.group = 2;
    Tensor Y = ct(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<int64_t>(node, "group", 2);
    Expect(node, {X, W}, {Y}, "test_cc_convtranspose_group_2_image_3", {opset}, "backend-test",
           registry);
  }

  // -------------------------------------------------------------------
  // Cases 10/11: ConvTranspose with output_shape and with kernel_shape +
  // output_padding (mirrors ``test_convtranspose_output_shape`` and
  // ``test_convtranspose_kernel_shape``).
  {
    std::vector<float> Xv(9);
    for (int i = 0; i < 9; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 2, 3, 3}, std::vector<float>(18, 1.0f));
    Tensor B;

    {
      kernel::ConvTranspose::Attributes attrs;
      attrs.strides = {3, 2};
      attrs.output_shape = {10, 8};
      Tensor Y = ct(X, W, B, attrs);
      Y.name = "Y";
      NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
      AddAttribute<std::vector<int64_t>>(node, "strides", {3, 2});
      AddAttribute<std::vector<int64_t>>(node, "output_shape", {10, 8});
      Expect(node, {X, W}, {Y}, "test_cc_convtranspose_output_shape", {opset}, "backend-test",
             registry);
    }

    {
      kernel::ConvTranspose::Attributes attrs;
      attrs.strides = {3, 2};
      attrs.output_shape = {10, 8};
      attrs.kernel_shape = {3, 3};
      attrs.output_padding = {1, 1};
      Tensor Y = ct(X, W, B, attrs);
      Y.name = "Y";
      NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
      AddAttribute<std::vector<int64_t>>(node, "strides", {3, 2});
      AddAttribute<std::vector<int64_t>>(node, "output_shape", {10, 8});
      AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
      AddAttribute<std::vector<int64_t>>(node, "output_padding", {1, 1});
      Expect(node, {X, W}, {Y}, "test_cc_convtranspose_kernel_shape", {opset}, "backend-test",
             registry);
    }
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
