// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

using onnx_kernels::kernel::AutoPad;

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

void RegisterConvTransposeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const auto ct = MakeReferenceKernel<onnx_kernels::kernel::ConvTranspose>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    constexpr int64_t x_count = 1 * 32 * 128 * 128;
    constexpr int64_t w_count = 32 * 32 * 3 * 3;
    constexpr int64_t y_count = 1 * 32 * 130 * 130;
    Expect(registry, std::move(node), "test_cc_convtranspose_benchmark", {opset},
           {x_count, w_count}, {y_count}, [ct]() -> IoData {
             Tensor X = RandnTensor(DataType::FLOAT, {1, 32, 128, 128}, 1401);
             Tensor W = RandnTensor(DataType::FLOAT, {32, 32, 3, 3}, 1402);
             Tensor B;
             onnx_kernels::kernel::ConvTranspose::Attributes attrs;
             attrs.kernel_shape = {3, 3};
             Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
           });
    return;
  }

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
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    Expect(registry, std::move(node), "test_cc_convtranspose", {opset}, [=]() -> IoData {
      Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
      Y.name = "Y";
      return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
    });
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
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.pads = {1, 2, 1, 2};
    attrs.strides = {3, 2};
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 2, 1, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {3, 2});
    Expect(registry, std::move(node), "test_cc_convtranspose_pads", {opset}, [=]() -> IoData {
      Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
      Y.name = "Y";
      return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
    });
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
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    NodeProto node = MakeConvTransposeNode({"X", "W", "B"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    Expect(registry, std::move(node), "test_cc_convtranspose_with_kernel", {opset},
           [=]() -> IoData {
             Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W), std::move(B)}, {std::move(Y)}};
           });
  }

  // -------------------------------------------------------------------
  // Case 4: 1D ConvTranspose (mirrors ``test_convtranspose_1d``).
  {
    Tensor X = Tensor::FromFloat("X", {1, 1, 3}, {0.0f, 1.0f, 2.0f});
    Tensor W = Tensor::FromFloat("W", {1, 2, 3}, std::vector<float>(6, 1.0f));
    Tensor B;
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_convtranspose_1d", {opset}, [=]() -> IoData {
      Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
      Y.name = "Y";
      return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
    });
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
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_convtranspose_3d", {opset}, [=]() -> IoData {
      Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
      Y.name = "Y";
      return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
    });
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
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    attrs.auto_pad = AutoPad::kSameUpper;
    attrs.strides = {2, 2};
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::string>(node, "auto_pad", "SAME_UPPER");
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_convtranspose_autopad_same", {opset},
           [=]() -> IoData {
             Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
           });
  }

  // -------------------------------------------------------------------
  // Case 7: ConvTranspose with dilations (mirrors ``test_convtranspose_dilations``).
  {
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3},
                                 {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f});
    Tensor W = Tensor::FromFloat("W", {1, 1, 2, 2}, {7.0f, 2.0f, 1.0f, 9.0f});
    Tensor B;
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    attrs.dilations = {2, 2};
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2});
    Expect(registry, std::move(node), "test_cc_convtranspose_dilations", {opset}, [=]() -> IoData {
      Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
      Y.name = "Y";
      return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
    });
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
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    attrs.group = 2;
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<int64_t>(node, "group", 2);
    Expect(registry, std::move(node), "test_cc_convtranspose_group_2", {opset}, [=]() -> IoData {
      Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
      Y.name = "Y";
      return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
    });
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
    onnx_kernels::kernel::ConvTranspose::Attributes attrs;
    attrs.group = 2;
    NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
    AddAttribute<int64_t>(node, "group", 2);
    Expect(registry, std::move(node), "test_cc_convtranspose_group_2_image_3", {opset},
           [=]() -> IoData {
             Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
           });
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
      onnx_kernels::kernel::ConvTranspose::Attributes attrs;
      attrs.strides = {3, 2};
      attrs.output_shape = {10, 8};
      NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
      AddAttribute<std::vector<int64_t>>(node, "strides", {3, 2});
      AddAttribute<std::vector<int64_t>>(node, "output_shape", {10, 8});
      Expect(registry, std::move(node), "test_cc_convtranspose_output_shape", {opset},
             [=]() -> IoData {
               Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
               Y.name = "Y";
               return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
             });
    }

    {
      onnx_kernels::kernel::ConvTranspose::Attributes attrs;
      attrs.strides = {3, 2};
      attrs.output_shape = {10, 8};
      attrs.kernel_shape = {3, 3};
      attrs.output_padding = {1, 1};
      NodeProto node = MakeConvTransposeNode({"X", "W"}, {"Y"});
      AddAttribute<std::vector<int64_t>>(node, "strides", {3, 2});
      AddAttribute<std::vector<int64_t>>(node, "output_shape", {10, 8});
      AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
      AddAttribute<std::vector<int64_t>>(node, "output_padding", {1, 1});
      Expect(registry, std::move(node), "test_cc_convtranspose_kernel_shape", {opset},
             [=]() -> IoData {
               Tensor Y = ct.Invoke([&](const auto &kernel) { return kernel(X, W, B, attrs); });
               Y.name = "Y";
               return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
             });
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
