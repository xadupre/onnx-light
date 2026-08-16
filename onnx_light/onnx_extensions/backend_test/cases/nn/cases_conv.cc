// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

using onnx_kernels::kernel::AutoPad;

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

void RegisterConvCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Conv conv{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    constexpr int64_t x_count = 1 * 32 * 128 * 128;
    constexpr int64_t w_count = 32 * 32 * 3 * 3;
    constexpr int64_t y_count = 1 * 32 * 126 * 126;
    Expect(registry, std::move(node), "test_cc_basic_conv_without_padding_benchmark", {opset},
           {x_count, w_count}, {y_count}, [conv]() -> IoData {
             Tensor X = RandnTensor(DataType::FLOAT, {1, 32, 128, 128}, 1001);
             Tensor W = RandnTensor(DataType::FLOAT, {32, 32, 3, 3}, 1002);
             Tensor B;
             onnx_kernels::kernel::Conv::Attributes attrs;
             attrs.kernel_shape = {3, 3};
             Tensor Y = conv(X, W, B, attrs);
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W)}, {std::move(Y)}};
           });
    return;
  }

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
    onnx_kernels::kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    Expect(registry, std::move(node), "test_cc_basic_conv_without_padding", {opset},
           [=]() -> IoData { return IoData{{std::move(X), std::move(W)}, {std::move(Y)}}; });
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
    onnx_kernels::kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.pads = {1, 1, 1, 1};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    Expect(registry, std::move(node), "test_cc_basic_conv_with_padding", {opset},
           [=]() -> IoData { return IoData{{std::move(X), std::move(W)}, {std::move(Y)}}; });
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
    onnx_kernels::kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.pads = {1, 1, 1, 1};
    attrs.strides = {2, 2};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_conv_with_strides_padding", {opset},
           [=]() -> IoData { return IoData{{std::move(X), std::move(W)}, {std::move(Y)}}; });
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
    onnx_kernels::kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.auto_pad = AutoPad::kSameUpper;
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W", "B"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::string>(node, "auto_pad", "SAME_UPPER");
    Expect(registry, std::move(node), "test_cc_conv_with_autopad_same", {opset}, [=]() -> IoData {
      return IoData{{std::move(X), std::move(W), std::move(B)}, {std::move(Y)}};
    });
  }

  // -------------------------------------------------------------------
  // Case 5: strides=2, no padding (mirrors
  // ``test_conv_with_strides_no_padding``). 5x7 input, 3x3 kernel.
  {
    std::vector<float> Xv(35);
    for (int i = 0; i < 35; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 7, 5}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor B;
    onnx_kernels::kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.strides = {2, 2};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_conv_with_strides_no_padding", {opset},
           [=]() -> IoData { return IoData{{std::move(X), std::move(W)}, {std::move(Y)}}; });
  }

  // -------------------------------------------------------------------
  // Case 6: strides=2 and asymmetric padding only along the H dimension
  // (mirrors ``test_conv_with_strides_and_asymmetric_padding``).
  {
    std::vector<float> Xv(35);
    for (int i = 0; i < 35; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 7, 5}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor B;
    onnx_kernels::kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.pads = {1, 0, 1, 0};
    attrs.strides = {2, 2};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 0, 1, 0});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_conv_with_strides_and_asymmetric_padding", {opset},
           [=]() -> IoData { return IoData{{std::move(X), std::move(W)}, {std::move(Y)}}; });
  }

  // -------------------------------------------------------------------
  // Case 7: SAME_UPPER auto_pad with stride=2 and asymmetric padding.
  // Reproduces the scenario from microsoft/onnxruntime#26734 where stride>1
  // with SAME_UPPER produced incorrect values in some ORT backends.
  // Input [1,1,4,4], kernel 3x3, stride=[2,2] → pads resolved to [0,0,1,1]
  // (pad_begin=0, pad_end=1 on each spatial axis), output shape [1,1,2,2].
  {
    std::vector<float> Xv(16);
    for (int i = 0; i < 16; ++i) {
      Xv[i] = static_cast<float>(i);
    }
    Tensor X = Tensor::FromFloat("X", {1, 1, 4, 4}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor B;
    onnx_kernels::kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.auto_pad = AutoPad::kSameUpper;
    attrs.strides = {2, 2};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::string>(node, "auto_pad", "SAME_UPPER");
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_conv_with_autopad_same_stride2", {opset},
           [=]() -> IoData { return IoData{{std::move(X), std::move(W)}, {std::move(Y)}}; });
  }

  // -------------------------------------------------------------------
  // Case 8: FLOAT16 inputs with bias. Exercises the half-precision dispatch
  // (promote to float32, compute, demote) that the expanded
  // ``CausalConvWithState`` function relies on.
  {
    std::vector<float> Xv(16);
    for (int i = 0; i < 16; ++i) {
      Xv[i] = static_cast<float>(i) * 0.5f;
    }
    Tensor X = MakeFloat16Tensor("X", {1, 1, 4, 4}, Xv);
    Tensor W = MakeFloat16Tensor("W", {1, 1, 3, 3}, std::vector<float>(9, 0.25f));
    Tensor B = MakeFloat16Tensor("B", {1}, {0.5f});
    onnx_kernels::kernel::Conv::Attributes attrs;
    attrs.kernel_shape = {3, 3};
    attrs.pads = {1, 1, 1, 1};
    Tensor Y = conv(X, W, B, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvNode({"X", "W", "B"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    Expect(registry, std::move(node), "test_cc_conv_fp16", {opset}, [=]() -> IoData {
      return IoData{{std::move(X), std::move(W), std::move(B)}, {std::move(Y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
