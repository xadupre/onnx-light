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

namespace {

NodeProto MakeConvIntegerNode(const std::vector<std::string> &inputs,
                              const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("ConvInteger");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

void RegisterConvIntegerCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(10);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::ConvInteger ci{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeConvIntegerNode({"X", "W", "x_zero_point"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    constexpr int64_t x_count = 1 * 32 * 128 * 128;
    constexpr int64_t w_count = 32 * 32 * 2 * 2;
    constexpr int64_t y_count = 1 * 32 * 127 * 127;
    Expect(registry, std::move(node), "test_cc_basic_convinteger_benchmark", {opset},
           {x_count, w_count, 1}, {y_count}, [ci]() -> IoData {
             Tensor X = Tensor::FromUint8("X", {1, 32, 128, 128},
                                          RandUint<uint8_t>(256, {1, 32, 128, 128}, 1301));
             Tensor W =
                 Tensor::FromUint8("W", {32, 32, 2, 2}, RandUint<uint8_t>(8, {32, 32, 2, 2}, 1302));
             Tensor xzp = Tensor::FromUint8("x_zero_point", {}, {1});
             Tensor wzp;
             onnx_kernels::kernel::ConvInteger::Attributes attrs;
             attrs.kernel_shape = {2, 2};
             Tensor Y = ci(X, W, xzp, wzp, attrs);
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W), std::move(xzp)}, {std::move(Y)}};
           });
    return;
  }

  // -------------------------------------------------------------------
  // Case 1: basic 2x2 ConvInteger without padding, with x_zero_point.
  {
    Tensor X = Tensor::FromUint8("X", {1, 1, 3, 3}, {2, 3, 4, 5, 6, 7, 8, 9, 10});
    Tensor W = Tensor::FromUint8("W", {1, 1, 2, 2}, {1, 1, 1, 1});
    Tensor xzp = Tensor::FromUint8("x_zero_point", {}, {1});
    Tensor wzp;
    onnx_kernels::kernel::ConvInteger::Attributes attrs;
    attrs.kernel_shape = {2, 2};
    Tensor Y = ci(X, W, xzp, wzp, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvIntegerNode({"X", "W", "x_zero_point"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    Expect(registry, std::move(node), "test_cc_basic_convinteger", {opset}, [=]() -> IoData {
      return IoData{{std::move(X), std::move(W), std::move(xzp)}, {std::move(Y)}};
    });
  }

  // -------------------------------------------------------------------
  // Case 2: 2x2 ConvInteger with padding=1.
  {
    Tensor X = Tensor::FromUint8("X", {1, 1, 3, 3}, {2, 3, 4, 5, 6, 7, 8, 9, 10});
    Tensor W = Tensor::FromUint8("W", {1, 1, 2, 2}, {1, 1, 1, 1});
    Tensor xzp = Tensor::FromUint8("x_zero_point", {}, {1});
    Tensor wzp;
    onnx_kernels::kernel::ConvInteger::Attributes attrs;
    attrs.kernel_shape = {2, 2};
    attrs.pads = {1, 1, 1, 1};
    Tensor Y = ci(X, W, xzp, wzp, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvIntegerNode({"X", "W", "x_zero_point"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    Expect(registry, std::move(node), "test_cc_convinteger_with_padding", {opset}, [=]() -> IoData {
      return IoData{{std::move(X), std::move(W), std::move(xzp)}, {std::move(Y)}};
    });
  }

  // -------------------------------------------------------------------
  // Case 3: mirrors ONNX ``test_convinteger_without_padding`` (1x1x3x3 input,
  // 1x1x2x2 weight, scalar x_zero_point=1, no kernel_shape attribute).
  {
    Tensor X = Tensor::FromUint8("X", {1, 1, 3, 3}, {2, 3, 4, 5, 6, 7, 8, 9, 10});
    Tensor W = Tensor::FromUint8("W", {1, 1, 2, 2}, {1, 1, 1, 1});
    Tensor xzp = Tensor::FromUint8("x_zero_point", {}, {1});
    Tensor wzp;
    onnx_kernels::kernel::ConvInteger::Attributes attrs;
    Tensor Y = ci(X, W, xzp, wzp, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvIntegerNode({"X", "W", "x_zero_point"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_convinteger_without_padding", {opset},
           [=]() -> IoData {
             return IoData{{std::move(X), std::move(W), std::move(xzp)}, {std::move(Y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
