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

void RegisterConvIntegerCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(10);
  const kernel::KernelContext ctx{opset};
  const kernel::ConvInteger ci{ctx};

  // -------------------------------------------------------------------
  // Case 1: basic 2x2 ConvInteger without padding, with x_zero_point.
  {
    Tensor X = Tensor::FromUint8("X", {1, 1, 3, 3}, {2, 3, 4, 5, 6, 7, 8, 9, 10});
    Tensor W = Tensor::FromUint8("W", {1, 1, 2, 2}, {1, 1, 1, 1});
    Tensor xzp = Tensor::FromUint8("x_zero_point", {}, {1});
    Tensor wzp;
    kernel::ConvInteger::Attributes attrs;
    attrs.kernel_shape = {2, 2};
    Tensor Y = ci(X, W, xzp, wzp, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvIntegerNode({"X", "W", "x_zero_point"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    Expect(node, {X, W, xzp}, {Y}, "test_cc_basic_convinteger", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 2: 2x2 ConvInteger with padding=1.
  {
    Tensor X = Tensor::FromUint8("X", {1, 1, 3, 3}, {2, 3, 4, 5, 6, 7, 8, 9, 10});
    Tensor W = Tensor::FromUint8("W", {1, 1, 2, 2}, {1, 1, 1, 1});
    Tensor xzp = Tensor::FromUint8("x_zero_point", {}, {1});
    Tensor wzp;
    kernel::ConvInteger::Attributes attrs;
    attrs.kernel_shape = {2, 2};
    attrs.pads = {1, 1, 1, 1};
    Tensor Y = ci(X, W, xzp, wzp, attrs);
    Y.name = "Y";
    NodeProto node = MakeConvIntegerNode({"X", "W", "x_zero_point"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    Expect(node, {X, W, xzp}, {Y}, "test_cc_convinteger_with_padding", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
