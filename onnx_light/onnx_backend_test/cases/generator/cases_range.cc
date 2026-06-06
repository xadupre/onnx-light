// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_kernels/kernels/generator/include_generator_kernels.h"
#include "onnx_kernels/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterRangeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};

  // Upstream test: range_float_type_positive_delta
  // start=1, limit=5, delta=2  ->  [1.0, 3.0]
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");

    const Tensor start = Tensor::FromFloat("start", {}, {1.0f});
    const Tensor limit = Tensor::FromFloat("limit", {}, {5.0f});
    const Tensor delta = Tensor::FromFloat("delta", {}, {2.0f});
    const Tensor output = kernel::Range(ctx)(start, limit, delta);
    Expect(node, {start, limit, delta}, {output}, "test_range_float_type_positive_delta", {opset},
           "backend-test", registry);
  }

  // Upstream test: range_int32_type_negative_delta
  // start=10, limit=6, delta=-3  ->  [10, 7]
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");

    const Tensor start = Tensor::FromInt32("start", {}, {10});
    const Tensor limit = Tensor::FromInt32("limit", {}, {6});
    const Tensor delta = Tensor::FromInt32("delta", {}, {-3});
    const Tensor output = kernel::Range(ctx)(start, limit, delta);
    Expect(node, {start, limit, delta}, {output}, "test_range_int32_type_negative_delta", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
