// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// Sum — element-wise variadic sum with NumPy-style broadcasting (since
// opset 8; opset 13 widens the type constraint to include bfloat16).
// ---------------------------------------------------------------------------
void RegisterSumCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Sum sum_kernel{ctx};

  // Single-input variant: Sum acts as Identity.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_output("sum");

    Tensor x0 = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor z = sum_kernel({x0});

    Expect(node, {x0}, {z}, "test_cc_sum_one_input", {opset}, "backend-test", registry);
  }

  // Two equal-shape inputs.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("sum");

    Tensor x0 = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor x1 = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor z = sum_kernel({x0, x1});

    Expect(node, {x0, x1}, {z}, "test_cc_sum_two_inputs", {opset}, "backend-test", registry);
  }

  // Three equal-shape inputs.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_input("data_2");
    node.add_output("sum");

    Tensor x0 = Tensor::FromFloat("", {3}, {1.0f, 0.0f, 1.0f});
    Tensor x1 = Tensor::FromFloat("", {3}, {3.0f, 4.0f, 5.0f});
    Tensor x2 = Tensor::FromFloat("", {3}, {6.0f, 0.0f, 5.0f});
    Tensor z = sum_kernel({x0, x1, x2});

    Expect(node, {x0, x1, x2}, {z}, "test_cc_sum_example", {opset}, "backend-test", registry);
  }

  // Broadcasting variant: scalar broadcast against rank-2 input.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("sum");

    Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor x1 = Tensor::FromFloat("", {}, {10.0f});
    Tensor z = sum_kernel({x0, x1});

    Expect(node, {x0, x1}, {z}, "test_cc_sum_bcast", {opset}, "backend-test", registry);
  }

  // Upstream-style multi-input random case mirroring
  // ``onnx.backend.test.case.node.sum.Sum``.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_input("data_2");
    node.add_output("sum");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x0 = RandnFloat(shape, /*seed=*/61);
    Tensor x1 = RandnFloat(shape, /*seed=*/62);
    Tensor x2 = RandnFloat(shape, /*seed=*/63);
    Tensor z = sum_kernel({x0, x1, x2});

    Expect(node, {x0, x1, x2}, {z}, "test_sum_example", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
