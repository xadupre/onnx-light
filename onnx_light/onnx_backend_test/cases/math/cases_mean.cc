// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Mean — element-wise variadic mean with NumPy-style broadcasting (since
// opset 8; opset 13 widens the type constraint to include bfloat16).
//
// Mirrors ``onnx.backend.test.case.node.mean.Mean`` from upstream ONNX:
// test_mean_example, test_mean_one_input, test_mean_two_inputs.
// ---------------------------------------------------------------------------
void RegisterMeanCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Mean mean_kernel{ctx};

  // Upstream ``test_mean_example``: three equal-shape inputs.
  {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_input("data_2");
    node.add_output("result");

    Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 0.0f, 2.0f});
    Tensor x1 = Tensor::FromFloat("", {3}, {1.0f, 3.0f, 4.0f});
    Tensor x2 = Tensor::FromFloat("", {3}, {2.0f, 6.0f, 6.0f});
    Tensor z = mean_kernel({x0, x1, x2});

    Expect(node, {x0, x1, x2}, {z}, "test_mean_example", {opset}, "backend-test", registry);
  }

  // Upstream ``test_mean_one_input``: single input acts as Identity.
  {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_output("result");

    Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 0.0f, 2.0f});
    Tensor z = mean_kernel({x0});

    Expect(node, {x0}, {z}, "test_mean_one_input", {opset}, "backend-test", registry);
  }

  // Upstream ``test_mean_two_inputs``: two equal-shape inputs.
  {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");

    Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 0.0f, 2.0f});
    Tensor x1 = Tensor::FromFloat("", {3}, {1.0f, 3.0f, 4.0f});
    Tensor z = mean_kernel({x0, x1});

    Expect(node, {x0, x1}, {z}, "test_mean_two_inputs", {opset}, "backend-test", registry);
  }

  // Broadcasting variant: scalar broadcast against rank-2 input.
  {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");

    Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor x1 = Tensor::FromFloat("", {}, {10.0f});
    Tensor z = mean_kernel({x0, x1});

    Expect(node, {x0, x1}, {z}, "test_cc_mean_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
