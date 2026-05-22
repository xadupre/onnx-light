// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"

#include <cmath>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Builds an OperatorSetIdProto for the default ai.onnx domain.
OperatorSetIdProto DefaultOpset(int64_t version) {
  OperatorSetIdProto osid;
  osid.set_domain("");
  osid.set_version(version);
  return osid;
}

} // namespace

// ---------------------------------------------------------------------------
// Add — z = x + y, element-wise with broadcasting (since opset 14).
// This is the case exercised by examples/run_add_node_test/main.cc.
// ---------------------------------------------------------------------------
void RegisterAddCases(std::vector<TestCase> &registry) {
  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    std::vector<float> z(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      z[i] = x[i] + y[i];
    }

    Expect(node, {Tensor::FromFloat("x", {2, 3}, x), Tensor::FromFloat("y", {2, 3}, y)},
           {Tensor::FromFloat("z", {2, 3}, z)}, "test_cc_add", {DefaultOpset(14)}, "backend-test",
           registry);
  }

  // Scalar broadcast variant: z[i] = x[i] + y (scalar).
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {0.5f};
    std::vector<float> z(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      z[i] = x[i] + y[0];
    }

    Expect(node, {Tensor::FromFloat("x", {2, 2}, x), Tensor::FromFloat("y", {}, y)},
           {Tensor::FromFloat("z", {2, 2}, z)}, "test_cc_add_bcast", {DefaultOpset(14)},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
