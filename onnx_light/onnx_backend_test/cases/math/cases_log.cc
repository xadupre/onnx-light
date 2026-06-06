// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

Tensor PositiveRandFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  std::vector<float> values = Rand<float>(shape, seed);
  for (float &v : values) {
    v += 1.0e-6f;
  }
  return Tensor::FromFloat("", shape, values);
}

} // namespace

// ---------------------------------------------------------------------------
// Log — y = log(x) (latest opset: 13).
// Registers both a small deterministic ``test_cc_log`` case and upstream ONNX
// backend test cases (``test_log_example`` and ``test_log``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterLogCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Log log_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Log");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {0.1f, 0.5f, 1.0f, 2.0f, 4.0f, 10.0f});
    Tensor y = log_kernel(x);
    Expect(node, {x}, {y}, "test_cc_log", {opset}, "backend-test", registry);
  }

  // From Log.export(): ``test_log_example`` uses positive inputs.
  {
    NodeProto node;
    node.set_op_type("Log");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2}, {1.0f, 10.0f});
    Tensor y = log_kernel(x);
    Expect(node, {x}, {y}, "test_log_example", {opset}, "backend-test", registry);
  }

  // From Log.export(): ``test_log`` uses random positive inputs.
  {
    NodeProto node;
    node.set_op_type("Log");
    node.add_input("x");
    node.add_output("y");

    Tensor x = PositiveRandFloat({3, 4, 5}, /*seed=*/1);
    Tensor y = log_kernel(x);
    Expect(node, {x}, {y}, "test_log", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
