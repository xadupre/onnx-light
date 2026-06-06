// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"

#include <cmath>
#include <cstdint>
#include <limits>
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
// PRelu — y = x when x >= 0, slope * x otherwise. Element-wise with
// unidirectional broadcasting of ``slope`` to ``x`` (since opset 16).
// ---------------------------------------------------------------------------
void RegisterPReluCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(16);
  const kernel::KernelContext ctx{opset};
  const kernel::PRelu prelu_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("PRelu");
    node.add_input("x");
    node.add_input("slope");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor slope = Tensor::FromFloat("", {2, 3}, {0.25f, 0.5f, 0.75f, 0.1f, 0.2f, 0.3f});
    Tensor y = prelu_kernel(x, slope);

    Expect(node, {x, slope}, {y}, "test_cc_prelu", {opset}, "backend-test", registry);
  }

  // Unidirectional broadcast variant: slope is a 1-D tensor along the last
  // axis of ``x`` (mirrors the upstream broadcast case shape pattern).
  {
    NodeProto node;
    node.set_op_type("PRelu");
    node.add_input("x");
    node.add_input("slope");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -2.0f, -3.0f, 1.0f, 2.0f, 3.0f});
    Tensor slope = Tensor::FromFloat("", {3}, {0.1f, 0.2f, 0.3f});
    Tensor y = prelu_kernel(x, slope);

    Expect(node, {x, slope}, {y}, "test_cc_prelu_bcast", {opset}, "backend-test", registry);
  }

  // Regression for microsoft/onnxruntime#28732: PRelu must preserve ``+inf``
  // and ``-inf`` inputs (positive inputs are returned unchanged; negative
  // inputs are multiplied by the slope, so ``-inf`` with a positive slope
  // remains ``-inf``). Naive implementations evaluating
  // ``max(0, x) + slope * min(0, x)`` produce ``NaN`` here because
  // ``slope * (+inf)`` is ``+inf`` and ``slope * (-inf)`` is ``-inf`` on the
  // wrong branch.
  {
    NodeProto node;
    node.set_op_type("PRelu");
    node.add_input("x");
    node.add_input("slope");
    node.add_output("y");

    const float pinf = std::numeric_limits<float>::infinity();
    const float ninf = -std::numeric_limits<float>::infinity();
    Tensor x = Tensor::FromFloat("", {4}, {pinf, ninf, 5e30f, -2.5f});
    Tensor slope = Tensor::FromFloat("", {4}, {0.25f, 0.5f, 0.25f, 0.25f});
    Tensor y = prelu_kernel(x, slope);

    Expect(node, {x, slope}, {y}, "test_cc_prelu_inf", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``PRelu`` operator (mirror the
  // ``onnx.backend.test.case.node.prelu.PRelu`` Python class). Inputs are
  // generated deterministically through the seeded ``Randn`` helper to mirror
  // the upstream ``np.random.randn(...)`` pattern; expected outputs are
  // computed by ``kernel::PRelu`` so the recorded expectations stay
  // self-consistent with this library.
  NodeProto node;
  node.set_op_type("PRelu");
  node.add_input("x");
  node.add_input("slope");
  node.add_output("y");

  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From PRelu.export():
      {"test_prelu_example",
       {RandnFloat({3, 4, 5}, /*seed=*/101), RandnFloat({3, 4, 5}, /*seed=*/102)}},
      // From PRelu.export_prelu_broadcast():
      {"test_prelu_broadcast",
       {RandnFloat({3, 4, 5}, /*seed=*/103), RandnFloat({5}, /*seed=*/104)}},
  };

  for (const auto &[name, inputs] : cases) {
    Tensor y = prelu_kernel(inputs[0], inputs[1]);
    Expect(node, inputs, {y}, name, {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
