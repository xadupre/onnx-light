// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return RandnTensor(DataType::FLOAT, shape, seed);
}

} // namespace

// ---------------------------------------------------------------------------
// PRelu — y = x when x >= 0, slope * x otherwise. Element-wise with
// unidirectional broadcasting of ``slope`` to ``x`` (since opset 16).
// ---------------------------------------------------------------------------
void RegisterPReluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(16);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::PRelu prelu_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkBinaryFloat("PRelu", prelu_kernel, "test_cc_prelu_benchmark", opset, registry);
    return;
  }

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("PRelu");
    node.add_input("x");
    node.add_input("slope");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_prelu", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor slope = Tensor::FromFloat("", {2, 3}, {0.25f, 0.5f, 0.75f, 0.1f, 0.2f, 0.3f});
      Tensor y = prelu_kernel(x, slope);

      return IoData{{std::move(x), std::move(slope)}, {std::move(y)}};
    });
  }

  // Unidirectional broadcast variant: slope is a 1-D tensor along the last
  // axis of ``x`` (mirrors the upstream broadcast case shape pattern).
  {
    NodeProto node;
    node.set_op_type("PRelu");
    node.add_input("x");
    node.add_input("slope");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_prelu_bcast", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -2.0f, -3.0f, 1.0f, 2.0f, 3.0f});
      Tensor slope = Tensor::FromFloat("", {3}, {0.1f, 0.2f, 0.3f});
      Tensor y = prelu_kernel(x, slope);

      return IoData{{std::move(x), std::move(slope)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_prelu_inf", {opset}, [=]() -> IoData {
      const float pinf = std::numeric_limits<float>::infinity();
      const float ninf = -std::numeric_limits<float>::infinity();
      Tensor x = Tensor::FromFloat("", {4}, {pinf, ninf, 5e30f, -2.5f});
      Tensor slope = Tensor::FromFloat("", {4}, {0.25f, 0.5f, 0.25f, 0.25f});
      Tensor y = prelu_kernel(x, slope);

      return IoData{{std::move(x), std::move(slope)}, {std::move(y)}};
    });
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

  const std::vector<std::pair<std::string, std::function<IoData()>>> cases = {
      // From PRelu.export():
      {"test_prelu_example",
       [=]() -> IoData {
         auto inputs_0 = RandnFloat({3, 4, 5}, /*seed=*/101);
         auto inputs_1 = RandnFloat({3, 4, 5}, /*seed=*/102);
         Tensor y = prelu_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(y)}};
       }},
      // From PRelu.export_prelu_broadcast():
      {"test_prelu_broadcast",
       [=]() -> IoData {
         auto inputs_0 = RandnFloat({3, 4, 5}, /*seed=*/103);
         auto inputs_1 = RandnFloat({5}, /*seed=*/104);
         Tensor y = prelu_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(y)}};
       }},
  };

  for (const auto &[name, make_io] : cases) {
    Expect(registry, node, name, {opset}, make_io);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
