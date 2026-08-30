// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
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
// Sum — element-wise variadic sum with NumPy-style broadcasting (since
// opset 8; opset 13 widens the type constraint to include bfloat16).
// ---------------------------------------------------------------------------
void RegisterSumCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const auto sum_kernel = MakeReferenceKernel<onnx_kernels::kernel::Sum>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("sum");
    const std::vector<int64_t> shape = {kBenchmarkElementwiseSize};
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_sum_benchmark", {opset}, {count, count}, {count},
           [sum_kernel, shape]() -> IoData {
             Tensor x0 = RandnTensor(DataType::FLOAT, shape, 423);
             Tensor x1 = RandnTensor(DataType::FLOAT, shape, 424);
             Tensor z = sum_kernel.Invoke([&](const auto &kernel) { return kernel({x0, x1}); });
             return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
           });
    return;
  }

  // Single-input variant: Sum acts as Identity.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_output("sum");
    Expect(registry, std::move(node), "test_cc_sum_one_input", {opset}, [=]() -> IoData {
      Tensor x0 = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor z = sum_kernel.Invoke([&](const auto &kernel) { return kernel({x0}); });

      return IoData{{std::move(x0)}, {std::move(z)}};
    });
  }

  // Two equal-shape inputs.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("sum");
    Expect(registry, std::move(node), "test_cc_sum_two_inputs", {opset}, [=]() -> IoData {
      Tensor x0 = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor x1 = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
      Tensor z = sum_kernel.Invoke([&](const auto &kernel) { return kernel({x0, x1}); });

      return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
    });
  }

  // Three equal-shape inputs.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_input("data_2");
    node.add_output("sum");
    Expect(registry, std::move(node), "test_cc_sum_example", {opset}, [=]() -> IoData {
      Tensor x0 = Tensor::FromFloat("", {3}, {1.0f, 0.0f, 1.0f});
      Tensor x1 = Tensor::FromFloat("", {3}, {3.0f, 4.0f, 5.0f});
      Tensor x2 = Tensor::FromFloat("", {3}, {6.0f, 0.0f, 5.0f});
      Tensor z = sum_kernel.Invoke([&](const auto &kernel) { return kernel({x0, x1, x2}); });

      return IoData{{std::move(x0), std::move(x1), std::move(x2)}, {std::move(z)}};
    });
  }

  // Broadcasting variant: scalar broadcast against rank-2 input.
  {
    NodeProto node;
    node.set_op_type("Sum");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("sum");
    Expect(registry, std::move(node), "test_cc_sum_bcast", {opset}, [=]() -> IoData {
      Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor x1 = Tensor::FromFloat("", {}, {10.0f});
      Tensor z = sum_kernel.Invoke([&](const auto &kernel) { return kernel({x0, x1}); });

      return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
    });
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
    Expect(registry, std::move(node), "test_sum_example", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x0 = RandnFloat(shape, /*seed=*/61);
      Tensor x1 = RandnFloat(shape, /*seed=*/62);
      Tensor x2 = RandnFloat(shape, /*seed=*/63);
      Tensor z = sum_kernel.Invoke([&](const auto &kernel) { return kernel({x0, x1, x2}); });

      return IoData{{std::move(x0), std::move(x1), std::move(x2)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
