// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// Mean — element-wise variadic mean with NumPy-style broadcasting (since
// opset 8; opset 13 widens the type constraint to include bfloat16).
//
// Mirrors ``onnx.backend.test.case.node.mean.Mean`` from upstream ONNX:
// test_mean_example, test_mean_one_input, test_mean_two_inputs.
// ---------------------------------------------------------------------------
void RegisterMeanCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");
    const std::vector<int64_t> shape = {kBenchmarkElementwiseSize};
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_mean_benchmark", {opset}, {count, count}, {count},
           [shape]() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext mean_kernel_ctx{opset};
             const onnx_kernels::kernel::Mean mean_kernel{mean_kernel_ctx};

             Tensor x0 = RandnTensor(DataType::FLOAT, shape, 425);
             Tensor x1 = RandnTensor(DataType::FLOAT, shape, 426);
             Tensor z = mean_kernel({x0, x1});
             return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
           });
    return;
  }

  // Upstream ``test_mean_example``: three equal-shape inputs.
  {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_input("data_2");
    node.add_output("result");
    Expect(registry, std::move(node), "test_mean_example", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mean_kernel_ctx{opset};
      const onnx_kernels::kernel::Mean mean_kernel{mean_kernel_ctx};

      Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 0.0f, 2.0f});
      Tensor x1 = Tensor::FromFloat("", {3}, {1.0f, 3.0f, 4.0f});
      Tensor x2 = Tensor::FromFloat("", {3}, {2.0f, 6.0f, 6.0f});
      Tensor z = mean_kernel({x0, x1, x2});

      return IoData{{std::move(x0), std::move(x1), std::move(x2)}, {std::move(z)}};
    });
  }

  // Upstream ``test_mean_one_input``: single input acts as Identity.
  {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_output("result");
    Expect(registry, std::move(node), "test_mean_one_input", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mean_kernel_ctx{opset};
      const onnx_kernels::kernel::Mean mean_kernel{mean_kernel_ctx};

      Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 0.0f, 2.0f});
      Tensor z = mean_kernel({x0});

      return IoData{{std::move(x0)}, {std::move(z)}};
    });
  }

  // Upstream ``test_mean_two_inputs``: two equal-shape inputs.
  {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");
    Expect(registry, std::move(node), "test_mean_two_inputs", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mean_kernel_ctx{opset};
      const onnx_kernels::kernel::Mean mean_kernel{mean_kernel_ctx};

      Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 0.0f, 2.0f});
      Tensor x1 = Tensor::FromFloat("", {3}, {1.0f, 3.0f, 4.0f});
      Tensor z = mean_kernel({x0, x1});

      return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
    });
  }

  // Broadcasting variant: scalar broadcast against rank-2 input.
  {
    NodeProto node;
    node.set_op_type("Mean");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");
    Expect(registry, std::move(node), "test_cc_mean_bcast", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mean_kernel_ctx{opset};
      const onnx_kernels::kernel::Mean mean_kernel{mean_kernel_ctx};

      Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor x1 = Tensor::FromFloat("", {}, {10.0f});
      Tensor z = mean_kernel({x0, x1});

      return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
