// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeNonZeroNode() {
  NodeProto node;
  node.set_op_type("NonZero");
  node.add_input("X");
  node.add_output("Y");
  return node;
}

} // namespace

void RegisterNonZeroCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNonZeroNode();
    Expect(registry, std::move(node), "test_cc_nonzero_2d_benchmark", {opset},
           {kBenchmarkElementwiseSize}, {2 * kBenchmarkElementwiseSize}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext nonzero_kernel_ctx{opset};
             const onnx_kernels::kernel::NonZero nonzero_kernel{nonzero_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {2048, 2048}, 2001);
             Tensor y = nonzero_kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // test_cc_nonzero_2d
  {
    Expect(registry, MakeNonZeroNode(), "test_cc_nonzero_2d", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext nonzero_kernel_ctx{opset};
      const onnx_kernels::kernel::NonZero nonzero_kernel{nonzero_kernel_ctx};

      const Tensor x = Tensor::FromFloat("X", {2, 2}, {1.0f, 0.0f, 1.0f, 1.0f});
      const Tensor y = nonzero_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_nonzero_1d
  {
    Expect(registry, MakeNonZeroNode(), "test_cc_nonzero_1d", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext nonzero_kernel_ctx{opset};
      const onnx_kernels::kernel::NonZero nonzero_kernel{nonzero_kernel_ctx};

      const Tensor x = Tensor::FromFloat("X", {5}, {0.0f, 1.0f, 0.0f, -1.0f, 2.0f});
      const Tensor y = nonzero_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_nonzero_bool
  {
    Expect(registry, MakeNonZeroNode(), "test_cc_nonzero_bool", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext nonzero_kernel_ctx{opset};
      const onnx_kernels::kernel::NonZero nonzero_kernel{nonzero_kernel_ctx};

      const Tensor x = Tensor::FromBool("X", {2, 3}, {1, 0, 1, 0, 1, 0});
      const Tensor y = nonzero_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_nonzero_int64
  {
    Expect(registry, MakeNonZeroNode(), "test_cc_nonzero_int64", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext nonzero_kernel_ctx{opset};
      const onnx_kernels::kernel::NonZero nonzero_kernel{nonzero_kernel_ctx};

      const Tensor x = Tensor::FromInt64("X", {2, 3}, {0, 1, 2, 0, 0, 3});
      const Tensor y = nonzero_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_nonzero_example — mirrors upstream
  // ``onnx.backend.test.case.node.nonzero.NonZero.export``:
  //   condition = [[1, 0], [1, 1]] as bool
  //   result    = [[0, 1, 1], [0, 0, 1]] (np.nonzero stacked, int64)
  {
    const Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 1});
    NodeProto node;
    node.set_op_type("NonZero");
    node.add_input("condition");
    node.add_output("result");
    Expect(registry, std::move(node), "test_cc_nonzero_example", {opset}, []() -> IoData {
      const Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 1});

      const OpsetId opset = DefaultOpset(13);

      const KernelContext nonzero_kernel_ctx{opset};
      const onnx_kernels::kernel::NonZero nonzero_kernel{nonzero_kernel_ctx};

      Tensor result = nonzero_kernel(condition);
      result.name = "result";
      return IoData{{std::move(condition)}, {std::move(result)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
