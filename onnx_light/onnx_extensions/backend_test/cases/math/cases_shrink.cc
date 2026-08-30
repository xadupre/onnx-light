// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterShrinkCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(9);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Shrink");
    node.add_input("x");
    node.add_output("y");
    AttributeProto *lambd = node.add_attribute();
    lambd->set_name("lambd");
    lambd->set_type(AttributeProto::FLOAT);
    lambd->set_f(1.5f);
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_shrink_benchmark", {opset}, {count}, {count},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(9);

             const KernelContext shrink_kernel_ctx{opset};
             const onnx_kernels::kernel::Shrink shrink_kernel{shrink_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 987654321ULL);
             Tensor y = shrink_kernel(x, /*bias=*/0.0f, /*lambd=*/1.5f);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  {
    // Mirrors the ONNX test_shrink_hard reference (bias=0.0, lambd=1.5).
    NodeProto node;
    node.set_op_type("Shrink");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *lambd = node.add_attribute();
    lambd->set_name("lambd");
    lambd->set_type(AttributeProto::FLOAT);
    lambd->set_f(1.5f);

    Expect(registry, std::move(node), "test_cc_shrink_hard", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(9);

      const KernelContext shrink_kernel_ctx{opset};
      const onnx_kernels::kernel::Shrink shrink_kernel{shrink_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
      Tensor y = shrink_kernel(x, /*bias=*/0.0f, /*lambd=*/1.5f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    // Mirrors the ONNX test_shrink_soft reference (bias=1.5, lambd=1.5).
    NodeProto node;
    node.set_op_type("Shrink");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *bias = node.add_attribute();
    bias->set_name("bias");
    bias->set_type(AttributeProto::FLOAT);
    bias->set_f(1.5f);

    AttributeProto *lambd = node.add_attribute();
    lambd->set_name("lambd");
    lambd->set_type(AttributeProto::FLOAT);
    lambd->set_f(1.5f);

    Expect(registry, std::move(node), "test_cc_shrink_soft", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(9);

      const KernelContext shrink_kernel_ctx{opset};
      const onnx_kernels::kernel::Shrink shrink_kernel{shrink_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
      Tensor y = shrink_kernel(x, /*bias=*/1.5f, /*lambd=*/1.5f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    // Default attributes: bias=0.0, lambd=0.5.
    NodeProto node;
    node.set_op_type("Shrink");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_shrink_default", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(9);

      const KernelContext shrink_kernel_ctx{opset};
      const onnx_kernels::kernel::Shrink shrink_kernel{shrink_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -0.5f, -0.1f, 0.1f, 0.5f, 1.0f});
      Tensor y = shrink_kernel(x, /*bias=*/0.0f, /*lambd=*/0.5f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
