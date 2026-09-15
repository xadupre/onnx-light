// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterThresholdedReluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(10);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("ThresholdedRelu");
    node.add_input("x");
    node.add_output("y");
    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_thresholdedrelu_benchmark", {opset}, {count},
           {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext thresholdedrelu_kernel_ctx{opset};
             const onnx_kernels::kernel::ThresholdedRelu thresholdedrelu_kernel{
                 thresholdedrelu_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 987654321ULL);
             Tensor y = thresholdedrelu_kernel(x, 2.0f);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  {
    NodeProto node;
    node.set_op_type("ThresholdedRelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    Expect(registry, std::move(node), "test_cc_thresholdedrelu_example", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(10);

      const KernelContext thresholdedrelu_kernel_ctx{opset};
      const onnx_kernels::kernel::ThresholdedRelu thresholdedrelu_kernel{
          thresholdedrelu_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {3, 4, 5}, std::vector<float>(60, 3.0f));
      Tensor y = thresholdedrelu_kernel(x, 2.0f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("ThresholdedRelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    Expect(registry, std::move(node), "test_cc_thresholdedrelu", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(10);

      const KernelContext thresholdedrelu_kernel_ctx{opset};
      const onnx_kernels::kernel::ThresholdedRelu thresholdedrelu_kernel{
          thresholdedrelu_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 2.5f, 3.0f});
      Tensor y = thresholdedrelu_kernel(x, 2.0f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("ThresholdedRelu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_thresholdedrelu_default", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(10);

      const KernelContext thresholdedrelu_kernel_ctx{opset};
      const onnx_kernels::kernel::ThresholdedRelu thresholdedrelu_kernel{
          thresholdedrelu_kernel_ctx};

      // No alpha attribute: defaults to 1.0.
      Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f});
      Tensor y = thresholdedrelu_kernel(x, 1.0f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
