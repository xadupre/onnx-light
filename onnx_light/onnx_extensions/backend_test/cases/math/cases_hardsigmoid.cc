// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterHardSigmoidCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("HardSigmoid");
    node.add_input("x");
    node.add_output("y");
    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(0.5f);
    AttributeProto *beta = node.add_attribute();
    beta->set_name("beta");
    beta->set_type(AttributeProto::FLOAT);
    beta->set_f(0.6f);
    const int64_t n = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_hardsigmoid_benchmark", {opset}, {n}, {n},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext hard_sigmoid_kernel_ctx{opset};
             const onnx_kernels::kernel::HardSigmoid hard_sigmoid_kernel{hard_sigmoid_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {n}, 987654321ULL);
             Tensor y = hard_sigmoid_kernel(x, 0.5f, 0.6f);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  {
    NodeProto node;
    node.set_op_type("HardSigmoid");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(0.5f);

    AttributeProto *beta = node.add_attribute();
    beta->set_name("beta");
    beta->set_type(AttributeProto::FLOAT);
    beta->set_f(0.6f);

    Expect(registry, std::move(node), "test_cc_hardsigmoid_example", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext hard_sigmoid_kernel_ctx{opset};
      const onnx_kernels::kernel::HardSigmoid hard_sigmoid_kernel{hard_sigmoid_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = hard_sigmoid_kernel(x, 0.5f, 0.6f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("HardSigmoid");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(0.5f);

    AttributeProto *beta = node.add_attribute();
    beta->set_name("beta");
    beta->set_type(AttributeProto::FLOAT);
    beta->set_f(0.6f);

    Expect(registry, std::move(node), "test_cc_hardsigmoid", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext hard_sigmoid_kernel_ctx{opset};
      const onnx_kernels::kernel::HardSigmoid hard_sigmoid_kernel{hard_sigmoid_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, -0.5f, 0.5f, 1.0f, 3.0f});
      Tensor y = hard_sigmoid_kernel(x, 0.5f, 0.6f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("HardSigmoid");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_hardsigmoid_default", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext hard_sigmoid_kernel_ctx{opset};
      const onnx_kernels::kernel::HardSigmoid hard_sigmoid_kernel{hard_sigmoid_kernel_ctx};

      // No alpha/beta attributes: defaults to 0.2 and 0.5.
      Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, -0.5f, 0.5f, 1.0f, 3.0f});
      Tensor y = hard_sigmoid_kernel(x, 0.2f, 0.5f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
