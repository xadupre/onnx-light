// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// SwiGLU — gated activation Y = Swish_alpha(A) * B with
// Swish_alpha(a) = a * sigmoid(alpha * a). Inputs A (gate) and B (value) must
// have identical shapes (no broadcasting). ``alpha`` defaults to 1.0.
// ---------------------------------------------------------------------------
void RegisterSwiGLUCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(28);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkBinaryFloat<onnx_kernels::kernel::SwiGLU>("SwiGLU", "test_cc_swiglu_benchmark",
                                                             opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("SwiGLU");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_swiglu", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(28);

      const KernelContext swiglu_kernel_ctx{opset};
      const onnx_kernels::kernel::SwiGLU swiglu_kernel{swiglu_kernel_ctx};

      // No alpha attribute: defaults to 1.0.
      Tensor a = Tensor::FromFloat("", {2, 4}, {1.0f, -2.0f, 3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 0.5f});
      Tensor b = Tensor::FromFloat("", {2, 4}, {0.5f, 1.0f, -1.0f, 2.0f, 2.0f, -1.0f, 0.5f, 1.0f});
      Tensor y = swiglu_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("SwiGLU");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(0.5f);

    Expect(registry, std::move(node), "test_cc_swiglu_alpha", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(28);

      const KernelContext swiglu_kernel_ctx{opset};
      const onnx_kernels::kernel::SwiGLU swiglu_kernel{swiglu_kernel_ctx};

      Tensor a = Tensor::FromFloat("", {2, 4}, {1.0f, -2.0f, 3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 0.5f});
      Tensor b = Tensor::FromFloat("", {2, 4}, {0.5f, 1.0f, -1.0f, 2.0f, 2.0f, -1.0f, 0.5f, 1.0f});
      Tensor y = swiglu_kernel(a, b, 0.5f);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("SwiGLU");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_swiglu_float16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(28);

      const KernelContext swiglu_kernel_ctx{opset};
      const onnx_kernels::kernel::SwiGLU swiglu_kernel{swiglu_kernel_ctx};

      // No alpha attribute: defaults to 1.0.
      Tensor a = MakeFloat16Tensor("", {2, 4}, {1.0f, -2.0f, 3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 0.5f});
      Tensor b = MakeFloat16Tensor("", {2, 4}, {0.5f, 1.0f, -1.0f, 2.0f, 2.0f, -1.0f, 0.5f, 1.0f});
      Tensor y = swiglu_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("SwiGLU");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_swiglu_bfloat16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(28);

      const KernelContext swiglu_kernel_ctx{opset};
      const onnx_kernels::kernel::SwiGLU swiglu_kernel{swiglu_kernel_ctx};

      // No alpha attribute: defaults to 1.0.
      Tensor a =
          MakeBfloat16Tensor("", {2, 4}, {1.0f, -2.0f, 3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 0.5f});
      Tensor b = MakeBfloat16Tensor("", {2, 4}, {0.5f, 1.0f, -1.0f, 2.0f, 2.0f, -1.0f, 0.5f, 1.0f});
      Tensor y = swiglu_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
