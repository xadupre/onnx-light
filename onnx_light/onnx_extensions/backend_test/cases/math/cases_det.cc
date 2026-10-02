// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <initializer_list>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterDetCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");
    const std::vector<int64_t> shape = {512, 64, 64};
    Expect(registry, std::move(node), "test_cc_det_benchmark", {opset}, {512 * 64 * 64}, {512},
           [shape]() -> IoData {
             const OpsetId opset = DefaultOpset(11);

             const KernelContext det_kernel_ctx{opset};
             const onnx_kernels::kernel::Det det_kernel{det_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, shape, 439);
             Tensor y = det_kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // 2-D input: output is a scalar (matches ONNX ``test_det_2d``).
  {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_det_2d", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext det_kernel_ctx{opset};
      const onnx_kernels::kernel::Det det_kernel{det_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = det_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // N-D input: batch of square matrices (matches ONNX ``test_det_nd``).
  {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_det_nd", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext det_kernel_ctx{opset};
      const onnx_kernels::kernel::Det det_kernel{det_kernel_ctx};

      Tensor x = Tensor::FromFloat(
          "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 3.0f, 3.0f, 1.0f});
      Tensor y = det_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Low-precision inputs preserve their dtype for batched determinants,
  // including a singular matrix.
  for (const bool use_bfloat16 : {false, true}) {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");
    const std::string name = use_bfloat16 ? "test_cc_det_bfloat16" : "test_cc_det_float16";
    Expect(registry, std::move(node), name, {opset}, [use_bfloat16]() -> IoData {
      const std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f,
                                      2.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
      if (use_bfloat16) {
        return IoData{{MakeBfloat16Tensor("", {3, 2, 2}, values)},
                      {MakeBfloat16Tensor("", {3}, {-2.0f, 0.0f, -2.0f})}};
      }
      return IoData{{MakeFloat16Tensor("", {3, 2, 2}, values)},
                    {MakeFloat16Tensor("", {3}, {-2.0f, 0.0f, -2.0f})}};
    });
  }

  // DOUBLE retains precision that is lost when the input is narrowed to FLOAT.
  {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_det_double", {opset}, []() -> IoData {
      const double expected = std::ldexp(1.0, -40);
      Tensor x = Tensor::FromDouble("", {2, 2}, {1.0, 1.0, 1.0, 1.0 + expected});
      Tensor y = Tensor::FromDouble("", {}, {expected});
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
