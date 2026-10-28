// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cmath>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// HammingWindow — generates a Hamming window of length ``size`` (since opset
// 17). Registers both the default periodic case and the symmetric variant
// (``periodic = 0``).
// ---------------------------------------------------------------------------
void RegisterHammingWindowCases(std::vector<TestCase> &registry, TestMode mode) {
  constexpr int32_t kSize = 10;
  const OpsetId opset = DefaultOpset(17);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("HammingWindow");
    node.add_input("x");
    node.add_output("y");
    constexpr int32_t size = 1 << 22;
    Expect(registry, std::move(node), "test_cc_hammingwindow_benchmark", {opset}, {1}, {size},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(17);

             const KernelContext hamming_kernel_ctx{opset};
             const onnx_kernels::kernel::HammingWindow hamming_kernel{hamming_kernel_ctx};

             Tensor x = Tensor::FromInt32("", {}, {size});
             Tensor y = hamming_kernel(x, /*periodic=*/true);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  for (const auto dtype : {DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8}) {
    for (const bool periodic : {true, false}) {
      NodeProto node;
      node.set_op_type("HammingWindow");
      node.add_input("x");
      node.add_output("y");
      auto *attr = node.add_attribute();
      attr->set_name("output_datatype");
      attr->set_type(AttributeProto::INT);
      attr->set_i(static_cast<int64_t>(dtype));
      attr = node.add_attribute();
      attr->set_name("periodic");
      attr->set_type(AttributeProto::INT);
      attr->set_i(periodic);
      const std::string type_name = dtype == DataType::FLOAT16    ? "float16"
                                    : dtype == DataType::BFLOAT16 ? "bfloat16"
                                                                  : "int8";
      Expect(
          registry, std::move(node),
          "test_cc_hammingwindow_" + type_name + (periodic ? "" : "_symmetric"), {opset},
          [dtype, periodic]() -> IoData {
            const int32_t size = dtype == DataType::INT8 ? 64 : 16384;
            Tensor x = Tensor::FromInt32("", {}, {size});
            std::vector<float> values(static_cast<size_t>(size));
            constexpr double pi = 3.14159265358979323846;
            for (int32_t i = 0; i < size; ++i) {
              values[static_cast<size_t>(i)] = static_cast<float>(
                  25.0 / 46.0 -
                  21.0 / 46.0 *
                      std::cos(2.0 * pi * (static_cast<double>(i) / (periodic ? size : size - 1))));
            }
            Tensor y = dtype == DataType::FLOAT16
                           ? core::runtime::MakeFloat16Tensor("", {size}, values)
                       : dtype == DataType::BFLOAT16
                           ? core::runtime::MakeBfloat16Tensor("", {size}, values)
                           : Tensor::FromInt8("", {size},
                                              std::vector<int8_t>(values.begin(), values.end()));
            return IoData{{std::move(x)}, {std::move(y)}};
          });
    }
  }

  // Default periodic variant.
  {
    NodeProto node;
    node.set_op_type("HammingWindow");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_hammingwindow", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(17);

      const KernelContext hamming_kernel_ctx{opset};
      const onnx_kernels::kernel::HammingWindow hamming_kernel{hamming_kernel_ctx};

      Tensor x = Tensor::FromInt32("", {}, {kSize});
      Tensor y = hamming_kernel(x, /*periodic=*/true);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Symmetric variant (periodic = 0).
  {
    NodeProto node;
    node.set_op_type("HammingWindow");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("periodic");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(0);
    Expect(registry, std::move(node), "test_cc_hammingwindow_symmetric", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(17);

      const KernelContext hamming_kernel_ctx{opset};
      const onnx_kernels::kernel::HammingWindow hamming_kernel{hamming_kernel_ctx};

      Tensor x = Tensor::FromInt32("", {}, {kSize});
      Tensor y = hamming_kernel(x, /*periodic=*/false);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
