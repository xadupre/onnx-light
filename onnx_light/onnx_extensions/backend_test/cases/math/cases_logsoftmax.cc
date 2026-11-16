// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterLogSoftmaxCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);
    const std::vector<int64_t> shape = {2048, 2048};
    const int64_t count = 2048 * 2048;
    Expect(registry, std::move(node), "test_cc_logsoftmax_benchmark", {opset}, {count}, {count},
           [shape]() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext logsoftmax_kernel_ctx{opset};
             const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{logsoftmax_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, shape, 432);
             Tensor y = logsoftmax_kernel(x, 1);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  for (const auto dtype :
       {DataType::FLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::DOUBLE}) {
    const std::string suffix = dtype == DataType::FLOAT16    ? "float16"
                               : dtype == DataType::BFLOAT16 ? "bfloat16"
                               : dtype == DataType::FLOAT    ? "float32"
                                                             : "float64";
    for (const int64_t axis : {0, 1, -1}) {
      NodeProto node;
      node.set_op_type("LogSoftmax");
      node.add_input("input");
      node.add_output("output");
      auto *attribute = node.add_attribute();
      attribute->set_name("axis");
      attribute->set_type(AttributeProto::INT);
      attribute->set_i(axis);
      Expect(
          registry, std::move(node),
          "test_cc_reference_logsoftmax_small_log_probability_" + suffix + "_axis_" +
              std::to_string(axis),
          {DefaultOpset(13)},
          [dtype, axis]() -> IoData {
            const double margin = dtype == DataType::DOUBLE    ? 40.0
                                  : dtype == DataType::FLOAT16 ? 10.0
                                                               : 18.0;
            const double normalizer = std::log1p(2 * std::exp(-margin));
            const std::vector<int64_t> shape =
                axis == 0 ? std::vector<int64_t>{3, 2, 3} : std::vector<int64_t>{2, 3, 3};
            const int64_t inner = axis == 0 ? 6 : axis == 1 ? 3 : 1;
            std::vector<double> scores(18), expected(18);
            for (int64_t row = 0; row < 6; ++row) {
              for (int64_t a = 0; a < 3; ++a) {
                const int64_t offset = (row / inner * 3 + a) * inner + row % inner;
                scores[offset] = a == row % 3 ? 0.0 : -margin;
                expected[offset] = scores[offset] - normalizer;
              }
            }
            if (dtype == DataType::DOUBLE) {
              return IoData{{Tensor::FromDouble("", shape, scores)},
                            {Tensor::FromDouble("", shape, expected)}};
            }
            Tensor x =
                Tensor::FromFloat("", shape, std::vector<float>(scores.begin(), scores.end()));
            Tensor y =
                Tensor::FromFloat("", shape, std::vector<float>(expected.begin(), expected.end()));
            if (core::runtime::IsHalfPrecision(dtype)) {
              x = core::runtime::DemoteFromFloat32(x, dtype);
              y = core::runtime::DemoteFromFloat32(y, dtype);
            }
            return IoData{{std::move(x)}, {std::move(y)}};
          },
          "backend-test", TestCaseTag::NONE, {}, 1e-3, 0.0);
    }
  }

  {
    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    Expect(
        registry, std::move(node), "test_cc_reference_logsoftmax_tied_maxima", {DefaultOpset(13)},
        []() -> IoData {
          const double inf = std::numeric_limits<double>::infinity();
          Tensor x = Tensor::FromDouble("", {3, 3}, {0, 0, -200, 4, 4, 4, 0, -inf, -inf});
          Tensor y =
              Tensor::FromDouble("", {3, 3},
                                 {-std::log(2.0), -std::log(2.0), -200 - std::log(2.0),
                                  -std::log(3.0), -std::log(3.0), -std::log(3.0), 0, -inf, -inf});
          return IoData{{std::move(x)}, {std::move(y)}};
        },
        "backend-test", TestCaseTag::NONE, {}, 1e-3, 0.0);
  }

  for (const bool empty : {false, true}) {
    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    auto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);
    Expect(
        registry, std::move(node),
        empty ? "test_cc_reference_logsoftmax_empty_input"
              : "test_cc_reference_logsoftmax_single_class",
        {DefaultOpset(13)},
        [empty]() -> IoData {
          if (empty) {
            return IoData{{Tensor::FromFloat("", {0, 3}, {})}, {Tensor::FromFloat("", {0, 3}, {})}};
          }
          return IoData{{Tensor::FromFloat("", {2, 1, 3}, {1, -2, 3, 4, 5, -6})},
                        {Tensor::FromFloat("", {2, 1, 3}, std::vector<float>(6, 0))}};
        },
        "backend-test", TestCaseTag::NONE, {}, 1e-3, 0.0);
  }

  {
    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    Expect(registry, std::move(node), "test_cc_reference_logsoftmax_nonfinite", {DefaultOpset(13)},
           []() -> IoData {
             const double inf = std::numeric_limits<double>::infinity();
             const double nan = std::numeric_limits<double>::quiet_NaN();
             return IoData{{Tensor::FromDouble("", {3, 2}, {inf, 0, -inf, -inf, nan, 0})},
                           {Tensor::FromDouble("", {3, 2}, std::vector<double>(6, nan))}};
           });
  }

  for (const auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    const std::string suffix = dtype == DataType::FLOAT16 ? "float16" : "bfloat16";
    Expect(registry, std::move(node), "test_cc_logsoftmax_many_classes_" + suffix,
           {DefaultOpset(13)}, [dtype]() -> IoData {
             constexpr int64_t classes = 65536;
             Tensor input = core::runtime::DemoteFromFloat32(
                 Tensor::FromFloat("", {1, classes}, std::vector<float>(classes, 0.0f)), dtype);
             Tensor output = core::runtime::DemoteFromFloat32(
                 Tensor::FromFloat(
                     "", {1, classes},
                     std::vector<float>(classes, -std::log(static_cast<float>(classes)))),
                 dtype);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }

  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);

    Expect(registry, std::move(node), "test_cc_logsoftmax_example_1", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext logsoftmax_kernel_ctx{opset};
      const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{logsoftmax_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = logsoftmax_kernel(x, 1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    Expect(registry, std::move(node), "test_cc_logsoftmax_default_axis", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext logsoftmax_kernel_ctx{opset};
      const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{logsoftmax_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = logsoftmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(-1);

    Expect(registry, std::move(node), "test_cc_logsoftmax_negative_axis", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext logsoftmax_kernel_ctx{opset};
      const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{logsoftmax_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = logsoftmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(0);

    Expect(registry, std::move(node), "test_cc_logsoftmax_axis_0", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext logsoftmax_kernel_ctx{opset};
      const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{logsoftmax_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = logsoftmax_kernel(x, 0);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);

    Expect(registry, std::move(node), "test_cc_logsoftmax_axis_1", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext logsoftmax_kernel_ctx{opset};
      const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{logsoftmax_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = logsoftmax_kernel(x, 1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(2);

    Expect(registry, std::move(node), "test_cc_logsoftmax_axis_2", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext logsoftmax_kernel_ctx{opset};
      const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{logsoftmax_kernel_ctx};

      Tensor x = Tensor::FromFloat(
          "", {2, 2, 3},
          {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 1.5f, 0.0f});
      Tensor y = logsoftmax_kernel(x, 2);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(23);
    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(-1);
    Expect(registry, std::move(node), "test_cc_logsoftmax_large_finite_gap", {opset},
           []() -> IoData {
             // log(softmax(x)) would underflow to -inf for the second class.
             Tensor x = Tensor::FromFloat("", {1, 2}, {0.0f, -104.0f});
             Tensor y = Tensor::FromFloat("", {1, 2}, {0.0f, -104.0f});
             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    Expect(registry, std::move(node), "test_cc_logsoftmax_large_number", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext logsoftmax_kernel_ctx{opset};
      const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{logsoftmax_kernel_ctx};

      // Numerical stability check: large inputs should not overflow.
      Tensor x =
          Tensor::FromFloat("", {2, 3}, {1000.0f, 1001.0f, 1002.0f, 1002.0f, 1001.0f, 1000.0f});
      Tensor y = logsoftmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
