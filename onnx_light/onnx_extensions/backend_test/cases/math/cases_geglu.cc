// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
namespace {
template <typename T> T ExpectedGeGLU(T a, T b, bool approximate) {
  const T activation =
      approximate ? T{0.5} * a *
                        (T{1} + std::tanh(static_cast<T>(0.7978845608028654L) *
                                          (a + static_cast<T>(0.044715) * a * a * a)))
                  : T{0.5} * a * (T{1} + std::erf(a * static_cast<T>(1.0L / 1.4142135623730951L)));
  return activation * b;
}

Tensor MakeTensor(DataType dtype, const Shape &shape, const std::vector<float> &values) {
  if (dtype == DataType::FLOAT16)
    return MakeFloat16Tensor("", shape, values);
  if (dtype == DataType::BFLOAT16)
    return MakeBfloat16Tensor("", shape, values);
  if (dtype == DataType::DOUBLE)
    return Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  return Tensor::FromFloat("", shape, values);
}
} // namespace

void RegisterGeGLUCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(29);
  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkBinaryFloat<onnx_kernels::kernel::GeGLU>("GeGLU", "test_cc_geglu_benchmark",
                                                            opset, registry);
    return;
  }
  for (const DataType dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const std::string approximate : {"", "none", "tanh"}) {
      for (const Shape &shape : {Shape{2, 4}, Shape{2, 2, 2}, Shape{}, Shape{2, 0, 3}}) {
        NodeProto node;
        node.set_op_type("GeGLU");
        node.add_input("A");
        node.add_input("B");
        node.add_output("Y");
        if (!approximate.empty()) {
          auto *attr = node.add_attribute();
          attr->set_name("approximate");
          attr->set_type(AttributeProto::STRING);
          attr->set_s(approximate);
        }
        const std::string name = "test_cc_geglu_" + std::to_string(static_cast<int32_t>(dtype)) +
                                 "_" + (approximate.empty() ? "default" : approximate) +
                                 (shape.empty()          ? "_scalar"
                                  : shape.product() == 0 ? "_empty"
                                  : shape.size() == 3    ? "_rank3"
                                                         : "");
        Expect(registry, std::move(node), name, {opset}, [dtype, approximate, shape]() -> IoData {
          const std::vector<float> a = shape.empty() ? std::vector<float>{-4.0f}
                                       : shape.product() == 0
                                           ? std::vector<float>{}
                                           : std::vector<float>{-4, -3, -1, -0.5f, 0, 0.5f, 1, 4};
          const std::vector<float> b = shape.empty() ? std::vector<float>{2.0f}
                                       : shape.product() == 0
                                           ? std::vector<float>{}
                                           : std::vector<float>{2, -1, 0.5f, 3, -2, -3, 0.5f, 1};
          std::vector<float> y(a.size());
          std::vector<double> yd(a.size());
          for (size_t i = 0; i < a.size(); ++i) {
            y[i] = ExpectedGeGLU(a[i], b[i], approximate == "tanh");
            yd[i] = ExpectedGeGLU(static_cast<double>(a[i]), static_cast<double>(b[i]),
                                  approximate == "tanh");
          }
          Tensor expected = dtype == DataType::DOUBLE ? Tensor::FromDouble("", shape, yd)
                                                      : MakeTensor(dtype, shape, y);
          return IoData{{MakeTensor(dtype, shape, a), MakeTensor(dtype, shape, b)},
                        {std::move(expected)}};
        });
      }
    }
  }
}
} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
