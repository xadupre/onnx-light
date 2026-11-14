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

namespace {

Tensor TypedTensor(DataType dtype, const Shape &shape, const std::vector<double> &values) {
  if (dtype == DataType::DOUBLE) {
    return Tensor::FromDouble("", shape, values);
  }
  std::vector<float> floats;
  floats.reserve(values.size());
  for (double value : values) {
    floats.push_back(static_cast<float>(value));
  }
  if (dtype == DataType::FLOAT16) {
    return MakeFloat16Tensor("", shape, floats);
  }
  if (dtype == DataType::BFLOAT16) {
    return MakeBfloat16Tensor("", shape, floats);
  }
  return Tensor::FromFloat("", shape, floats);
}

} // namespace

void RegisterExpm1Cases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(29);
  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Expm1>("Expm1", "test_cc_expm1_benchmark",
                                                           opset, registry);
    return;
  }

  const auto add_case = [&](const std::string &name, DataType dtype, const Shape &shape,
                            const std::vector<double> &values, double atol = 1e-7) {
    NodeProto node;
    node.set_op_type("Expm1");
    node.add_input("x");
    node.add_output("y");
    Expect(
        registry, std::move(node), name, {opset},
        [dtype, shape, values]() -> IoData {
          Tensor input = TypedTensor(dtype, shape, values);
          const Tensor rounded = PromoteToFloat32(input);
          std::vector<double> expected;
          expected.reserve(values.size());
          for (int64_t i = 0; i < input.element_count(); ++i) {
            const double value =
                dtype == DataType::DOUBLE ? input.AsDouble()[i] : rounded.AsFloat()[i];
            expected.push_back(std::expm1(value));
          }
          return {{std::move(input)}, {TypedTensor(dtype, shape, expected)}};
        },
        "backend-test", TestCaseTag::NONE, {}, dtype == DataType::BFLOAT16 ? 1e-2 : 1e-3, atol);
  };
  const double infinity = std::numeric_limits<double>::infinity();
  const std::string prefix = "test_cc_supported_dtype_Expm1_";
  for (DataType dtype :
       {DataType::FLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::DOUBLE}) {
    const std::string suffix = dtype == DataType::FLOAT16    ? "float16"
                               : dtype == DataType::BFLOAT16 ? "bfloat16"
                               : dtype == DataType::FLOAT    ? "float32"
                                                             : "float64";
    add_case(prefix + "finite_" + suffix, dtype, {2, 4}, {-10, -0.5, 0, 0.25, 0.5, 1, 3, 5});
    const double tiny = dtype == DataType::FLOAT16    ? 0x1p-14
                        : dtype == DataType::BFLOAT16 ? 0x1p-10
                        : dtype == DataType::FLOAT    ? 1e-8
                                                      : 1e-20;
    add_case(prefix + "small_inputs_" + suffix, dtype, {4}, {-tiny, -0.0, 0.0, tiny}, 0.0);
    add_case(prefix + "special_values_" + suffix, dtype, {5},
             {-0.0, 0.0, -infinity, infinity, std::numeric_limits<double>::quiet_NaN()});
    add_case(prefix + "overflow_" + suffix, dtype, {2}, {1000, -1000});
    add_case(prefix + "scalar_" + suffix, dtype, {}, {1.0});
    add_case(prefix + "empty_" + suffix, dtype, {0, 3}, {});
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
