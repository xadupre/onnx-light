// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
namespace {

NodeProto MakeLogCumSumExpNode(int64_t axis, bool default_axis = false) {
  NodeProto node;
  node.set_op_type("LogCumSumExp");
  node.add_input("x");
  node.add_output("y");
  if (!default_axis) {
    auto *attr = node.add_attribute();
    attr->set_name("axis");
    attr->set_type(AttributeProto::INT);
    attr->set_i(axis);
  }
  return node;
}

Tensor MakeTensor(int32_t dtype, const Shape &shape, const std::vector<float> &values) {
  switch (dtype) {
  case DataType::FLOAT16:
    return MakeFloat16Tensor("", shape, values);
  case DataType::BFLOAT16:
    return MakeBfloat16Tensor("", shape, values);
  case DataType::DOUBLE:
    return Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  default:
    return Tensor::FromFloat("", shape, values);
  }
}

} // namespace

void RegisterLogCumSumExpCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(29);
  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::LogCumSumExp>(
        "LogCumSumExp", "test_cc_logcumsumexp_benchmark", opset, registry);
    return;
  }
  for (const int32_t dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    Expect(registry, MakeLogCumSumExpNode(0, true),
           "test_cc_logcumsumexp_default_dtype_" + std::to_string(dtype), {opset},
           [dtype]() -> IoData {
             Tensor x = MakeTensor(dtype, {4}, {0, 0, 0, 0});
             Tensor y =
                 dtype == DataType::DOUBLE
                     ? Tensor::FromDouble(
                           "", {4},
                           {0, std::log(2.0), std::log(2.0) + std::log1p(0.5), std::log(4.0)})
                     : MakeTensor(dtype, {4}, {0, std::log(2.0f), std::log(3.0f), std::log(4.0f)});
             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
  for (const int64_t axis : {int64_t{0}, int64_t{1}, int64_t{-1}, int64_t{-2}}) {
    Expect(registry, MakeLogCumSumExpNode(axis),
           "test_cc_logcumsumexp_axis_" + std::to_string(axis), {opset}, [axis]() -> IoData {
             Tensor x = Tensor::FromDouble("", {2, 3}, {0, 0, 0, 0, 0, 0});
             const double l2 = std::log(2.0), l3 = l2 + std::log1p(0.5);
             Tensor y = Tensor::FromDouble("", {2, 3},
                                           axis == 0 || axis == -2
                                               ? std::vector<double>{0, 0, 0, l2, l2, l2}
                                               : std::vector<double>{0, l2, l3, 0, l2, l3});
             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
  Expect(registry, MakeLogCumSumExpNode(-1), "test_cc_logcumsumexp_dynamic_range", {opset},
         []() -> IoData {
           Tensor x = Tensor::FromDouble("", {2, 3}, {1000, 1000, -1000, -1000, -1000, 1000});
           const double high = 1000 + std::log(2.0), low = -1000 + std::log(2.0);
           Tensor y = Tensor::FromDouble("", {2, 3}, {1000, high, high, -1000, low, 1000});
           return IoData{{std::move(x)}, {std::move(y)}};
         });
  Expect(registry, MakeLogCumSumExpNode(0), "test_cc_logcumsumexp_special_values", {opset},
         []() -> IoData {
           const float inf = std::numeric_limits<float>::infinity();
           const float nan = std::numeric_limits<float>::quiet_NaN();
           Tensor x = Tensor::FromFloat("", {6}, {-inf, -inf, 0, inf, nan, 1});
           Tensor y = Tensor::FromFloat("", {6}, {-inf, -inf, 0, inf, nan, nan});
           return IoData{{std::move(x)}, {std::move(y)}};
         });
  for (const int64_t axis : {int64_t{0}, int64_t{1}, int64_t{2}}) {
    Expect(registry, MakeLogCumSumExpNode(axis),
           "test_cc_logcumsumexp_empty_axis_" + std::to_string(axis), {opset}, []() -> IoData {
             Tensor x = Tensor::FromFloat("", {2, 0, 3}, {});
             Tensor y = Tensor::FromFloat("", {2, 0, 3}, {});
             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
