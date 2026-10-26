// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace core::backend_test;

namespace {

template <typename T> std::vector<T> CastValues(const std::vector<double> &values) {
  std::vector<T> converted;
  converted.reserve(values.size());
  for (double value : values) {
    converted.push_back(static_cast<T>(value));
  }
  return converted;
}

Tensor TypedTensor(DataType dtype, const Shape &shape, const std::vector<double> &values) {
  switch (dtype) {
  case DataType::DOUBLE:
    return Tensor::FromDouble("", shape, values);
  case DataType::FLOAT16:
    return MakeFloat16Tensor("", shape, CastValues<float>(values));
  case DataType::BFLOAT16:
    return MakeBfloat16Tensor("", shape, CastValues<float>(values));
  case DataType::INT32:
    return Tensor::FromInt32("", shape, CastValues<int32_t>(values));
  case DataType::INT64:
    return Tensor::FromInt64("", shape, CastValues<int64_t>(values));
  case DataType::UINT32:
    return Tensor::FromUint32("", shape, CastValues<uint32_t>(values));
  case DataType::UINT64:
    return Tensor::FromUint64("", shape, CastValues<uint64_t>(values));
  default:
    throw std::invalid_argument("Unsupported correctness-case numeric dtype.");
  }
}

std::string DtypeCaseName(const std::string &op, DataType dtype) {
  return "test_cc_supported_dtype_" + op + "_" +
         TensorProto::DataType_Name(static_cast<TensorProto::DataType>(dtype));
}

void ExpectTyped(std::vector<TestCase> &registry, NodeProto node, int64_t opset, DataType dtype,
                 std::function<IoData()> make_io) {
  const std::string name = DtypeCaseName(node.op_type().value(), dtype);
  const double rtol = dtype == DataType::BFLOAT16  ? 0.016
                      : dtype == DataType::FLOAT16 ? 0.002
                                                   : 1e-12;
  Expect(registry, std::move(node), name, {DefaultOpset(opset)}, std::move(make_io), "backend-test",
         TestCaseTag::NONE, {}, rtol, 1e-14);
}

struct UnaryCase {
  const char *op;
  int64_t opset;
  std::vector<DataType> types;
  std::vector<double> input;
  double (*reference)(double);
};

void CollectUnaryCases(std::vector<TestCase> &registry, const std::string &op_type) {
  // Inputs are exactly representable in both half formats, so the independent
  // scalar reference sees the same values as the runtime.
  static const std::vector<UnaryCase> entries = {
      {"Acos",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-1, -0.5, 0, 0.5, 1},
       [](double x) { return std::acos(x); }},
      {"Acosh",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {1, 1.5, 2, 4},
       [](double x) { return std::acosh(x); }},
      {"Asin",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-1, -0.5, 0, 0.5, 1},
       [](double x) { return std::asin(x); }},
      {"Asinh",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-4, -0.5, 0, 0.5, 4},
       [](double x) { return std::asinh(x); }},
      {"Atan",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-4, -0.5, 0, 0.5, 4},
       [](double x) { return std::atan(x); }},
      {"Atanh",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-0.75, -0.5, 0, 0.5, 0.75},
       [](double x) { return std::atanh(x); }},
      {"Cos",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-2, -0.5, 0, 0.5, 2},
       [](double x) { return std::cos(x); }},
      {"Cosh",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-2, -0.5, 0, 0.5, 2},
       [](double x) { return std::cosh(x); }},
      {"HardSwish",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-4, -3, -1, 0, 1, 3, 4},
       [](double x) { return x * std::clamp(x / 6 + 0.5, 0.0, 1.0); }},
      {"Round",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.25},
       [](double x) {
         const double lower = std::floor(x);
         const double fraction = x - lower;
         return fraction < 0.5 || (fraction == 0.5 && std::fmod(lower, 2) == 0) ? lower : lower + 1;
       }},
      {"Sin",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-2, -0.5, 0, 0.5, 2},
       [](double x) { return std::sin(x); }},
      {"Sinh",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-2, -0.5, 0, 0.5, 2},
       [](double x) { return std::sinh(x); }},
      {"Tan",
       22,
       {DataType::FLOAT16, DataType::BFLOAT16},
       {-1, -0.5, 0, 0.5, 1},
       [](double x) { return std::tan(x); }},
      {"Ceil",
       13,
       {DataType::DOUBLE},
       {-2.75, -0.25, 0, 0.25, 2.75},
       [](double x) { return std::ceil(x); }},
      {"Erf", 13, {DataType::DOUBLE}, {-3, -0.5, 0, 0.5, 3}, [](double x) { return std::erf(x); }},
      {"Exp",
       13,
       {DataType::DOUBLE},
       {-10, -0.5, 0, 0.5, 10},
       [](double x) { return std::exp(x); }},
      {"Floor",
       13,
       {DataType::DOUBLE},
       {-2.75, -0.25, 0, 0.25, 2.75},
       [](double x) { return std::floor(x); }},
      {"Log", 13, {DataType::DOUBLE}, {0.125, 0.5, 1, 2, 8}, [](double x) { return std::log(x); }},
      {"Reciprocal",
       13,
       {DataType::DOUBLE},
       {-8, -0.5, 0.125, 1, 4},
       [](double x) { return 1 / x; }},
      {"Sign",
       13,
       {DataType::DOUBLE},
       {-10, -0.5, 0, 0.5, 10},
       [](double x) { return x > 0   ? 1.0
                             : x < 0 ? -1.0
                                     : 0.0; }},
      {"Softplus",
       13,
       {DataType::DOUBLE},
       {-10, -0.5, 0, 0.5, 10},
       [](double x) { return std::log1p(std::exp(x)); }},
      {"Softsign",
       22,
       {DataType::DOUBLE},
       {-10, -0.5, 0, 0.5, 10},
       [](double x) { return x / (1 + std::abs(x)); }},
      {"Sqrt", 13, {DataType::DOUBLE}, {0, 0.25, 1, 2, 16}, [](double x) { return std::sqrt(x); }},
      {"Mish",
       22,
       {DataType::BFLOAT16, DataType::DOUBLE, DataType::FLOAT16},
       {-4, -0.5, 0, 0.5, 4},
       [](double x) { return x * std::tanh(std::log1p(std::exp(x))); }},
  };
  for (const UnaryCase &entry : entries) {
    if (!op_type.empty() && op_type != entry.op) {
      continue;
    }
    for (DataType dtype : entry.types) {
      ExpectTyped(registry, MakeNode(entry.op, {"X"}, {"Y"}), entry.opset, dtype,
                  [entry, dtype]() -> IoData {
                    std::vector<double> expected;
                    expected.reserve(entry.input.size());
                    for (double value : entry.input) {
                      expected.push_back(entry.reference(value));
                    }
                    const Shape shape{static_cast<int64_t>(entry.input.size())};
                    return {{TypedTensor(dtype, shape, entry.input)},
                            {TypedTensor(dtype, shape, expected)}};
                  });
    }
  }
}

void CollectMatrixCases(std::vector<TestCase> &registry, const std::string &op_type) {
  for (const std::string op : {"MatMul", "PRelu", "Gemm"}) {
    if (!op_type.empty() && op_type != op) {
      continue;
    }
    std::vector<DataType> types{DataType::BFLOAT16, DataType::DOUBLE, DataType::FLOAT16};
    if (op != "Gemm") {
      types.insert(types.end(),
                   {DataType::INT32, DataType::INT64, DataType::UINT32, DataType::UINT64});
    }
    for (DataType dtype : types) {
      NodeProto node = MakeNode(op.c_str(), {"A", "B"}, {"Y"});
      if (op == "Gemm") {
        node.add_input("C");
        AddAttribute<float>(node, "alpha", 0.5f);
        AddAttribute<float>(node, "beta", 2.0f);
      }
      ExpectTyped(
          registry, std::move(node), op == "PRelu" ? 16 : 13, dtype, [op, dtype]() -> IoData {
            if (op == "PRelu") {
              const bool is_unsigned = dtype == DataType::UINT32 || dtype == DataType::UINT64;
              const std::vector<double> input = is_unsigned
                                                    ? std::vector<double>{0, 1, 4, 8, 16, 32}
                                                    : std::vector<double>{-4, -2, 0, 1, 2, 4};
              const std::vector<double> expected =
                  is_unsigned ? input : std::vector<double>{-8, -6, 0, 1, 2, 4};
              return {{TypedTensor(dtype, {2, 3}, input), TypedTensor(dtype, {3}, {2, 3, 4})},
                      {TypedTensor(dtype, {2, 3}, expected)}};
            }
            // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]].
            Tensors inputs{TypedTensor(dtype, {2, 3}, {1, 2, 3, 4, 5, 6}),
                           TypedTensor(dtype, {3, 2}, {1, 2, 3, 4, 5, 6})};
            if (op == "Gemm") {
              inputs.emplace_back(TypedTensor(dtype, {2}, {1, -2}));
            }
            return {std::move(inputs),
                    {TypedTensor(dtype, {2, 2},
                                 op == "Gemm" ? std::vector<double>{13, 10, 26.5, 28}
                                              : std::vector<double>{22, 28, 49, 64})}};
          });
    }
  }
}

void CollectReductionCases(std::vector<TestCase> &registry, const std::string &op_type) {
  for (const std::string op : {"ReduceSum", "ReduceMean", "Softmax", "LogSoftmax", "GlobalLpPool",
                               "MeanVarianceNormalization"}) {
    if (!op_type.empty() && op_type != op) {
      continue;
    }
    std::vector<DataType> types{DataType::DOUBLE};
    if (op != "ReduceSum" && op != "MeanVarianceNormalization") {
      types.push_back(DataType::FLOAT16);
      if (op != "LogSoftmax") {
        types.push_back(DataType::BFLOAT16);
      }
    }
    for (DataType dtype : types) {
      NodeProto node = MakeNode(op.c_str(), {"X"}, {"Y"});
      int64_t opset = 22;
      if (op == "ReduceSum" || op == "ReduceMean") {
        node.add_input("axes");
        AddAttribute<int64_t>(node, "keepdims", 0);
      } else if (op == "Softmax" || op == "LogSoftmax") {
        AddAttribute<int64_t>(node, "axis", -1);
      } else if (op == "GlobalLpPool") {
        AddAttribute<int64_t>(node, "p", 2);
      } else {
        opset = 13;
        AddAttribute<std::vector<int64_t>>(node, "axes", {1});
      }
      ExpectTyped(registry, std::move(node), opset, dtype, [op, dtype]() -> IoData {
        if (op == "GlobalLpPool") {
          return {{TypedTensor(dtype, {1, 2, 2}, {3, -4, 0, 5})},
                  {TypedTensor(dtype, {1, 2, 1}, {5, 5})}};
        }
        if (op == "MeanVarianceNormalization") {
          return {{TypedTensor(dtype, {2, 2}, {1000, 3000, 2000, 6000})},
                  {TypedTensor(dtype, {2, 2}, {-1, 1, -1, 1})}};
        }
        if (op == "ReduceSum" || op == "ReduceMean") {
          return {
              {TypedTensor(dtype, {2, 3}, {-2, 0, 5, 4, -1, 6}), Tensor::FromInt64("", {1}, {1})},
              {TypedTensor(dtype, {2},
                           op == "ReduceSum" ? std::vector<double>{3, 9}
                                             : std::vector<double>{1, 3})}};
        }
        const double sum = 1 + std::exp(1.0) + std::exp(2.0);
        std::vector<double> row =
            op == "Softmax"
                ? std::vector<double>{1 / sum, std::exp(1.0) / sum, std::exp(2.0) / sum}
                : std::vector<double>{-std::log(sum), 1 - std::log(sum), 2 - std::log(sum)};
        std::vector<double> expected = row;
        expected.insert(expected.end(), row.begin(), row.end());
        return {{TypedTensor(dtype, {2, 3}, {-1, 0, 1, 7, 8, 9})},
                {TypedTensor(dtype, {2, 3}, expected)}};
      });
    }
  }
}

void CollectOtherNumericCases(std::vector<TestCase> &registry, const std::string &op_type) {
  if (op_type.empty() || op_type == "Conv") {
    ExpectTyped(registry, MakeNode("Conv", {"X", "W", "B"}, {"Y"}), 22, DataType::BFLOAT16,
                []() -> IoData {
                  return {{TypedTensor(DataType::BFLOAT16, {1, 1, 4}, {1, -2, 3, 4}),
                           TypedTensor(DataType::BFLOAT16, {1, 1, 2}, {2, -1}),
                           TypedTensor(DataType::BFLOAT16, {1}, {1})},
                          {TypedTensor(DataType::BFLOAT16, {1, 1, 3}, {5, -6, 3})}};
                });
  }
  if (op_type.empty() || op_type == "SwiGLU") {
    ExpectTyped(registry, MakeNode("SwiGLU", {"A", "B"}, {"Y"}), 28, DataType::DOUBLE,
                []() -> IoData {
                  return {{TypedTensor(DataType::DOUBLE, {4}, {-2, 0, 1, 2}),
                           TypedTensor(DataType::DOUBLE, {4}, {3, -1, 2, -2})},
                          {TypedTensor(DataType::DOUBLE, {4},
                                       {-6 / (1 + std::exp(2.0)), 0, 2 / (1 + std::exp(-1.0)),
                                        -4 / (1 + std::exp(-2.0))})}};
                });
  }
  for (const std::string op : {"BitwiseAnd", "BitwiseOr", "BitwiseXor"}) {
    if (!op_type.empty() && op_type != op) {
      continue;
    }
    ExpectTyped(
        registry, MakeNode(op.c_str(), {"A", "B"}, {"Y"}), 18, DataType::UINT16, [op]() -> IoData {
          return {{Tensor::FromUint16("", {2, 2}, {0, 65535, 43690, 21845}),
                   Tensor::FromUint16("", {2}, {3855, 255})},
                  {Tensor::FromUint16("", {2, 2},
                                      op == "BitwiseAnd" ? std::vector<uint16_t>{0, 255, 2570, 85}
                                      : op == "BitwiseOr"
                                          ? std::vector<uint16_t>{3855, 65535, 44975, 22015}
                                          : std::vector<uint16_t>{3855, 65280, 42405, 21930})}};
        });
  }
}

void CollectTransposeCases(std::vector<TestCase> &registry, const std::string &op_type) {
  if (!op_type.empty() && op_type != "Transpose") {
    return;
  }
  for (DataType dtype : {DataType::BFLOAT16, DataType::BOOL, DataType::DOUBLE, DataType::FLOAT16,
                         DataType::FLOAT8E4M3FN, DataType::FLOAT8E4M3FNUZ, DataType::FLOAT8E5M2,
                         DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0, DataType::INT16,
                         DataType::INT32, DataType::INT64, DataType::INT8, DataType::UINT16,
                         DataType::UINT32, DataType::UINT64, DataType::UINT8}) {
    NodeProto node = MakeNode("Transpose", {"X"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "perm", {1, 0});
    ExpectTyped(registry, std::move(node), 25, dtype, [dtype]() -> IoData {
      const size_t width = ElementSize(dtype);
      Tensor input = MakeOutputTensor(dtype, {2, 3}, 6 * width, nullptr);
      Tensor expected = MakeOutputTensor(dtype, {3, 2}, 6 * width, nullptr);
      const uint8_t patterns[] = {0x00, 0x01, 0x55, 0x7f, 0x80, 0xaa};
      for (size_t i = 0; i < 6; ++i) {
        std::memset(input.mutable_bytes() + i * width,
                    dtype == DataType::BOOL ? i % 2 : patterns[i], width);
      }
      const size_t indices[] = {0, 3, 1, 4, 2, 5};
      for (size_t i = 0; i < 6; ++i) {
        std::memcpy(expected.mutable_bytes() + i * width, input.bytes() + indices[i] * width,
                    width);
      }
      return {{std::move(input)}, {std::move(expected)}};
    });
  }
}

} // namespace

ONNX_LIGHT_BACKEND_TEST_LOCAL void CollectSupportedTypeTestCases(std::vector<TestCase> &registry,
                                                                 const std::string &op_type,
                                                                 TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    return;
  }
  CollectUnaryCases(registry, op_type);
  CollectMatrixCases(registry, op_type);
  CollectReductionCases(registry, op_type);
  CollectOtherNumericCases(registry, op_type);
  CollectTransposeCases(registry, op_type);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
