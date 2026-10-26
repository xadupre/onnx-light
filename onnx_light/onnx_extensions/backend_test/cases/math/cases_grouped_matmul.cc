// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterGroupedMatMulCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    return;
  }
  auto add = [&](const std::string &name, const Tensors &inputs, const Tensor &output,
                 bool explicit_empty = false) {
    NodeProto node;
    node.set_op_type("GroupedMatMul");
    for (const char *input : {"input", "weights", "group_indices"}) {
      node.add_input(input);
    }
    if (inputs.size() == 4 || explicit_empty) {
      node.add_input(explicit_empty ? "" : "bias");
    }
    node.add_output("output");
    Expect(node, inputs, {output}, name, {DefaultOpset(29)}, "backend-test", registry);
  };

  const Tensor input = Tensor::FromFloat("", {4, 3}, {1, 0, -1, 0, 1, 2, 1, 1, 0, 0, 0, 1});
  const Tensor weights = Tensor::FromFloat("", {2, 3, 2}, {1, 0, 0, 1, -1, 0, 0, 1, 1, 0, 0, 1});
  const Tensor indices = Tensor::FromInt64("", {4, 1}, {0, 1, 0, 1});
  const Tensor output = Tensor::FromFloat("", {4, 1, 2}, {2, 0, 1, 2, 1, 1, 0, 1});
  add("test_cc_groupedmatmul", {input, weights, indices}, output);
  add("test_cc_groupedmatmul_without_bias_explicit_empty_input", {input, weights, indices}, output,
      true);

  for (const int dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    auto make = [dtype](const Shape &shape, const std::vector<float> &values) {
      switch (dtype) {
      case DataType::DOUBLE:
        return Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
      case DataType::FLOAT16:
        return MakeFloat16Tensor("", shape, values);
      case DataType::BFLOAT16:
        return MakeBfloat16Tensor("", shape, values);
      default:
        return Tensor::FromFloat("", shape, values);
      }
    };
    for (const bool int32_indices : {false, true}) {
      if (dtype == DataType::FLOAT16 || dtype == DataType::DOUBLE) {
        add(std::string("test_cc_groupedmatmul_") +
                (dtype == DataType::FLOAT16 ? "float16" : "float64") +
                (int32_indices ? "_int32" : "_int64"),
            {make({2, 2}, {1, 2, 3, 4}), make({2, 2, 2}, {1, 0, 0, 1, 0, 1, 1, 0}),
             int32_indices ? Tensor::FromInt32("", {2, 2}, {0, 1, 1, 0})
                           : Tensor::FromInt64("", {2, 2}, {0, 1, 1, 0})},
            make({2, 2, 2}, {1, 2, 2, 1, 4, 3, 3, 4}));
      }
      add("test_cc_groupedmatmul_with_bias_" + std::to_string(dtype) +
              (int32_indices ? "_int32" : "_int64"),
          {make({2, 2}, {1, 2, 3, 4}), make({2, 2, 2}, {1, 0, 0, 1, 0, 1, 1, 0}),
           int32_indices ? Tensor::FromInt32("", {2, 2}, {0, 1, 1, 0})
                         : Tensor::FromInt64("", {2, 2}, {0, 1, 1, 0}),
           make({2, 2}, {0.5f, 1, 1.5f, 2})},
          make({2, 2, 2}, {1.5f, 3, 3.5f, 3, 5.5f, 5, 3.5f, 5}));
    }
  }

  add("test_cc_groupedmatmul_with_unused_group",
      {Tensor::FromFloat("", {4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}),
       Tensor::FromFloat("", {3, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
       Tensor::FromInt64("", {4, 1}, {0, 0, 2, 2})},
      Tensor::FromFloat("", {4, 1, 2}, {4, 7, 8, 15, 100, 111, 136, 151}));
  add("test_cc_groupedmatmul_single_group",
      {Tensor::FromFloat("", {3, 2}, {1, 2, 3, 4, 5, 6}),
       Tensor::FromFloat("", {1, 2, 3}, {1, 2, 3, 4, 5, 6}),
       Tensor::FromInt64("", {3, 1}, {0, 0, 0})},
      Tensor::FromFloat("", {3, 1, 3}, {9, 12, 15, 19, 26, 33, 29, 40, 51}));
  for (const bool zero_tokens : {false, true}) {
    for (const bool bias : {false, true}) {
      const int64_t m = zero_tokens ? 0 : 2;
      const int64_t s = zero_tokens ? 2 : 0;
      Tensors inputs{Tensor::FromFloat("", {m, 3}, std::vector<float>(m * 3)),
                     Tensor::FromFloat("", {2, 3, 4}, std::vector<float>(24)),
                     Tensor::FromInt64("", {m, s}, {})};
      if (bias) {
        inputs.push_back(Tensor::FromFloat("", {2, 4}, std::vector<float>(8)));
      }
      add(std::string("test_cc_groupedmatmul_zero_") + (zero_tokens ? "tokens" : "selections") +
              (bias ? "_with_bias" : ""),
          inputs, Tensor::FromFloat("", {m, s, 4}, {}));
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
