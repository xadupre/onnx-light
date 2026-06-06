// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Split — y_0, y_1, ... = split(input[, split], axis[, num_outputs])
// Mirrors the upstream ONNX node tests in
// ``onnx/backend/test/case/node/split.py``: equal/variable parts (1-D and 2-D),
// default axis, zero-size splits, uneven splits with ``num_outputs``.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeSplitNode(const std::vector<std::string> &output_names, int64_t axis, bool has_axis,
                        int64_t num_outputs = 0, bool has_split_input = false) {
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("input");
  if (has_split_input) {
    node.add_input("split");
  }
  for (const std::string &n : output_names) {
    node.add_output(n);
  }
  if (has_axis) {
    AddAttribute<int64_t>(node, "axis", axis);
  }
  if (num_outputs > 0) {
    AddAttribute<int64_t>(node, "num_outputs", num_outputs);
  }
  return node;
}

Tensor MakeInt64Vector(const std::vector<int64_t> &values) {
  Tensor t;
  t.name = "";
  t.data_type = static_cast<int32_t>(DataType::INT64);
  t.shape = {static_cast<int64_t>(values.size())};
  t.data.assign(reinterpret_cast<const uint8_t *>(values.data()),
                reinterpret_cast<const uint8_t *>(values.data() + values.size()));
  return t;
}

} // namespace

void RegisterSplitCases(std::vector<TestCase> &registry) {
  const OpsetId opset13 = DefaultOpset(13);
  const OpsetId opset18 = DefaultOpset(18);
  const kernel::Split split_kernel{kernel::KernelContext{opset18}};

  // ---- opset 13: equal parts, 1-D, axis=0 ----
  {
    Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    NodeProto node = MakeSplitNode({"output_1", "output_2", "output_3"}, /*axis=*/0,
                                   /*has_axis=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{}, /*num_outputs=*/3);
    Expect(node, {x}, outs, "test_cc_split_equal_parts_1d_opset13", {opset13}, "backend-test",
           registry);
  }

  // ---- opset 13: variable parts, 1-D, axis=0, split input [2, 4] ----
  {
    Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor split_in = MakeInt64Vector({2, 4});
    NodeProto node = MakeSplitNode({"output_1", "output_2"}, /*axis=*/0, /*has_axis=*/true,
                                   /*num_outputs=*/0, /*has_split_input=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{2, 4}, /*num_outputs=*/0);
    Expect(node, {x, split_in}, outs, "test_cc_split_variable_parts_1d_opset13", {opset13},
           "backend-test", registry);
  }

  // ---- opset 13: equal parts, 2-D, axis=1 ----
  {
    Tensor x = Tensor::FromFloat(
        "", {2, 6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    NodeProto node = MakeSplitNode({"output_1", "output_2"}, /*axis=*/1, /*has_axis=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/1, /*split=*/{}, /*num_outputs=*/2);
    Expect(node, {x}, outs, "test_cc_split_equal_parts_2d_opset13", {opset13}, "backend-test",
           registry);
  }

  // ---- opset 13: variable parts, 2-D, axis=1, split [2, 4] ----
  {
    Tensor x = Tensor::FromFloat(
        "", {2, 6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor split_in = MakeInt64Vector({2, 4});
    NodeProto node = MakeSplitNode({"output_1", "output_2"}, /*axis=*/1, /*has_axis=*/true,
                                   /*num_outputs=*/0, /*has_split_input=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/1, /*split=*/{2, 4}, /*num_outputs=*/0);
    Expect(node, {x, split_in}, outs, "test_cc_split_variable_parts_2d_opset13", {opset13},
           "backend-test", registry);
  }

  // ---- opset 13: default axis (=0), equal parts, 1-D ----
  {
    Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    NodeProto node = MakeSplitNode({"output_1", "output_2", "output_3"}, /*axis=*/0,
                                   /*has_axis=*/false);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{}, /*num_outputs=*/3);
    Expect(node, {x}, outs, "test_cc_split_equal_parts_default_axis_opset13", {opset13},
           "backend-test", registry);
  }

  // ---- opset 13: default axis, variable parts ----
  {
    Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor split_in = MakeInt64Vector({2, 4});
    NodeProto node = MakeSplitNode({"output_1", "output_2"}, /*axis=*/0, /*has_axis=*/false,
                                   /*num_outputs=*/0, /*has_split_input=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{2, 4}, /*num_outputs=*/0);
    Expect(node, {x, split_in}, outs, "test_cc_split_variable_parts_default_axis_opset13",
           {opset13}, "backend-test", registry);
  }

  // ---- opset 13: zero-size splits ----
  {
    Tensor x = Tensor::FromFloat("", {0}, {});
    Tensor split_in = MakeInt64Vector({0, 0, 0});
    NodeProto node = MakeSplitNode({"output_1", "output_2", "output_3"}, /*axis=*/0,
                                   /*has_axis=*/false, /*num_outputs=*/0, /*has_split_input=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{0, 0, 0}, /*num_outputs=*/0);
    Expect(node, {x, split_in}, outs, "test_cc_split_zero_size_splits_opset13", {opset13},
           "backend-test", registry);
  }

  // ---- opset 18: equal parts via num_outputs, 1-D ----
  {
    Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    NodeProto node = MakeSplitNode({"output_1", "output_2", "output_3"}, /*axis=*/0,
                                   /*has_axis=*/true, /*num_outputs=*/3);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{}, /*num_outputs=*/3);
    Expect(node, {x}, outs, "test_cc_split_equal_parts_1d_opset18", {opset18}, "backend-test",
           registry);
  }

  // ---- opset 18: variable parts via split input, 1-D ----
  {
    Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor split_in = MakeInt64Vector({2, 4});
    NodeProto node = MakeSplitNode({"output_1", "output_2"}, /*axis=*/0, /*has_axis=*/true,
                                   /*num_outputs=*/0, /*has_split_input=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{2, 4}, /*num_outputs=*/0);
    Expect(node, {x, split_in}, outs, "test_cc_split_variable_parts_1d_opset18", {opset18},
           "backend-test", registry);
  }

  // ---- opset 18: equal parts via num_outputs, 2-D (axis=1, dim=6 / 2 = 3) ----
  {
    Tensor x = Tensor::FromFloat(
        "", {2, 6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    NodeProto node = MakeSplitNode({"output_1", "output_2"}, /*axis=*/1, /*has_axis=*/true,
                                   /*num_outputs=*/2);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/1, /*split=*/{}, /*num_outputs=*/2);
    Expect(node, {x}, outs, "test_cc_split_equal_parts_2d", {opset18}, "backend-test", registry);
  }

  // ---- opset 18: variable parts, 2-D, split [2,4] ----
  {
    Tensor x = Tensor::FromFloat(
        "", {2, 6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor split_in = MakeInt64Vector({2, 4});
    NodeProto node = MakeSplitNode({"output_1", "output_2"}, /*axis=*/1, /*has_axis=*/true,
                                   /*num_outputs=*/0, /*has_split_input=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/1, /*split=*/{2, 4}, /*num_outputs=*/0);
    Expect(node, {x, split_in}, outs, "test_cc_split_variable_parts_2d_opset18", {opset18},
           "backend-test", registry);
  }

  // ---- opset 18: default axis via num_outputs ----
  {
    Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    NodeProto node = MakeSplitNode({"output_1", "output_2", "output_3"}, /*axis=*/0,
                                   /*has_axis=*/false, /*num_outputs=*/3);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{}, /*num_outputs=*/3);
    Expect(node, {x}, outs, "test_cc_split_equal_parts_default_axis_opset18", {opset18},
           "backend-test", registry);
  }

  // ---- opset 18: variable parts, default axis ----
  {
    Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor split_in = MakeInt64Vector({2, 4});
    NodeProto node = MakeSplitNode({"output_1", "output_2"}, /*axis=*/0, /*has_axis=*/false,
                                   /*num_outputs=*/0, /*has_split_input=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{2, 4}, /*num_outputs=*/0);
    Expect(node, {x, split_in}, outs, "test_cc_split_variable_parts_default_axis_opset18",
           {opset18}, "backend-test", registry);
  }

  // ---- opset 18: zero-size splits via split input ----
  {
    Tensor x = Tensor::FromFloat("", {0}, {});
    Tensor split_in = MakeInt64Vector({0, 0, 0});
    NodeProto node = MakeSplitNode({"output_1", "output_2", "output_3"}, /*axis=*/0,
                                   /*has_axis=*/false, /*num_outputs=*/0, /*has_split_input=*/true);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{0, 0, 0}, /*num_outputs=*/0);
    Expect(node, {x, split_in}, outs, "test_cc_split_zero_size_splits_opset18", {opset18},
           "backend-test", registry);
  }

  // ---- opset 18: 1-D uneven split via num_outputs (axis_dim=7, n=4 -> [2,2,2,1]) ----
  {
    Tensor x = Tensor::FromFloat("", {7}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    NodeProto node = MakeSplitNode({"output_1", "output_2", "output_3", "output_4"}, /*axis=*/0,
                                   /*has_axis=*/false, /*num_outputs=*/4);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/0, /*split=*/{}, /*num_outputs=*/4);
    Expect(node, {x}, outs, "test_cc_split_1d_uneven_split_opset18", {opset18}, "backend-test",
           registry);
  }

  // ---- opset 18: 2-D uneven split via num_outputs (axis=1, dim=8, n=3 -> [3,3,2]) ----
  {
    Tensor x = Tensor::FromFloat("", {2, 8},
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    NodeProto node = MakeSplitNode({"output_1", "output_2", "output_3"}, /*axis=*/1,
                                   /*has_axis=*/true, /*num_outputs=*/3);
    std::vector<Tensor> outs = split_kernel(x, /*axis=*/1, /*split=*/{}, /*num_outputs=*/3);
    Expect(node, {x}, outs, "test_cc_split_2d_uneven_split_opset18", {opset18}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
