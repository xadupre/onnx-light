// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeSliceNode(bool with_axes, bool with_steps) {
  NodeProto node;
  node.set_op_type("Slice");
  node.add_input("data");
  node.add_input("starts");
  node.add_input("ends");
  if (with_axes) {
    node.add_input("axes");
  }
  if (with_steps) {
    if (!with_axes) {
      node.add_input("");
    }
    node.add_input("steps");
  }
  node.add_output("output");
  return node;
}

Tensor MakeInt64VectorTensor(const std::vector<int64_t> &values) {
  std::vector<uint8_t> data(values.size() * sizeof(int64_t));
  if (!values.empty()) {
    std::memcpy(data.data(), values.data(), data.size());
  }
  return Tensor("", static_cast<int32_t>(DataType::INT64), {static_cast<int64_t>(values.size())},
                std::move(data));
}

// Builds a deterministic 20x10x5 float tensor with values ``i`` at flat
// index ``i``. Used by the ONNX-parity test cases below in lieu of the
// random data the upstream ``onnx`` reference cases use, so that the
// expected output can be computed once by the kernel itself.
Tensor MakeRangeTensor20x10x5() {
  std::vector<float> values(20 * 10 * 5);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i);
  }
  return Tensor::FromFloat("", {20, 10, 5}, values);
}

} // namespace

void RegisterSliceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Slice slice_kernel{ctx};

  {
    const Tensor data = Tensor::FromFloat("", {2, 4},
                                          {
                                              1.0f,
                                              2.0f,
                                              3.0f,
                                              4.0f,
                                              5.0f,
                                              6.0f,
                                              7.0f,
                                              8.0f,
                                          });
    const Tensor starts = MakeInt64VectorTensor({1, 0});
    const Tensor ends = MakeInt64VectorTensor({2, 3});
    const Tensor axes = MakeInt64VectorTensor({0, 1});
    const Tensor steps = MakeInt64VectorTensor({1, 2});
    const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
    Expect(MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           {data, starts, ends, axes, steps}, {output}, "test_cc_slice_axes_steps", {opset},
           "backend-test", registry);
  }

  {
    const Tensor data = Tensor::FromFloat("", {2, 4},
                                          {
                                              1.0f,
                                              2.0f,
                                              3.0f,
                                              4.0f,
                                              5.0f,
                                              6.0f,
                                              7.0f,
                                              8.0f,
                                          });
    const Tensor starts = MakeInt64VectorTensor({0, 1});
    const Tensor ends = MakeInt64VectorTensor({-1, 1000});
    const Tensor output = slice_kernel(data, starts, ends);
    Expect(MakeSliceNode(/*with_axes=*/false, /*with_steps=*/false), {data, starts, ends}, {output},
           "test_cc_slice_default_axes_steps", {opset}, "backend-test", registry);
  }

  // Mirrors ONNX ``test_slice``: 2-D slice (axes=[0,1], steps=[1,1]) on the
  // first two dimensions of a 20x10x5 input.
  {
    const Tensor data = MakeRangeTensor20x10x5();
    const Tensor starts = MakeInt64VectorTensor({0, 0});
    const Tensor ends = MakeInt64VectorTensor({3, 10});
    const Tensor axes = MakeInt64VectorTensor({0, 1});
    const Tensor steps = MakeInt64VectorTensor({1, 1});
    const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
    Expect(MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           {data, starts, ends, axes, steps}, {output}, "test_cc_slice", {opset}, "backend-test",
           registry);
  }

  // Mirrors ONNX ``test_slice_neg``: slice axis 1 with a negative ``end``.
  {
    const Tensor data = MakeRangeTensor20x10x5();
    const Tensor starts = MakeInt64VectorTensor({0});
    const Tensor ends = MakeInt64VectorTensor({-1});
    const Tensor axes = MakeInt64VectorTensor({1});
    const Tensor steps = MakeInt64VectorTensor({1});
    const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
    Expect(MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           {data, starts, ends, axes, steps}, {output}, "test_cc_slice_neg", {opset},
           "backend-test", registry);
  }

  // Mirrors ONNX ``test_slice_start_out_of_bounds``: both ``start`` and
  // ``end`` are far past the axis length, producing an empty slice.
  {
    const Tensor data = MakeRangeTensor20x10x5();
    const Tensor starts = MakeInt64VectorTensor({1000});
    const Tensor ends = MakeInt64VectorTensor({1000});
    const Tensor axes = MakeInt64VectorTensor({1});
    const Tensor steps = MakeInt64VectorTensor({1});
    const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
    Expect(MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           {data, starts, ends, axes, steps}, {output}, "test_cc_slice_start_out_of_bounds",
           {opset}, "backend-test", registry);
  }

  // Mirrors ONNX ``test_slice_end_out_of_bounds``: ``end`` is past the axis
  // length and is clamped to it.
  {
    const Tensor data = MakeRangeTensor20x10x5();
    const Tensor starts = MakeInt64VectorTensor({1});
    const Tensor ends = MakeInt64VectorTensor({1000});
    const Tensor axes = MakeInt64VectorTensor({1});
    const Tensor steps = MakeInt64VectorTensor({1});
    const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
    Expect(MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           {data, starts, ends, axes, steps}, {output}, "test_cc_slice_end_out_of_bounds", {opset},
           "backend-test", registry);
  }

  // Mirrors ONNX ``test_slice_default_steps``: 3-D slice with axes provided
  // but no ``steps`` input (defaults to 1).
  {
    const Tensor data = MakeRangeTensor20x10x5();
    const Tensor starts = MakeInt64VectorTensor({0, 0, 3});
    const Tensor ends = MakeInt64VectorTensor({20, 10, 4});
    const Tensor axes = MakeInt64VectorTensor({0, 1, 2});
    const Tensor output = slice_kernel(data, starts, ends, &axes);
    Expect(MakeSliceNode(/*with_axes=*/true, /*with_steps=*/false), {data, starts, ends, axes},
           {output}, "test_cc_slice_default_steps", {opset}, "backend-test", registry);
  }

  // Mirrors ONNX ``test_slice_neg_steps``: 3-D reverse slice with negative
  // ``steps`` on every axis.
  {
    const Tensor data = MakeRangeTensor20x10x5();
    const Tensor starts = MakeInt64VectorTensor({20, 10, 4});
    const Tensor ends = MakeInt64VectorTensor({0, 0, 1});
    const Tensor axes = MakeInt64VectorTensor({0, 1, 2});
    const Tensor steps = MakeInt64VectorTensor({-1, -3, -2});
    const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
    Expect(MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           {data, starts, ends, axes, steps}, {output}, "test_cc_slice_neg_steps", {opset},
           "backend-test", registry);
  }

  // Mirrors ONNX ``test_slice_negative_axes``: 3-D slice using negative
  // ``axes`` values that count from the end of ``data``'s rank.
  {
    const Tensor data = MakeRangeTensor20x10x5();
    const Tensor starts = MakeInt64VectorTensor({0, 0, 3});
    const Tensor ends = MakeInt64VectorTensor({20, 10, 4});
    const Tensor axes = MakeInt64VectorTensor({0, -2, -1});
    const Tensor output = slice_kernel(data, starts, ends, &axes);
    Expect(MakeSliceNode(/*with_axes=*/true, /*with_steps=*/false), {data, starts, ends, axes},
           {output}, "test_cc_slice_negative_axes", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test

} // namespace ONNX_LIGHT_NAMESPACE
