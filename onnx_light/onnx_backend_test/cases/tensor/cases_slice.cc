// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
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
}

} // namespace onnx_backend_test

} // namespace ONNX_LIGHT_NAMESPACE
