// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

Tensor MakeAxesTensor(const std::vector<int64_t> &axes) {
  std::vector<uint8_t> data(axes.size() * sizeof(int64_t));
  if (!axes.empty()) {
    std::memcpy(data.data(), axes.data(), data.size());
  }
  return Tensor("", DataType::INT64, {static_cast<int64_t>(axes.size())}, std::move(data));
}

NodeProto MakeSqueezeNode() {
  NodeProto node;
  node.set_op_type("Squeeze");
  node.add_input("data");
  node.add_input("axes");
  node.add_output("squeezed");
  return node;
}

} // namespace

void RegisterSqueezeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Squeeze squeeze_kernel{ctx};

  // test_cc_squeeze_axes
  {
    const Tensor data = Tensor::FromFloat("", {2, 1, 3, 1}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    const Tensor axes = MakeAxesTensor({1, 3});
    const Tensor squeezed = squeeze_kernel(data, {1, 3});
    Expect(MakeSqueezeNode(), {data, axes}, {squeezed}, "test_cc_squeeze_axes", {opset},
           "backend-test", registry);
  }

  // test_cc_squeeze_all_singleton
  {
    const Tensor data = Tensor::FromFloat("", {1, 2, 1, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    const Tensor axes = MakeAxesTensor({});
    const Tensor squeezed = squeeze_kernel(data, {});
    Expect(MakeSqueezeNode(), {data, axes}, {squeezed}, "test_cc_squeeze_all_singleton", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
