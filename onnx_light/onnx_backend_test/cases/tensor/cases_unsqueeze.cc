// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

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

NodeProto MakeUnsqueezeNode() {
  NodeProto node;
  node.set_op_type("Unsqueeze");
  node.add_input("data");
  node.add_input("axes");
  node.add_output("expanded");
  return node;
}

} // namespace

void RegisterUnsqueezeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Unsqueeze unsqueeze_kernel{ctx};

  // test_cc_unsqueeze_axes
  {
    const Tensor data = Tensor::FromFloat("", {2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    const std::vector<int64_t> axes{0, 2};
    const Tensor axes_tensor = MakeAxesTensor(axes);
    const Tensor expanded = unsqueeze_kernel(data, axes);
    Expect(MakeUnsqueezeNode(), {data, axes_tensor}, {expanded}, "test_cc_unsqueeze_axes", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
