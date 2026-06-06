// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
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

NodeProto MakeUnsqueezeNodeXY() {
  NodeProto node;
  node.set_op_type("Unsqueeze");
  node.add_input("x");
  node.add_input("axes");
  node.add_output("y");
  return node;
}

std::vector<float> Iota(int64_t n) {
  std::vector<float> v;
  v.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    v.push_back(static_cast<float>(i));
  }
  return v;
}

void RegisterUnsqueezeOneAxisCase(std::vector<TestCase> &registry, const OpsetId &opset,
                                  const kernel::Unsqueeze &unsqueeze_kernel, int64_t axis,
                                  const std::string &name) {
  const Tensor x = Tensor::FromFloat("", {3, 4, 5}, Iota(60));
  const std::vector<int64_t> axes{axis};
  const Tensor axes_tensor = MakeAxesTensor(axes);
  const Tensor y = unsqueeze_kernel(x, axes);
  Expect(MakeUnsqueezeNodeXY(), {x, axes_tensor}, {y}, name, {opset}, "backend-test", registry);
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

  // Cases imported from ONNX backend tests
  // (onnx/backend/test/case/node/unsqueeze.py).

  // test_unsqueeze_axis_0, test_unsqueeze_axis_1, test_unsqueeze_axis_2
  RegisterUnsqueezeOneAxisCase(registry, opset, unsqueeze_kernel, 0, "test_unsqueeze_axis_0");
  RegisterUnsqueezeOneAxisCase(registry, opset, unsqueeze_kernel, 1, "test_unsqueeze_axis_1");
  RegisterUnsqueezeOneAxisCase(registry, opset, unsqueeze_kernel, 2, "test_unsqueeze_axis_2");

  // test_unsqueeze_two_axes
  {
    const Tensor x = Tensor::FromFloat("", {3, 4, 5}, Iota(60));
    const std::vector<int64_t> axes{1, 4};
    const Tensor axes_tensor = MakeAxesTensor(axes);
    const Tensor y = unsqueeze_kernel(x, axes);
    Expect(MakeUnsqueezeNodeXY(), {x, axes_tensor}, {y}, "test_unsqueeze_two_axes", {opset},
           "backend-test", registry);
  }

  // test_unsqueeze_three_axes
  {
    const Tensor x = Tensor::FromFloat("", {3, 4, 5}, Iota(60));
    const std::vector<int64_t> axes{2, 4, 5};
    const Tensor axes_tensor = MakeAxesTensor(axes);
    const Tensor y = unsqueeze_kernel(x, axes);
    Expect(MakeUnsqueezeNodeXY(), {x, axes_tensor}, {y}, "test_unsqueeze_three_axes", {opset},
           "backend-test", registry);
  }

  // test_unsqueeze_unsorted_axes
  {
    const Tensor x = Tensor::FromFloat("", {3, 4, 5}, Iota(60));
    // ONNX exports the axes in unsorted order; kernel sorts them internally,
    // so the output is identical to test_unsqueeze_three_axes.
    const std::vector<int64_t> axes{5, 4, 2};
    const Tensor axes_tensor = MakeAxesTensor(axes);
    const Tensor y = unsqueeze_kernel(x, axes);
    Expect(MakeUnsqueezeNodeXY(), {x, axes_tensor}, {y}, "test_unsqueeze_unsorted_axes", {opset},
           "backend-test", registry);
  }

  // test_unsqueeze_negative_axes
  {
    const Tensor x = Tensor::FromFloat("", {1, 3, 1, 5}, Iota(15));
    const std::vector<int64_t> axes{-2};
    const Tensor axes_tensor = MakeAxesTensor(axes);
    const Tensor y = unsqueeze_kernel(x, axes);
    Expect(MakeUnsqueezeNodeXY(), {x, axes_tensor}, {y}, "test_unsqueeze_negative_axes", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
