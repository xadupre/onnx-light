// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
                                  int64_t axis, const std::string &name) {
  Expect(registry, MakeUnsqueezeNodeXY(), name, {opset}, [opset, axis]() -> IoData {
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::Unsqueeze unsqueeze_kernel{ctx};

    const Tensor x = Tensor::FromFloat("", {3, 4, 5}, Iota(60));
    const std::vector<int64_t> axes{axis};
    const Tensor axes_tensor = MakeAxesTensor(axes);
    const Tensor y = unsqueeze_kernel(x, axes);
    return IoData{{std::move(x), std::move(axes_tensor)}, {std::move(y)}};
  });
}

} // namespace

void RegisterUnsqueezeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeUnsqueezeNode();
    const std::vector<int64_t> axes{0, 2};
    Expect(registry, std::move(node), "test_cc_unsqueeze_axes_benchmark", {opset}, {4194304, 2},
           {4194304}, [axes]() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext unsqueeze_kernel_ctx{opset};
             const onnx_kernels::kernel::Unsqueeze unsqueeze_kernel{unsqueeze_kernel_ctx};

             Tensor data = RandnTensor(DataType::FLOAT, {2048, 2048}, 2001);
             Tensor axes_tensor = MakeAxesTensor(axes);
             Tensor expanded = unsqueeze_kernel(data, axes);
             return IoData{{std::move(data), std::move(axes_tensor)}, {std::move(expanded)}};
           });
    return;
  }

  // test_cc_unsqueeze_axes
  {
    Expect(registry, MakeUnsqueezeNode(), "test_cc_unsqueeze_axes", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext unsqueeze_kernel_ctx{opset};
      const onnx_kernels::kernel::Unsqueeze unsqueeze_kernel{unsqueeze_kernel_ctx};

      const Tensor data = Tensor::FromFloat("", {2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
      const std::vector<int64_t> axes{0, 2};
      const Tensor axes_tensor = MakeAxesTensor(axes);
      const Tensor expanded = unsqueeze_kernel(data, axes);
      return IoData{{std::move(data), std::move(axes_tensor)}, {std::move(expanded)}};
    });
  }

  // Cases imported from ONNX backend tests
  // (onnx/backend/test/case/node/unsqueeze.py).

  // test_unsqueeze_axis_0, test_unsqueeze_axis_1, test_unsqueeze_axis_2
  RegisterUnsqueezeOneAxisCase(registry, opset, 0, "test_unsqueeze_axis_0");
  RegisterUnsqueezeOneAxisCase(registry, opset, 1, "test_unsqueeze_axis_1");
  RegisterUnsqueezeOneAxisCase(registry, opset, 2, "test_unsqueeze_axis_2");

  // test_unsqueeze_two_axes
  {
    Expect(registry, MakeUnsqueezeNodeXY(), "test_unsqueeze_two_axes", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext unsqueeze_kernel_ctx{opset};
      const onnx_kernels::kernel::Unsqueeze unsqueeze_kernel{unsqueeze_kernel_ctx};

      const Tensor x = Tensor::FromFloat("", {3, 4, 5}, Iota(60));
      const std::vector<int64_t> axes{1, 4};
      const Tensor axes_tensor = MakeAxesTensor(axes);
      const Tensor y = unsqueeze_kernel(x, axes);
      return IoData{{std::move(x), std::move(axes_tensor)}, {std::move(y)}};
    });
  }

  // test_unsqueeze_three_axes
  {
    Expect(registry, MakeUnsqueezeNodeXY(), "test_unsqueeze_three_axes", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext unsqueeze_kernel_ctx{opset};
      const onnx_kernels::kernel::Unsqueeze unsqueeze_kernel{unsqueeze_kernel_ctx};

      const Tensor x = Tensor::FromFloat("", {3, 4, 5}, Iota(60));
      const std::vector<int64_t> axes{2, 4, 5};
      const Tensor axes_tensor = MakeAxesTensor(axes);
      const Tensor y = unsqueeze_kernel(x, axes);
      return IoData{{std::move(x), std::move(axes_tensor)}, {std::move(y)}};
    });
  }

  // test_unsqueeze_unsorted_axes
  {
    Expect(registry, MakeUnsqueezeNodeXY(), "test_unsqueeze_unsorted_axes", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext unsqueeze_kernel_ctx{opset};
             const onnx_kernels::kernel::Unsqueeze unsqueeze_kernel{unsqueeze_kernel_ctx};

             const Tensor x = Tensor::FromFloat("", {3, 4, 5}, Iota(60));
             // ONNX exports the axes in unsorted order; kernel sorts them internally,
             // so the output is identical to test_unsqueeze_three_axes.
             const std::vector<int64_t> axes{5, 4, 2};
             const Tensor axes_tensor = MakeAxesTensor(axes);
             const Tensor y = unsqueeze_kernel(x, axes);
             return IoData{{std::move(x), std::move(axes_tensor)}, {std::move(y)}};
           });
  }

  // test_unsqueeze_negative_axes
  {
    Expect(registry, MakeUnsqueezeNodeXY(), "test_unsqueeze_negative_axes", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext unsqueeze_kernel_ctx{opset};
             const onnx_kernels::kernel::Unsqueeze unsqueeze_kernel{unsqueeze_kernel_ctx};

             const Tensor x = Tensor::FromFloat("", {1, 3, 1, 5}, Iota(15));
             const std::vector<int64_t> axes{-2};
             const Tensor axes_tensor = MakeAxesTensor(axes);
             const Tensor y = unsqueeze_kernel(x, axes);
             return IoData{{std::move(x), std::move(axes_tensor)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
