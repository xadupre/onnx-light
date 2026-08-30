// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeScatterNode(int64_t axis, bool set_axis_attr) {
  NodeProto node;
  node.set_op_type("Scatter");
  node.add_input("data");
  node.add_input("indices");
  node.add_input("updates");
  node.add_output("y");
  if (set_axis_attr) {
    AddAttribute<int64_t>(node, "axis", axis);
  }
  return node;
}

} // namespace

void RegisterScatterCases(std::vector<TestCase> &registry, TestMode mode) {
  // ``Scatter`` is deprecated since opset 11; the upstream ONNX backend test
  // cases pin the opset to 10. Use the same opset here so the generated
  // models are valid (Scatter is not registered in opset >= 11).
  const OpsetId opset = DefaultOpset(10);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeScatterNode(0, /*set_axis_attr=*/false);
    Expect(registry, std::move(node), "test_cc_scatter_without_axis_benchmark", {opset},
           {4194304, 4194304, 4194304}, {4194304}, []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext scatter_kernel_ctx{opset};
             const onnx_kernels::kernel::Scatter scatter_kernel{scatter_kernel_ctx};

             Tensor data = Tensor::FromFloat("", {4096, 1024},
                                             std::vector<float>(kBenchmarkElementwiseSize, 0.0f));
             std::vector<int64_t> index_values(kBenchmarkElementwiseSize);
             for (int64_t i = 0; i < kBenchmarkElementwiseSize; ++i) {
               index_values[static_cast<std::size_t>(i)] = i % 4096;
             }
             Tensor indices = Tensor::FromInt64("", {4096, 1024}, index_values);
             Tensor updates = RandnTensor(DataType::FLOAT, {4096, 1024}, 2001);
             onnx_kernels::kernel::Scatter::Attributes attrs;
             Tensor output = scatter_kernel(data, indices, updates, attrs);
             return IoData{{std::move(data), std::move(indices), std::move(updates)},
                           {std::move(output)}};
           });
    return;
  }

  // test_cc_scatter_without_axis — mirrors upstream ``test_scatter_without_axis``.
  {
    Expect(registry, MakeScatterNode(0, /*set_axis_attr=*/false), "test_cc_scatter_without_axis",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext scatter_kernel_ctx{opset};
             const onnx_kernels::kernel::Scatter scatter_kernel{scatter_kernel_ctx};

             Tensor data = Tensor::FromFloat("", {3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
             Tensor indices = Tensor::FromInt64("", {2, 3}, {1, 0, 2, 0, 2, 1});
             Tensor updates = Tensor::FromFloat("", {2, 3}, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
             onnx_kernels::kernel::Scatter::Attributes attrs;
             Tensor output = scatter_kernel(data, indices, updates, attrs);
             return IoData{{std::move(data), std::move(indices), std::move(updates)},
                           {std::move(output)}};
           });
  }

  // test_cc_scatter_with_axis — mirrors upstream ``test_scatter_with_axis``.
  {
    Expect(registry, MakeScatterNode(1, /*set_axis_attr=*/true), "test_cc_scatter_with_axis",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext scatter_kernel_ctx{opset};
             const onnx_kernels::kernel::Scatter scatter_kernel{scatter_kernel_ctx};

             Tensor data = Tensor::FromFloat("", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
             Tensor indices = Tensor::FromInt64("", {1, 2}, {1, 3});
             Tensor updates = Tensor::FromFloat("", {1, 2}, {1.1f, 2.1f});
             onnx_kernels::kernel::Scatter::Attributes attrs;
             attrs.axis = 1;
             Tensor output = scatter_kernel(data, indices, updates, attrs);
             return IoData{{std::move(data), std::move(indices), std::move(updates)},
                           {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
