// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeGatherElementsNode(int64_t axis) {
  NodeProto node;
  node.set_op_type("GatherElements");
  node.add_input("data");
  node.add_input("indices");
  node.add_output("output");
  AddAttribute<int64_t>(node, "axis", axis);
  return node;
}

} // namespace

void RegisterGatherElementsCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::GatherElements ge_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeGatherElementsNode(1);
    Expect(registry, std::move(node), "test_cc_gather_elements_0_benchmark", {opset},
           {kBenchmarkElementwiseSize, kBenchmarkElementwiseSize}, {kBenchmarkElementwiseSize},
           [ge_kernel]() -> IoData {
             Tensor data = Tensor::FromFloat("", {4096, 1024}, Randn<float>({4096, 1024}, 2001));
             std::vector<int64_t> index_values(kBenchmarkElementwiseSize);
             for (int64_t i = 0; i < kBenchmarkElementwiseSize; ++i) {
               index_values[static_cast<std::size_t>(i)] = i % 1024;
             }
             Tensor indices = Tensor::FromInt64("", {4096, 1024}, index_values);
             Tensor output = ge_kernel(data, indices, 1);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
    return;
  }

  // test_cc_gather_elements_0 — axis=1, mirrors upstream
  // ``test_gather_elements_0`` (small 2x2 example).
  {
    Expect(registry, MakeGatherElementsNode(1), "test_cc_gather_elements_0", {opset},
           [=]() -> IoData {
             Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
             Tensor indices = Tensor::FromInt64("", {2, 2}, {0, 0, 1, 0});
             Tensor output = ge_kernel(data, indices, 1);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
  }

  // test_cc_gather_elements_1 — axis=0, mirrors upstream
  // ``test_gather_elements_1`` (3x3 example).
  {
    Expect(
        registry, MakeGatherElementsNode(0), "test_cc_gather_elements_1", {opset}, [=]() -> IoData {
          Tensor data =
              Tensor::FromFloat("", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
          Tensor indices = Tensor::FromInt64("", {2, 3}, {1, 2, 0, 2, 0, 0});
          Tensor output = ge_kernel(data, indices, 0);
          return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
        });
  }

  // test_cc_gather_elements_negative_indices — negative indices wrap.
  {
    Expect(registry, MakeGatherElementsNode(0), "test_cc_gather_elements_negative_indices", {opset},
           [=]() -> IoData {
             Tensor data = Tensor::FromFloat(
                 "", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
             Tensor indices = Tensor::FromInt64("", {2, 3}, {-1, -2, 0, -2, 0, 0});
             Tensor output = ge_kernel(data, indices, 0);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
