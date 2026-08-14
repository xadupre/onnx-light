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

NodeProto MakeGatherNode(int64_t axis) {
  NodeProto node;
  node.set_op_type("Gather");
  node.add_input("data");
  node.add_input("indices");
  node.add_output("output");
  AddAttribute<int64_t>(node, "axis", axis);
  return node;
}

} // namespace

void RegisterGatherCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Gather gather_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeGatherNode(0);
    Expect(registry, std::move(node), "test_cc_gather_0_benchmark", {opset},
           {kBenchmarkElementwiseSize, 4096}, {kBenchmarkElementwiseSize},
           [gather_kernel]() -> IoData {
             Tensor data = RandnTensor(DataType::FLOAT, {4096, 1024}, 2001);
             std::vector<int64_t> index_values(4096);
             for (int64_t i = 0; i < 4096; ++i) {
               index_values[static_cast<std::size_t>(i)] = i;
             }
             Tensor indices = Tensor::FromInt64("", {4096}, index_values);
             Tensor output = gather_kernel(data, indices, 0);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
    return;
  }

  // test_cc_gather_0 — mirrors the upstream ``test_gather_0`` node test:
  // gather along axis=0 with 2-D indices.
  {
    Expect(registry, MakeGatherNode(0), "test_cc_gather_0", {opset}, [=]() -> IoData {
      Tensor data = Tensor::FromFloat("", {5, 4},
                                      {0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f,
                                       2.2f, 2.3f, 3.0f, 3.1f, 3.2f, 3.3f, 4.0f, 4.1f, 4.2f, 4.3f});
      Tensor indices = Tensor::FromInt64("", {2, 2}, {0, 1, 1, 2});
      Tensor output = gather_kernel(data, indices, 0);
      return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
    });
  }

  // test_cc_gather_1 — gather along axis=1.
  {
    Expect(registry, MakeGatherNode(1), "test_cc_gather_1", {opset}, [=]() -> IoData {
      Tensor data =
          Tensor::FromFloat("", {3, 3}, {1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f});
      Tensor indices = Tensor::FromInt64("", {1, 2}, {0, 2});
      Tensor output = gather_kernel(data, indices, 1);
      return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
    });
  }

  // test_cc_gather_2d_indices — mirrors the upstream ``test_gather_2d_indices``
  // node test: gather along axis=1 with 2-D indices.
  {
    Expect(registry, MakeGatherNode(1), "test_cc_gather_2d_indices", {opset}, [=]() -> IoData {
      Tensor data =
          Tensor::FromFloat("", {3, 3}, {1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f});
      Tensor indices = Tensor::FromInt64("", {1, 2}, {0, 2});
      Tensor output = gather_kernel(data, indices, 1);
      return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
    });
  }

  // test_cc_gather_negative_indices — negative indices wrap around the axis.
  {
    Expect(registry, MakeGatherNode(0), "test_cc_gather_negative_indices", {opset},
           [=]() -> IoData {
             Tensor data = Tensor::FromFloat("", {5}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
             Tensor indices = Tensor::FromInt64("", {3}, {0, -1, -2});
             Tensor output = gather_kernel(data, indices, 0);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
