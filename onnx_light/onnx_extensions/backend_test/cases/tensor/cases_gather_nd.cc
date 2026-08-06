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

NodeProto MakeGatherNDNode(int64_t batch_dims, bool include_batch_dims) {
  NodeProto node;
  node.set_op_type("GatherND");
  node.add_input("data");
  node.add_input("indices");
  node.add_output("output");
  if (include_batch_dims) {
    AddAttribute<int64_t>(node, "batch_dims", batch_dims);
  }
  return node;
}

} // namespace

void RegisterGatherNDCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::GatherND gnd_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeGatherNDNode(0, true);
    Expect(registry, std::move(node), "test_cc_gathernd_example_int32_benchmark", {opset},
           {kBenchmarkElementwiseSize, kBenchmarkElementwiseSize * 2}, {kBenchmarkElementwiseSize},
           [gnd_kernel]() -> IoData {
             std::vector<int32_t> data_values(kBenchmarkElementwiseSize);
             for (int64_t i = 0; i < kBenchmarkElementwiseSize; ++i) {
               data_values[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % 65536);
             }
             Tensor data = Tensor::FromInt32("", {4096, 1024}, data_values);
             std::vector<int64_t> index_values(kBenchmarkElementwiseSize * 2);
             for (int64_t i = 0; i < kBenchmarkElementwiseSize; ++i) {
               index_values[static_cast<std::size_t>(2 * i)] = i % 4096;
               index_values[static_cast<std::size_t>(2 * i + 1)] = i % 1024;
             }
             Tensor indices = Tensor::FromInt64("", {kBenchmarkElementwiseSize, 2}, index_values);
             Tensor output = gnd_kernel(data, indices, 0);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
    return;
  }

  // test_cc_gathernd_example_int32 — mirrors upstream
  // ``test_gathernd_example_int32`` (2x2 data with k_last == rank).
  {
    Expect(registry, MakeGatherNDNode(0, true), "test_cc_gathernd_example_int32", {opset},
           [=]() -> IoData {
             Tensor data = Tensor::FromInt32("", {2, 2}, {0, 1, 2, 3});
             Tensor indices = Tensor::FromInt64("", {2, 2}, {0, 0, 1, 1});
             Tensor output = gnd_kernel(data, indices, 0);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
  }

  // test_cc_gathernd_example_float32 — partial gather (k_last < rank); each
  // index selects a 1-D slice of length 2.
  {
    Expect(registry, MakeGatherNDNode(0, true), "test_cc_gathernd_example_float32", {opset},
           [=]() -> IoData {
             Tensor data = Tensor::FromFloat("", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
             Tensor indices = Tensor::FromInt64("", {2, 1}, {1, 0});
             Tensor output = gnd_kernel(data, indices, 0);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
  }

  // test_cc_gathernd_example_int32_batch_dim1 — batch_dims=1 variant.
  // For each batch i, indices[i] picks rows of data[i] independently.
  {
    Expect(registry, MakeGatherNDNode(1, true), "test_cc_gathernd_example_int32_batch_dim1",
           {opset}, [=]() -> IoData {
             Tensor data = Tensor::FromInt32("", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
             Tensor indices = Tensor::FromInt64("", {2, 1}, {1, 0});
             Tensor output = gnd_kernel(data, indices, 1);
             return IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
