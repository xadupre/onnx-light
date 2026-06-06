// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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

void RegisterGatherNDCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::GatherND gnd_kernel{ctx};

  // test_cc_gathernd_example_int32 — mirrors upstream
  // ``test_gathernd_example_int32`` (2x2 data with k_last == rank).
  {
    Tensor data = Tensor::FromInt32("", {2, 2}, {0, 1, 2, 3});
    Tensor indices = Tensor::FromInt64("", {2, 2}, {0, 0, 1, 1});
    Tensor output = gnd_kernel(data, indices, 0);
    Expect(MakeGatherNDNode(0, true), {data, indices}, {output}, "test_cc_gathernd_example_int32",
           {opset}, "backend-test", registry);
  }

  // test_cc_gathernd_example_float32 — partial gather (k_last < rank); each
  // index selects a 1-D slice of length 2.
  {
    Tensor data = Tensor::FromFloat("", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
    Tensor indices = Tensor::FromInt64("", {2, 1}, {1, 0});
    Tensor output = gnd_kernel(data, indices, 0);
    Expect(MakeGatherNDNode(0, true), {data, indices}, {output}, "test_cc_gathernd_example_float32",
           {opset}, "backend-test", registry);
  }

  // test_cc_gathernd_example_int32_batch_dim1 — batch_dims=1 variant.
  // For each batch i, indices[i] picks rows of data[i] independently.
  {
    Tensor data = Tensor::FromInt32("", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
    Tensor indices = Tensor::FromInt64("", {2, 1}, {1, 0});
    Tensor output = gnd_kernel(data, indices, 1);
    Expect(MakeGatherNDNode(1, true), {data, indices}, {output},
           "test_cc_gathernd_example_int32_batch_dim1", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
