// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Expand — y = broadcast(input, shape) following numpy-style broadcasting
// (since opset 8). Mirrors the upstream ONNX node tests in
// ``onnx/backend/test/case/node/expand.py``.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeExpandNode() {
  NodeProto node;
  node.set_op_type("Expand");
  node.add_input("input");
  node.add_input("shape");
  node.add_output("output");
  return node;
}

// Builds a 1-D INT64 shape tensor from a list of dimension values.
Tensor MakeShapeTensor(const std::vector<int64_t> &dims) {
  const std::vector<int64_t> shape_shape = {static_cast<int64_t>(dims.size())};
  std::vector<uint8_t> data(dims.size() * sizeof(int64_t));
  std::memcpy(data.data(), dims.data(), data.size());
  return Tensor("", static_cast<int32_t>(TensorProto::DataType::INT64), shape_shape,
                std::move(data));
}

} // namespace

void RegisterExpandCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Expand expand_kernel{ctx};

  // test_cc_expand_dim_changed
  // input: shape [3, 1], values [1.0, 2.0, 3.0]
  // shape: [2, 3, 6]
  // output: shape [2, 3, 6] — input is broadcast along axis 0 (new dim) and axis 2.
  {
    const Tensor input = Tensor::FromFloat("", {3, 1}, {1.0f, 2.0f, 3.0f});
    const Tensor shape = MakeShapeTensor({2, 3, 6});
    const Tensor output = expand_kernel(input, shape);
    Expect(MakeExpandNode(), {input, shape}, {output}, "test_cc_expand_dim_changed", {opset},
           "backend-test", registry);
  }

  // test_cc_expand_dim_unchanged
  // input: shape [3, 1], values [1.0, 2.0, 3.0]
  // shape: [3, 4]
  // output: shape [3, 4] — input is broadcast along axis 1 only.
  {
    const Tensor input = Tensor::FromFloat("", {3, 1}, {1.0f, 2.0f, 3.0f});
    const Tensor shape = MakeShapeTensor({3, 4});
    const Tensor output = expand_kernel(input, shape);
    Expect(MakeExpandNode(), {input, shape}, {output}, "test_cc_expand_dim_unchanged", {opset},
           "backend-test", registry);
  }

  // test_cc_expand_1d_to_2d
  // input: shape [4], values [1.0, 2.0, 3.0, 4.0]
  // shape: [3, 4]
  // output: shape [3, 4] — input is broadcast by adding a leading batch dim.
  {
    const Tensor input = Tensor::FromFloat("", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor shape = MakeShapeTensor({3, 4});
    const Tensor output = expand_kernel(input, shape);
    Expect(MakeExpandNode(), {input, shape}, {output}, "test_cc_expand_1d_to_2d", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
