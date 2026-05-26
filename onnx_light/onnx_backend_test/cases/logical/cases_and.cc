// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/random.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Computes the broadcasted output shape from ``a`` and ``b`` following the
// standard NumPy/ONNX multidirectional broadcasting rules.
std::vector<int64_t> BroadcastShape(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
  const size_t rank = a.size() > b.size() ? a.size() : b.size();
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
    const int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
    out[rank - 1 - i] = da > db ? da : db;
  }
  return out;
}

// Element-wise AND with full NumPy-style broadcasting over BOOL tensors. The
// returned tensor has shape ``BroadcastShape(x.shape, y.shape)``.
Tensor BroadcastAnd(const Tensor &x, const Tensor &y) {
  const std::vector<int64_t> out_shape = BroadcastShape(x.shape, y.shape);
  const size_t rank = out_shape.size();

  auto aligned = [&](const std::vector<int64_t> &s) {
    std::vector<int64_t> a(rank, 1);
    for (size_t i = 0; i < s.size(); ++i) {
      a[rank - s.size() + i] = s[i];
    }
    return a;
  };
  const std::vector<int64_t> sx = aligned(x.shape);
  const std::vector<int64_t> sy = aligned(y.shape);

  // Pre-compute row-major strides (in elements) for x and y in the broadcasted
  // coordinate space: a stride of 0 for a broadcast (size-1) dimension.
  std::vector<int64_t> strides_x(rank, 0), strides_y(rank, 0);
  int64_t acc_x = 1, acc_y = 1;
  for (size_t i = rank; i-- > 0;) {
    strides_x[i] = sx[i] == 1 ? 0 : acc_x;
    strides_y[i] = sy[i] == 1 ? 0 : acc_y;
    acc_x *= sx[i];
    acc_y *= sy[i];
  }

  int64_t element_count = 1;
  for (int64_t d : out_shape) {
    element_count *= d;
  }
  std::vector<uint8_t> out(static_cast<size_t>(element_count), 0);
  const uint8_t *px = x.data.data();
  const uint8_t *py = y.data.data();

  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat = 0; flat < element_count; ++flat) {
    int64_t ox = 0, oy = 0;
    for (size_t d = 0; d < rank; ++d) {
      ox += idx[d] * strides_x[d];
      oy += idx[d] * strides_y[d];
    }
    out[static_cast<size_t>(flat)] = (px[ox] != 0 && py[oy] != 0) ? 1 : 0;
    // Advance idx in row-major order.
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < out_shape[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
  return Tensor("", static_cast<int32_t>(TensorProto::DataType::BOOL), out_shape, std::move(out));
}

// Registers one ``test_and*`` upstream-ONNX case: x, y, and z = x AND y with
// full NumPy broadcasting. Seeds are fixed per shape to keep the data
// deterministic across runs.
void RegisterAndOnnxCase(const std::string &name, const std::vector<int64_t> &x_shape,
                         uint64_t x_seed, const std::vector<int64_t> &y_shape, uint64_t y_seed,
                         const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type("And");
  node.add_input("x");
  node.add_input("y");
  node.add_output("and");

  Tensor x = RandBool(x_shape, x_seed);
  Tensor y = RandBool(y_shape, y_seed);
  Tensor z = BroadcastAnd(x, y);

  Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// And — z = x AND y, element-wise with broadcasting (since opset 7).
// Inputs and outputs are BOOL tensors (one byte per element).
// ---------------------------------------------------------------------------
void RegisterAndCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(7);
  const kernel::And and_kernel{kernel::KernelContext(opset)};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("And");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
    Tensor z = and_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_and", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] AND y (scalar).
  {
    NodeProto node;
    node.set_op_type("And");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y("", TensorProto::DataType::BOOL, {}, {1});
    Tensor z = and_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_and_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``And`` operator (mirror the
  // ``onnx.backend.test.case.node.and_.And`` Python class). Inputs are
  // generated deterministically through the seeded ``Randn`` helper and
  // thresholded at 0 to match the upstream ``(randn(...) > 0).astype(bool)``
  // pattern; expected outputs are produced by NumPy-style broadcasting AND.
  //
  // From And.export():
  RegisterAndOnnxCase("test_and2d", {3, 4}, 1, {3, 4}, 2, opset, registry);
  RegisterAndOnnxCase("test_and3d", {3, 4, 5}, 3, {3, 4, 5}, 4, opset, registry);
  RegisterAndOnnxCase("test_and4d", {3, 4, 5, 6}, 5, {3, 4, 5, 6}, 6, opset, registry);
  // From And.export_and_broadcast():
  RegisterAndOnnxCase("test_and_bcast3v1d", {3, 4, 5}, 7, {5}, 8, opset, registry);
  RegisterAndOnnxCase("test_and_bcast3v2d", {3, 4, 5}, 9, {4, 5}, 10, opset, registry);
  RegisterAndOnnxCase("test_and_bcast4v2d", {3, 4, 5, 6}, 11, {5, 6}, 12, opset, registry);
  RegisterAndOnnxCase("test_and_bcast4v3d", {3, 4, 5, 6}, 13, {4, 5, 6}, 14, opset, registry);
  RegisterAndOnnxCase("test_and_bcast4v4d", {1, 4, 1, 6}, 15, {3, 1, 5, 6}, 16, opset, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
