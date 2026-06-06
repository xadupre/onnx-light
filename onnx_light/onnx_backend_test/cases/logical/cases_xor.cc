// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Registers one ``test_xor*`` upstream-ONNX case: x, y, and z = x XOR y with
// full NumPy broadcasting computed by ``kernel::Xor``. Seeds are fixed per
// shape to keep the data deterministic across runs.
void RegisterXorOnnxCase(const std::string &name, const std::vector<int64_t> &x_shape,
                         uint64_t x_seed, const std::vector<int64_t> &y_shape, uint64_t y_seed,
                         const kernel::Xor &xor_kernel, const OpsetId &opset,
                         std::vector<TestCase> &registry) {
  NodeProto node = MakeNode("Xor", {"x", "y"}, {"xor"});

  Tensor x = RandBool(x_shape, x_seed);
  Tensor y = RandBool(y_shape, y_seed);
  Tensor z = xor_kernel(x, y);

  Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// Xor — z = x XOR y, element-wise with broadcasting (since opset 7).
// Inputs and outputs are BOOL tensors (one byte per element).
// ---------------------------------------------------------------------------
void RegisterXorCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(7);
  const kernel::KernelContext ctx{opset};
  const kernel::Xor xor_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node = MakeNode("Xor", {"x", "y"}, {"z"});

    Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y("", DataType::BOOL, {2, 2}, {1, 1, 0, 0});
    Tensor z = xor_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_xor", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] XOR y (scalar).
  {
    NodeProto node = MakeNode("Xor", {"x", "y"}, {"z"});

    Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y("", DataType::BOOL, {}, {1});
    Tensor z = xor_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_xor_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Xor`` operator (mirror the
  // ``onnx.backend.test.case.node.xor.Xor`` Python class). Inputs are
  // generated deterministically through the seeded ``RandBool`` helper to
  // match the upstream ``(randn(...) > 0).astype(bool)`` pattern; expected
  // outputs are computed by ``kernel::Xor`` with multidirectional broadcasting.
  //
  // From Xor.export():
  RegisterXorOnnxCase("test_xor2d", {3, 4}, 201, {3, 4}, 202, xor_kernel, opset, registry);
  RegisterXorOnnxCase("test_xor3d", {3, 4, 5}, 203, {3, 4, 5}, 204, xor_kernel, opset, registry);
  RegisterXorOnnxCase("test_xor4d", {3, 4, 5, 6}, 205, {3, 4, 5, 6}, 206, xor_kernel, opset,
                      registry);
  // From Xor.export_xor_broadcast():
  RegisterXorOnnxCase("test_xor_bcast3v1d", {3, 4, 5}, 207, {5}, 208, xor_kernel, opset, registry);
  RegisterXorOnnxCase("test_xor_bcast3v2d", {3, 4, 5}, 209, {4, 5}, 210, xor_kernel, opset,
                      registry);
  RegisterXorOnnxCase("test_xor_bcast4v2d", {3, 4, 5, 6}, 211, {5, 6}, 212, xor_kernel, opset,
                      registry);
  RegisterXorOnnxCase("test_xor_bcast4v3d", {3, 4, 5, 6}, 213, {4, 5, 6}, 214, xor_kernel, opset,
                      registry);
  RegisterXorOnnxCase("test_xor_bcast4v4d", {1, 4, 1, 6}, 215, {3, 1, 5, 6}, 216, xor_kernel, opset,
                      registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
