// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

// Registers one ``test_and*`` upstream-ONNX case: x, y, and z = x AND y with
// full NumPy broadcasting computed by ``kernel::And``. Seeds are fixed per
// shape to keep the data deterministic across runs.
void RegisterAndOnnxCase(const std::string &name, const std::vector<int64_t> &x_shape,
                         uint64_t x_seed, const std::vector<int64_t> &y_shape, uint64_t y_seed,
                         const kernel::And &and_kernel, const OpsetId &opset,
                         std::vector<TestCase> &registry) {
  NodeProto node = MakeNode("And", {"x", "y"}, {"and"});

  Tensor x = RandBool(x_shape, x_seed);
  Tensor y = RandBool(y_shape, y_seed);
  Tensor z = and_kernel(x, y);

  Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// And — z = x AND y, element-wise with broadcasting (since opset 7).
// Inputs and outputs are BOOL tensors (one byte per element).
// ---------------------------------------------------------------------------
void RegisterAndCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(7);
  const kernel::KernelContext ctx{opset};
  const kernel::And and_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node = MakeNode("And", {"x", "y"}, {"z"});

    Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y("", DataType::BOOL, {2, 2}, {1, 1, 0, 0});
    Tensor z = and_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_and", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] AND y (scalar).
  {
    NodeProto node = MakeNode("And", {"x", "y"}, {"z"});

    Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y("", DataType::BOOL, {}, {1});
    Tensor z = and_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_and_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``And`` operator (mirror the
  // ``onnx.backend.test.case.node.and_.And`` Python class). Inputs are
  // generated deterministically through the seeded ``RandBool`` helper to
  // match the upstream ``(randn(...) > 0).astype(bool)`` pattern; expected
  // outputs are computed by ``kernel::And`` with multidirectional broadcasting.
  //
  // From And.export():
  RegisterAndOnnxCase("test_and2d", {3, 4}, 1, {3, 4}, 2, and_kernel, opset, registry);
  RegisterAndOnnxCase("test_and3d", {3, 4, 5}, 3, {3, 4, 5}, 4, and_kernel, opset, registry);
  RegisterAndOnnxCase("test_and4d", {3, 4, 5, 6}, 5, {3, 4, 5, 6}, 6, and_kernel, opset, registry);
  // From And.export_and_broadcast():
  RegisterAndOnnxCase("test_and_bcast3v1d", {3, 4, 5}, 7, {5}, 8, and_kernel, opset, registry);
  RegisterAndOnnxCase("test_and_bcast3v2d", {3, 4, 5}, 9, {4, 5}, 10, and_kernel, opset, registry);
  RegisterAndOnnxCase("test_and_bcast4v2d", {3, 4, 5, 6}, 11, {5, 6}, 12, and_kernel, opset,
                      registry);
  RegisterAndOnnxCase("test_and_bcast4v3d", {3, 4, 5, 6}, 13, {4, 5, 6}, 14, and_kernel, opset,
                      registry);
  RegisterAndOnnxCase("test_and_bcast4v4d", {1, 4, 1, 6}, 15, {3, 1, 5, 6}, 16, and_kernel, opset,
                      registry);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
