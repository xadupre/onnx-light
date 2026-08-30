// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/logical/include_logical_cases.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Registers one ``test_xor*`` upstream-ONNX case: x, y, and z = x XOR y with
// full NumPy broadcasting computed by ``kernel::Xor``. Seeds are fixed per
// shape to keep the data deterministic across runs.
void RegisterXorOnnxCase(const std::string &name, const std::vector<int64_t> &x_shape,
                         uint64_t x_seed, const std::vector<int64_t> &y_shape, uint64_t y_seed,
                         const auto &xor_kernel, const OpsetId &opset,
                         std::vector<TestCase> &registry) {
  NodeProto node = MakeNode("Xor", {"x", "y"}, {"xor"});
  Expect(registry, std::move(node), name, {opset}, [=]() -> IoData {
    Tensor x = RandBool(x_shape, x_seed);
    Tensor y = RandBool(y_shape, y_seed);
    Tensor z = xor_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });

    return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
  });
}

} // namespace

// ---------------------------------------------------------------------------
// Xor — z = x XOR y, element-wise with broadcasting (since opset 7).
// Inputs and outputs are BOOL tensors (one byte per element).
// ---------------------------------------------------------------------------
void RegisterXorCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(7);
  const auto xor_kernel = MakeReferenceKernel<onnx_kernels::kernel::Xor>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("Xor", {"x", "y"}, {"z"});

    const std::vector<int64_t> shape = {1024, 4096};
    const int64_t count = 1024 * 4096;
    Expect(registry, std::move(node), "test_cc_xor_benchmark", {opset}, {count, count}, {count},
           [xor_kernel, shape]() -> IoData {
             Tensor x = RandBool(shape, /*seed=*/9101);
             Tensor y = RandBool(shape, /*seed=*/9102);
             Tensor z = xor_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Equal-shape variant.
  {
    NodeProto node = MakeNode("Xor", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_xor", {opset}, [=]() -> IoData {
      Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
      Tensor y("", DataType::BOOL, {2, 2}, {1, 1, 0, 0});
      Tensor z = xor_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Scalar broadcast variant: z[i] = x[i] XOR y (scalar).
  {
    NodeProto node = MakeNode("Xor", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_xor_bcast", {opset}, [=]() -> IoData {
      Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
      Tensor y("", DataType::BOOL, {}, {1});
      Tensor z = xor_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
