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

// Registers one ``test_and*`` upstream-ONNX case: x, y, and z = x AND y with
// full NumPy broadcasting computed by ``kernel::And``. Seeds are fixed per
// shape to keep the data deterministic across runs.
void RegisterAndOnnxCase(const std::string &name, const std::vector<int64_t> &x_shape,
                         uint64_t x_seed, const std::vector<int64_t> &y_shape, uint64_t y_seed,
                         const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode("And", {"x", "y"}, {"and"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const onnx_kernels::kernel::And and_kernel{ctx};

           Tensor x = RandBool(x_shape, x_seed);
           Tensor y = RandBool(y_shape, y_seed);
           Tensor z = and_kernel(x, y);

           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

} // namespace

// ---------------------------------------------------------------------------
// And — z = x AND y, element-wise with broadcasting (since opset 7).
// Inputs and outputs are BOOL tensors (one byte per element).
// ---------------------------------------------------------------------------
void RegisterAndCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(7);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("And", {"x", "y"}, {"z"});

    const std::vector<int64_t> shape = {1024, 4096};
    const int64_t count = 1024 * 4096;
    Expect(registry, std::move(node), "test_cc_and_benchmark", {opset}, {count, count}, {count},
           [shape]() -> IoData {
             const OpsetId opset = DefaultOpset(7);

             const KernelContext and_kernel_ctx{opset};
             const onnx_kernels::kernel::And and_kernel{and_kernel_ctx};

             Tensor x = RandBool(shape, /*seed=*/9101);
             Tensor y = RandBool(shape, /*seed=*/9102);
             Tensor z = and_kernel(x, y);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Equal-shape variant.
  {
    NodeProto node = MakeNode("And", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_and", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(7);

      const KernelContext and_kernel_ctx{opset};
      const onnx_kernels::kernel::And and_kernel{and_kernel_ctx};

      Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
      Tensor y("", DataType::BOOL, {2, 2}, {1, 1, 0, 0});
      Tensor z = and_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Scalar broadcast variant: z[i] = x[i] AND y (scalar).
  {
    NodeProto node = MakeNode("And", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_and_bcast", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(7);

      const KernelContext and_kernel_ctx{opset};
      const onnx_kernels::kernel::And and_kernel{and_kernel_ctx};

      Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
      Tensor y("", DataType::BOOL, {}, {1});
      Tensor z = and_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX backend test cases for the ``And`` operator (mirror the
  // ``onnx.backend.test.case.node.and_.And`` Python class). Inputs are
  // generated deterministically through the seeded ``RandBool`` helper to
  // match the upstream ``(randn(...) > 0).astype(bool)`` pattern; expected
  // outputs are computed by ``kernel::And`` with multidirectional broadcasting.
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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
