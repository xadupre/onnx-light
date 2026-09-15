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

// Registers one ``test_or*`` upstream-ONNX case: x, y, and z = x OR y with
// full NumPy broadcasting computed by ``kernel::Or``. Seeds are fixed per
// shape to keep the data deterministic across runs.
void RegisterOrOnnxCase(const std::string &name, const std::vector<int64_t> &x_shape,
                        uint64_t x_seed, const std::vector<int64_t> &y_shape, uint64_t y_seed,
                        const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode("Or", {"x", "y"}, {"or"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const onnx_kernels::kernel::Or or_kernel{ctx};

           Tensor x = RandBool(x_shape, x_seed);
           Tensor y = RandBool(y_shape, y_seed);
           Tensor z = or_kernel(x, y);

           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

} // namespace

// ---------------------------------------------------------------------------
// Or — z = x OR y, element-wise with broadcasting (since opset 7).
// Inputs and outputs are BOOL tensors (one byte per element).
// ---------------------------------------------------------------------------
void RegisterOrCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(7);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("Or", {"x", "y"}, {"z"});

    const std::vector<int64_t> shape = {1024, 4096};
    const int64_t count = 1024 * 4096;
    Expect(registry, std::move(node), "test_cc_or_benchmark", {opset}, {count, count}, {count},
           [shape]() -> IoData {
             const OpsetId opset = DefaultOpset(7);

             const KernelContext or_kernel_ctx{opset};
             const onnx_kernels::kernel::Or or_kernel{or_kernel_ctx};

             Tensor x = RandBool(shape, /*seed=*/9101);
             Tensor y = RandBool(shape, /*seed=*/9102);
             Tensor z = or_kernel(x, y);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Equal-shape variant.
  {
    NodeProto node = MakeNode("Or", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_or", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(7);

      const KernelContext or_kernel_ctx{opset};
      const onnx_kernels::kernel::Or or_kernel{or_kernel_ctx};

      Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
      Tensor y("", DataType::BOOL, {2, 2}, {1, 1, 0, 0});
      Tensor z = or_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Scalar broadcast variant: z[i] = x[i] OR y (scalar).
  {
    NodeProto node = MakeNode("Or", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_or_bcast", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(7);

      const KernelContext or_kernel_ctx{opset};
      const onnx_kernels::kernel::Or or_kernel{or_kernel_ctx};

      Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
      Tensor y("", DataType::BOOL, {}, {0});
      Tensor z = or_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Or`` operator (mirror the
  // ``onnx.backend.test.case.node.or_.Or`` Python class). Inputs are
  // generated deterministically through the seeded ``RandBool`` helper to
  // match the upstream ``(randn(...) > 0).astype(bool)`` pattern; expected
  // outputs are computed by ``kernel::Or`` with multidirectional broadcasting.
  //
  // From Or.export():
  RegisterOrOnnxCase("test_or2d", {3, 4}, 101, {3, 4}, 102, opset, registry);
  RegisterOrOnnxCase("test_or3d", {3, 4, 5}, 103, {3, 4, 5}, 104, opset, registry);
  RegisterOrOnnxCase("test_or4d", {3, 4, 5, 6}, 105, {3, 4, 5, 6}, 106, opset, registry);
  // From Or.export_or_broadcast():
  RegisterOrOnnxCase("test_or_bcast3v1d", {3, 4, 5}, 107, {5}, 108, opset, registry);
  RegisterOrOnnxCase("test_or_bcast3v2d", {3, 4, 5}, 109, {4, 5}, 110, opset, registry);
  RegisterOrOnnxCase("test_or_bcast4v2d", {3, 4, 5, 6}, 111, {5, 6}, 112, opset, registry);
  RegisterOrOnnxCase("test_or_bcast4v3d", {3, 4, 5, 6}, 113, {4, 5, 6}, 114, opset, registry);
  RegisterOrOnnxCase("test_or_bcast4v4d", {1, 4, 1, 6}, 115, {3, 1, 5, 6}, 116, opset, registry);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
