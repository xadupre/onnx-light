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

// Registers one ``test_not*`` upstream-ONNX case: x and y = NOT x.
void RegisterNotOnnxCase(const std::string &name, const std::vector<int64_t> &shape, uint64_t seed,
                         const auto &not_kernel, const OpsetId &opset,
                         std::vector<TestCase> &registry) {
  NodeProto node = MakeNode("Not", {"x"}, {"not"});
  Expect(registry, std::move(node), name, {opset}, [=]() -> IoData {
    Tensor x = RandBool(shape, seed);
    Tensor y = not_kernel.Invoke([&](const auto &kernel) { return kernel(x); });

    return IoData{{std::move(x)}, {std::move(y)}};
  });
}

} // namespace

// ---------------------------------------------------------------------------
// Not — y = NOT x, element-wise on a BOOL tensor (opset 1).
// Registers a small deterministic ``test_cc_not`` case and the upstream ONNX
// backend test cases (``test_not_2d``, ``test_not_3d``, ``test_not_4d``)
// mirrored from ``onnx.backend.test.case.node.not.Not``.
// ---------------------------------------------------------------------------
void RegisterNotCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(1);
  const auto not_kernel = MakeReferenceKernel<onnx_kernels::kernel::Not>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("Not", {"x"}, {"y"});

    const std::vector<int64_t> shape = {1024, 4096};
    const int64_t count = 1024 * 4096;
    Expect(registry, std::move(node), "test_cc_not_benchmark", {opset}, {count}, {count},
           [not_kernel, shape]() -> IoData {
             Tensor x = RandBool(shape, /*seed=*/9103);
             Tensor y = not_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  {
    NodeProto node = MakeNode("Not", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_cc_not", {opset}, [=]() -> IoData {
      Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
      Tensor y = not_kernel.Invoke([&](const auto &kernel) { return kernel(x); });

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Not`` operator (mirror the
  // ``onnx.backend.test.case.node.not.Not`` Python class). Inputs are
  // generated deterministically through the seeded ``RandBool`` helper to
  // match the upstream ``(randn(...) > 0).astype(bool)`` pattern; expected
  // outputs are computed by ``kernel::Not``.
  RegisterNotOnnxCase("test_not_2d", {3, 4}, 101, not_kernel, opset, registry);
  RegisterNotOnnxCase("test_not_3d", {3, 4, 5}, 102, not_kernel, opset, registry);
  RegisterNotOnnxCase("test_not_4d", {3, 4, 5, 6}, 103, not_kernel, opset, registry);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
