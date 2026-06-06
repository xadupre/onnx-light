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

// Registers one ``test_not*`` upstream-ONNX case: x and y = NOT x.
void RegisterNotOnnxCase(const std::string &name, const std::vector<int64_t> &shape, uint64_t seed,
                         const kernel::Not &not_kernel, const OpsetId &opset,
                         std::vector<TestCase> &registry) {
  NodeProto node = MakeNode("Not", {"x"}, {"not"});

  Tensor x = RandBool(shape, seed);
  Tensor y = not_kernel(x);

  Expect(node, {x}, {y}, name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// Not — y = NOT x, element-wise on a BOOL tensor (opset 1).
// Registers a small deterministic ``test_cc_not`` case and the upstream ONNX
// backend test cases (``test_not_2d``, ``test_not_3d``, ``test_not_4d``)
// mirrored from ``onnx.backend.test.case.node.not.Not``.
// ---------------------------------------------------------------------------
void RegisterNotCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(1);
  const kernel::KernelContext ctx{opset};
  const kernel::Not not_kernel{ctx};

  {
    NodeProto node = MakeNode("Not", {"x"}, {"y"});

    Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y = not_kernel(x);

    Expect(node, {x}, {y}, "test_cc_not", {opset}, "backend-test", registry);
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

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
