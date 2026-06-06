// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// GreaterOrEqual — z = x >= y, element-wise with broadcasting (since opset 12).
// Inputs are numeric tensors of the same dtype, the output is BOOL.
// ---------------------------------------------------------------------------
void RegisterGreaterOrEqualCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(16);
  const kernel::KernelContext ctx{opset};
  const kernel::GreaterOrEqual ge_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("GreaterOrEqual");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    Tensor z = ge_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_greater_or_equal", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] >= y (scalar).
  {
    NodeProto node;
    node.set_op_type("GreaterOrEqual");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.5f});
    Tensor z = ge_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_greater_or_equal_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``GreaterOrEqual`` operator
  // (mirror the ``onnx.backend.test.case.node.greater_equal.Greater`` Python
  // class). All numeric input dtypes accepted by
  // :ref:`kernel::GreaterOrEqual` are covered: FLOAT, INT8, INT16, UINT8,
  // UINT16, UINT32 and UINT64. Inputs are generated deterministically through
  // the seeded ``Randn``/``RandInt`` helpers to match the upstream
  // ``np.random.randn(...).astype(dtype)`` and
  // ``np.random.randint(24, size=..., dtype=...)`` patterns; expected outputs
  // are computed by ``kernel::GreaterOrEqual``.

  NodeProto node;
  node.set_op_type("GreaterOrEqual");
  node.add_input("x");
  node.add_input("y");
  node.add_output("greater_equal");

  // Each upstream case is a (test_name, [x, y]) pair; the expected output is
  // computed by ``ge_kernel`` to keep the registry self-consistent.
  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From Greater.export() in greater_equal.py:
      {"test_greater_equal",
       {RandnFloat({3, 4, 5}, /*seed=*/121), RandnFloat({3, 4, 5}, /*seed=*/122)}},
      {"test_greater_equal_int8",
       {Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/131)),
        Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/132))}},
      {"test_greater_equal_int16",
       {Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/133)),
        Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/134))}},
      {"test_greater_equal_uint8",
       {Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/135)),
        Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/136))}},
      {"test_greater_equal_uint16",
       {Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/137)),
        Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/138))}},
      {"test_greater_equal_uint32",
       {Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/139)),
        Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/140))}},
      {"test_greater_equal_uint64",
       {Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/141)),
        Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/142))}},
      // From Greater.export_greater_broadcast() in greater_equal.py:
      {"test_greater_equal_bcast",
       {RandnFloat({3, 4, 5}, /*seed=*/123), RandnFloat({5}, /*seed=*/124)}},
  };

  for (const auto &[name, inputs] : cases) {
    Tensor z = ge_kernel(inputs[0], inputs[1]);
    Expect(node, inputs, {z}, name, {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
