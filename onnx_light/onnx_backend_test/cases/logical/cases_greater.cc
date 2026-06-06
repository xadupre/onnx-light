// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/random.h"

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
// Greater — z = x > y, element-wise with broadcasting (since opset 7).
// Inputs are numeric tensors of the same dtype, the output is BOOL.
// ---------------------------------------------------------------------------
void RegisterGreaterCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Greater greater_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    Tensor z = greater_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_greater", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] > y (scalar).
  {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.5f});
    Tensor z = greater_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_greater_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Greater`` operator (mirror the
  // ``onnx.backend.test.case.node.greater.Greater`` Python class). All numeric
  // input dtypes accepted by :ref:`kernel::Greater` are covered: FLOAT,
  // INT8, INT16, UINT8, UINT16, UINT32 and UINT64. Inputs are generated
  // deterministically through the seeded ``Randn``/``RandInt`` helpers to
  // match the upstream ``np.random.randn(...).astype(dtype)`` and
  // ``np.random.randint(24, size=..., dtype=...)`` patterns; expected outputs
  // are computed by ``kernel::Greater``.

  NodeProto node;
  node.set_op_type("Greater");
  node.add_input("x");
  node.add_input("y");
  node.add_output("greater");

  // Each upstream case is a (test_name, [x, y]) pair; the expected output is
  // computed by ``greater_kernel`` to keep the registry self-consistent.
  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From Greater.export():
      {"test_greater", {RandnFloat({3, 4, 5}, /*seed=*/21), RandnFloat({3, 4, 5}, /*seed=*/22)}},
      {"test_greater_int8",
       {Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/31)),
        Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/32))}},
      {"test_greater_int16",
       {Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/33)),
        Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/34))}},
      {"test_greater_uint8",
       {Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/35)),
        Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/36))}},
      {"test_greater_uint16",
       {Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/37)),
        Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/38))}},
      {"test_greater_uint32",
       {Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/39)),
        Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/40))}},
      {"test_greater_uint64",
       {Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/41)),
        Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/42))}},
      // From Greater.export_greater_broadcast():
      {"test_greater_bcast", {RandnFloat({3, 4, 5}, /*seed=*/23), RandnFloat({5}, /*seed=*/24)}},
  };

  for (const auto &[name, inputs] : cases) {
    Tensor z = greater_kernel(inputs[0], inputs[1]);
    Expect(node, inputs, {z}, name, {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
