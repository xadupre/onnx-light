// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
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
// Mul — z = x * y, element-wise with broadcasting (since opset 14).
// ---------------------------------------------------------------------------
void RegisterMulCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Mul mul_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Mul");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor z = mul_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_mul", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant.
  {
    NodeProto node;
    node.set_op_type("Mul");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.0f});
    Tensor z = mul_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_mul_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Mul`` operator (mirror the
  // ``onnx.backend.test.case.node.mul.Mul`` Python class). All numeric input
  // dtypes accepted by :ref:`kernel::Mul` are covered: FLOAT, INT8, INT16,
  // UINT8, UINT16, UINT32 and UINT64. Inputs are generated deterministically
  // through the seeded ``Randn``/``RandnInt``/``RandUint`` helpers to mirror
  // the upstream ``np.random.randn(...)`` and ``np.random.randint(...)``
  // patterns; expected outputs are computed by ``kernel::Mul``.

  NodeProto node;
  node.set_op_type("Mul");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  // ``test_mul_example`` is a scalar-free fixed-vector case; the remaining
  // entries follow the seeded ``Randn``/``RandnInt``/``RandUint`` pattern.
  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From Mul.export():
      {"test_mul_example",
       {Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f}),
        Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f})}},
      {"test_mul", {RandnFloat({3, 4, 5}, /*seed=*/17), RandnFloat({3, 4, 5}, /*seed=*/18)}},
      {"test_mul_int8",
       {Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/41)),
        Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/42))}},
      {"test_mul_int16",
       {Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/43)),
        Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/44))}},
      {"test_mul_uint8",
       {Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(4, {3, 4, 5}, /*seed=*/45)),
        Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/46))}},
      {"test_mul_uint16",
       {Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(4, {3, 4, 5}, /*seed=*/47)),
        Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/48))}},
      {"test_mul_uint32",
       {Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(4, {3, 4, 5}, /*seed=*/49)),
        Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/50))}},
      {"test_mul_uint64",
       {Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(4, {3, 4, 5}, /*seed=*/51)),
        Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/52))}},
      // From Mul.export_mul_broadcast():
      {"test_mul_bcast", {RandnFloat({3, 4, 5}, /*seed=*/19), RandnFloat({5}, /*seed=*/20)}},
  };

  for (const auto &[name, inputs] : cases) {
    Tensor z = mul_kernel(inputs[0], inputs[1]);
    Expect(node, inputs, {z}, name, {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
