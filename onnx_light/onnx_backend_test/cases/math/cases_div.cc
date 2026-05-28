// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/random.h"
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

Tensor RandFloatUnitOffset(const std::vector<int64_t> &shape, uint64_t seed) {
  // Mirrors the upstream ``np.random.rand(...) + 1.0`` pattern: values are
  // pseudo-random doubles in ``[1, 2)``, guaranteed to avoid division by zero.
  const std::vector<double> values = Rand(shape, seed);
  std::vector<float> floats(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    floats[i] = static_cast<float>(values[i] + 1.0);
  }
  return Tensor::FromFloat("", shape, floats);
}

// Builds a ``RandnInt<TInt>``-distributed divisor and guarantees it is
// non-zero, matching the spirit of the upstream ``+ 1`` shift used to avoid
// division by zero. Any zero value is replaced by ``1``.
template <typename TInt>
std::vector<TInt> RandnIntNonZero(const std::vector<int64_t> &shape, uint64_t seed) {
  std::vector<TInt> values = RandnInt<TInt>(shape, seed);
  for (auto &v : values) {
    if (v == 0) {
      v = static_cast<TInt>(1);
    }
  }
  return values;
}

// Builds a ``RandUint<TUInt>``-distributed divisor in ``[1, high]`` by
// drawing in ``[0, high)`` and adding ``1``, mirroring the upstream
// ``np.random.randint(high, ...) + 1`` pattern used by the ``Div`` uint
// cases. Always non-zero by construction.
template <typename TUInt>
std::vector<TUInt> RandUintNonZero(int64_t high, const std::vector<int64_t> &shape, uint64_t seed) {
  std::vector<TUInt> values = RandUint<TUInt>(high, shape, seed);
  for (auto &v : values) {
    v = static_cast<TUInt>(v + 1);
  }
  return values;
}

} // namespace

// ---------------------------------------------------------------------------
// Div — z = x / y, element-wise with broadcasting (since opset 14).
// ---------------------------------------------------------------------------
void RegisterDivCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Div div_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor y = Tensor::FromFloat("", {2, 3}, {2.0f, 4.0f, 5.0f, 8.0f, 10.0f, 12.0f});
    Tensor z = div_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_div", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant.
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.0f});
    Tensor z = div_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_div_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Div`` operator (mirror the
  // ``onnx.backend.test.case.node.div.Div`` Python class). All numeric input
  // dtypes accepted by :ref:`kernel::Div` are covered: FLOAT, INT8, INT16,
  // INT32 (the ``test_div_int32_trunc`` fixed-vector case exercising
  // truncating signed division), UINT8, UINT16, UINT32 and UINT64. The
  // divisor for the integer variants is shifted by ``+1`` to mirror the
  // upstream ``np.random.randint(24, ...) + 1`` pattern, which guarantees a
  // non-zero divisor.

  NodeProto node;
  node.set_op_type("Div");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From Div.export():
      {"test_div_example",
       {Tensor::FromFloat("", {2}, {3.0f, 4.0f}), Tensor::FromFloat("", {2}, {1.0f, 2.0f})}},
      {"test_div",
       {RandnFloat({3, 4, 5}, /*seed=*/33), RandFloatUnitOffset({3, 4, 5}, /*seed=*/34)}},
      {"test_div_int8",
       {Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/61)),
        Tensor::FromInt8("", {3, 4, 5}, RandnIntNonZero<int8_t>({3, 4, 5}, /*seed=*/62))}},
      {"test_div_int16",
       {Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/63)),
        Tensor::FromInt16("", {3, 4, 5}, RandnIntNonZero<int16_t>({3, 4, 5}, /*seed=*/64))}},
      // ``test_div_int32_trunc`` uses a hand-rolled fixed-vector pair that
      // exercises truncation toward zero (e.g. ``-3 / 2 == -1``).
      {"test_div_int32_trunc",
       {Tensor::FromInt32("", {4}, {-3, 3, -3, 3}), Tensor::FromInt32("", {4}, {2, 2, -2, -2})}},
      {"test_div_uint8",
       {Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/65)),
        Tensor::FromUint8("", {3, 4, 5}, RandUintNonZero<uint8_t>(24, {3, 4, 5}, /*seed=*/66))}},
      {"test_div_uint16",
       {Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/67)),
        Tensor::FromUint16("", {3, 4, 5}, RandUintNonZero<uint16_t>(24, {3, 4, 5}, /*seed=*/68))}},
      {"test_div_uint32",
       {Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/69)),
        Tensor::FromUint32("", {3, 4, 5}, RandUintNonZero<uint32_t>(24, {3, 4, 5}, /*seed=*/70))}},
      {"test_div_uint64",
       {Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/71)),
        Tensor::FromUint64("", {3, 4, 5}, RandUintNonZero<uint64_t>(24, {3, 4, 5}, /*seed=*/72))}},
      // From Div.export_div_broadcast():
      {"test_div_bcast",
       {RandnFloat({3, 4, 5}, /*seed=*/35), RandFloatUnitOffset({5}, /*seed=*/36)}},
  };

  for (const auto &[name, inputs] : cases) {
    Tensor z = div_kernel(inputs[0], inputs[1]);
    Expect(node, inputs, {z}, name, {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
