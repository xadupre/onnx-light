// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
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
// Sub — z = x - y, element-wise with broadcasting (since opset 14).
// ---------------------------------------------------------------------------
void RegisterSubCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Sub sub_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Sub");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor z = sub_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_sub", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] - y (scalar).
  {
    NodeProto node;
    node.set_op_type("Sub");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {0.5f});
    Tensor z = sub_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_sub_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Sub`` operator (mirror the
  // ``onnx.backend.test.case.node.sub.Sub`` Python class). All numeric input
  // dtypes accepted by :ref:`kernel::Sub` are covered: FLOAT, INT8, INT16,
  // UINT8, UINT16, UINT32 and UINT64. Inputs are generated deterministically
  // through the seeded ``Randn``/``RandnInt``/``RandUint`` helpers to match
  // the upstream ``np.random.randn(...).astype(dtype)`` and
  // ``np.random.randint(..., dtype=...)`` patterns; expected outputs are
  // computed by ``kernel::Sub``.

  // From Sub.export(): the scripted ``test_sub_example`` case.
  {
    NodeProto node;
    node.set_op_type("Sub");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    // test_sub_example: expected z = [-2, 0, 2].
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
    Tensor z = sub_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_sub_example", {opset}, "backend-test", registry);
  }

  NodeProto node;
  node.set_op_type("Sub");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  // Each upstream case is a (test_name, [x, y]) pair; the expected output is
  // computed by ``sub_kernel`` to keep the registry self-consistent.
  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From Sub.export():
      {"test_sub", {RandnFloat({3, 4, 5}, /*seed=*/1), RandnFloat({3, 4, 5}, /*seed=*/2)}},
      {"test_sub_int8",
       {Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/11)),
        Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/12))}},
      {"test_sub_int16",
       {Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/13)),
        Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/14))}},
      {"test_sub_uint8",
       {Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/15)),
        Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(12, {3, 4, 5}, /*seed=*/16))}},
      {"test_sub_uint16",
       {Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/17)),
        Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(12, {3, 4, 5}, /*seed=*/18))}},
      {"test_sub_uint32",
       {Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/19)),
        Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(12, {3, 4, 5}, /*seed=*/20))}},
      {"test_sub_uint64",
       {Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/21)),
        Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(12, {3, 4, 5}, /*seed=*/22))}},
      // From Sub.export_sub_broadcast():
      {"test_sub_bcast", {RandnFloat({3, 4, 5}, /*seed=*/3), RandnFloat({5}, /*seed=*/4)}},
  };

  for (const auto &[name, inputs] : cases) {
    Tensor z = sub_kernel(inputs[0], inputs[1]);
    Expect(node, inputs, {z}, name, {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
