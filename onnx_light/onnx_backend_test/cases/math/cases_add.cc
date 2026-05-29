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

} // namespace

// ---------------------------------------------------------------------------
// Add — z = x + y, element-wise with broadcasting (since opset 14).
// This is the case exercised by examples/run_add_node_test/main.cc.
// ---------------------------------------------------------------------------
void RegisterAddCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Add add_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor z = add_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_add", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] + y (scalar).
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {0.5f});
    Tensor z = add_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_add_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Add`` operator (mirror the
  // ``onnx.backend.test.case.node.add.Add`` Python class). All numeric input
  // dtypes accepted by :ref:`kernel::Add` are covered: FLOAT, INT8, INT16,
  // UINT8, UINT16, UINT32 and UINT64. Inputs are generated deterministically
  // through the seeded ``Randn``/``RandnInt``/``RandUint`` helpers to mirror
  // the upstream ``np.random.randn(...)`` and ``np.random.randint(...)``
  // patterns; expected outputs are computed by ``kernel::Add``.

  NodeProto node;
  node.set_op_type("Add");
  node.add_input("x");
  node.add_input("y");
  node.add_output("sum");

  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From Add.export():
      {"test_add", {RandnFloat({3, 4, 5}, /*seed=*/5), RandnFloat({3, 4, 5}, /*seed=*/6)}},
      {"test_add_int8",
       {Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/41)),
        Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/42))}},
      {"test_add_int16",
       {Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/43)),
        Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/44))}},
      {"test_add_uint8",
       {Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/45)),
        Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/46))}},
      {"test_add_uint16",
       {Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/47)),
        Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/48))}},
      {"test_add_uint32",
       {Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/49)),
        Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/50))}},
      {"test_add_uint64",
       {Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/51)),
        Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/52))}},
      // From Add.export_add_broadcast():
      {"test_add_bcast", {RandnFloat({3, 4, 5}, /*seed=*/7), RandnFloat({5}, /*seed=*/8)}},
  };

  for (const auto &[name, inputs] : cases) {
    Tensor z = add_kernel(inputs[0], inputs[1]);
    Expect(node, inputs, {z}, name, {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
