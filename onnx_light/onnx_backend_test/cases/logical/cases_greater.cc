// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/random.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
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
  const kernel::Greater greater_kernel{kernel::KernelContext(opset)};

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

  // From Greater.export():
  {
    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/21);
    Tensor y = RandnFloat({3, 4, 5}, /*seed=*/22);
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/31));
    Tensor y = Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/32));
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater_int8", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/33));
    Tensor y = Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/34));
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater_int16", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/35));
    Tensor y = Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/36));
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater_uint8", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/37));
    Tensor y = Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/38));
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater_uint16", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/39));
    Tensor y = Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/40));
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater_uint32", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/41));
    Tensor y = Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/42));
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater_uint64", {opset}, "backend-test", registry);
  }
  // From Greater.export_greater_broadcast():
  {
    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/23);
    Tensor y = RandnFloat({5}, /*seed=*/24);
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
