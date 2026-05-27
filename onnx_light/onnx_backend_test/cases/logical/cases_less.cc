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
// Less — z = x < y, element-wise with broadcasting (since opset 7).
// Inputs are numeric tensors of the same dtype, the output is BOOL.
// ---------------------------------------------------------------------------
void RegisterLessCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::Less less_kernel{kernel::KernelContext(opset)};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Less");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    Tensor z = less_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_less", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] < y (scalar).
  {
    NodeProto node;
    node.set_op_type("Less");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.5f});
    Tensor z = less_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_less_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Less`` operator (mirror the
  // ``onnx.backend.test.case.node.less.Less`` Python class). All numeric
  // input dtypes accepted by :ref:`kernel::Less` are covered: FLOAT, INT8,
  // INT16, UINT8, UINT16, UINT32 and UINT64. Inputs are generated
  // deterministically through the seeded ``Randn``/``RandInt`` helpers to
  // match the upstream ``np.random.randn(...).astype(dtype)`` and
  // ``np.random.randint(24, size=..., dtype=...)`` patterns; expected outputs
  // are computed by ``kernel::Less``.

  NodeProto node;
  node.set_op_type("Less");
  node.add_input("x");
  node.add_input("y");
  node.add_output("less");

  // From Less.export():
  {
    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/25);
    Tensor y = RandnFloat({3, 4, 5}, /*seed=*/26);
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/51));
    Tensor y = Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/52));
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less_int8", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/53));
    Tensor y = Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/54));
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less_int16", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/55));
    Tensor y = Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/56));
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less_uint8", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/57));
    Tensor y = Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/58));
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less_uint16", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/59));
    Tensor y = Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/60));
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less_uint32", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/61));
    Tensor y = Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/62));
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less_uint64", {opset}, "backend-test", registry);
  }
  // From Less.export_less_broadcast():
  {
    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/27);
    Tensor y = RandnFloat({5}, /*seed=*/28);
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
