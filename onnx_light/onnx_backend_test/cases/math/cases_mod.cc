// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Builds a ``Mod`` NodeProto. ``fmod`` is the ``int`` attribute documented by
// the ONNX schema: when omitted (``fmod == 0``) the operator behaves like
// Python/numpy ``mod`` (sign follows divisor, integer-only); when set to
// ``1`` it behaves like C ``fmod`` (sign follows dividend, both integer and
// floating-point inputs accepted).
NodeProto MakeModNode(int64_t fmod) {
  NodeProto node;
  node.set_op_type("Mod");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");
  if (fmod != 0) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("fmod");
    attr->set_type(AttributeProto::INT);
    attr->set_i(fmod);
  }
  return node;
}

std::vector<int32_t> Arange30() {
  std::vector<int32_t> values(30);
  std::iota(values.begin(), values.end(), 0);
  return values;
}

} // namespace

// ---------------------------------------------------------------------------
// Mod — z = x mod y, element-wise with broadcasting (since opset 13).
// ---------------------------------------------------------------------------
void RegisterModCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Mod mod_kernel{ctx};

  // Upstream ONNX backend test cases for the ``Mod`` operator (mirror the
  // ``onnx.backend.test.case.node.mod.Mod`` Python class). Floating-point
  // variants always set ``fmod=1``; integer variants exercise both the
  // default ``mod`` semantics and the ``fmod=1`` variant.

  // From Mod.export_mod_mixed_sign_float32() / _float64().
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Tensor x = Tensor::FromFloat("", {6}, {-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
    Tensor y = Tensor::FromFloat("", {6}, {2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
    Tensor z = mod_kernel(x, y, /*fmod=*/1);
    Expect(node, {x, y}, {z}, "test_mod_mixed_sign_float32", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Tensor x = Tensor::FromDouble("", {6}, {-4.3, 7.2, 5.0, 4.3, -7.2, 8.0});
    Tensor y = Tensor::FromDouble("", {6}, {2.1, -3.4, 8.0, -2.1, 3.4, 5.0});
    Tensor z = mod_kernel(x, y, /*fmod=*/1);
    Expect(node, {x, y}, {z}, "test_mod_mixed_sign_float64", {opset}, "backend-test", registry);
  }

  // From Mod.export_mod_mixed_sign_int{8,16,32,64}().
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromInt8("", {6}, {-4, 7, 5, 4, -7, 8});
    Tensor y = Tensor::FromInt8("", {6}, {2, -3, 8, -2, 3, 5});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_mixed_sign_int8", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromInt16("", {6}, {-4, 7, 5, 4, -7, 8});
    Tensor y = Tensor::FromInt16("", {6}, {2, -3, 8, -2, 3, 5});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_mixed_sign_int16", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromInt32("", {6}, {-4, 7, 5, 4, -7, 8});
    Tensor y = Tensor::FromInt32("", {6}, {2, -3, 8, -2, 3, 5});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_mixed_sign_int32", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromInt64("", {6}, {-4, 7, 5, 4, -7, 8});
    Tensor y = Tensor::FromInt64("", {6}, {2, -3, 8, -2, 3, 5});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_mixed_sign_int64", {opset}, "backend-test", registry);
  }

  // From Mod.export_mod_uint{8,16,32,64}().
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromUint8("", {3}, {4, 7, 5});
    Tensor y = Tensor::FromUint8("", {3}, {2, 3, 8});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_uint8", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromUint16("", {3}, {4, 7, 5});
    Tensor y = Tensor::FromUint16("", {3}, {2, 3, 8});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_uint16", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromUint32("", {3}, {4u, 7u, 5u});
    Tensor y = Tensor::FromUint32("", {3}, {2u, 3u, 8u});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_uint32", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromUint64("", {3}, {4ull, 7ull, 5ull});
    Tensor y = Tensor::FromUint64("", {3}, {2ull, 3ull, 8ull});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_uint64", {opset}, "backend-test", registry);
  }

  // From Mod.export_mod_int64_fmod() — sign follows dividend.
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Tensor x = Tensor::FromInt64("", {6}, {-4, 7, 5, 4, -7, 8});
    Tensor y = Tensor::FromInt64("", {6}, {2, -3, 8, -2, 3, 5});
    Tensor z = mod_kernel(x, y, /*fmod=*/1);
    Expect(node, {x, y}, {z}, "test_mod_int64_fmod", {opset}, "backend-test", registry);
  }

  // From Mod.export_mod_broadcast() — int32, scalar-ish divisor.
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Tensor x = Tensor::FromInt32("", {3, 2, 5}, Arange30());
    Tensor y = Tensor::FromInt32("", {1}, {7});
    Tensor z = mod_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mod_broadcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
