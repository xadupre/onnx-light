// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Build a ``BitCast`` node with the required ``to`` attribute.
NodeProto MakeBitCastNode(int32_t to) {
  NodeProto node = MakeNode("BitCast", {"x"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("to");
  attr->set_type(AttributeProto::INT);
  attr->set_i(static_cast<int64_t>(to));
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// BitCast — y = reinterpret_cast<to>(x), preserving the bit pattern
// (since opset 26). Mirrors the upstream
// ``onnx.backend.test.case.node.bitcast.BitCast`` class.
// ---------------------------------------------------------------------------
void RegisterBitCastCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(26);
  const kernel::KernelContext ctx{opset};
  const kernel::BitCast k{ctx};

  // 32-bit reinterpret: FLOAT <-> INT32 (same bit-width).
  {
    NodeProto node = MakeBitCastNode(DataType::INT32);
    Tensor x = Tensor::FromFloat("", {3}, {0.0f, 1.0f, -1.0f});
    Tensor y = k(x, DataType::INT32);
    Expect(node, {x}, {y}, "test_cc_bitcast_float_to_int32", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeBitCastNode(DataType::FLOAT);
    Tensor x = Tensor::FromInt32("", {3}, {0, 1065353216, -1082130432});
    Tensor y = k(x, DataType::FLOAT);
    Expect(node, {x}, {y}, "test_cc_bitcast_int32_to_float", {opset}, "backend-test", registry);
  }

  // 64-bit reinterpret: DOUBLE <-> INT64.
  {
    NodeProto node = MakeBitCastNode(DataType::INT64);
    Tensor x = Tensor::FromDouble("", {2, 2}, {0.0, 1.0, -1.0, 3.14});
    Tensor y = k(x, DataType::INT64);
    Expect(node, {x}, {y}, "test_cc_bitcast_double_to_int64", {opset}, "backend-test", registry);
  }

  // 16-bit reinterpret: UINT16 <-> INT16.
  {
    NodeProto node = MakeBitCastNode(DataType::INT16);
    Tensor x = Tensor::FromUint16("", {4}, {0u, 0x3C00u, 0x4000u, 0xBC00u});
    Tensor y = k(x, DataType::INT16);
    Expect(node, {x}, {y}, "test_cc_bitcast_uint16_to_int16", {opset}, "backend-test", registry);
  }

  // 8-bit reinterpret: UINT8 <-> INT8 with a deterministic random buffer.
  {
    NodeProto node = MakeBitCastNode(DataType::INT8);
    Tensor x = Tensor::FromUint8("", {3, 4}, RandUint<uint8_t>(256, {3, 4}, 2001));
    Tensor y = k(x, DataType::INT8);
    Expect(node, {x}, {y}, "test_cc_bitcast_uint8_to_int8", {opset}, "backend-test", registry);
  }

  // ---------------------------------------------------------------------------
  // Upstream ONNX node cases — mirrors
  // ``onnx.backend.test.case.node.bitcast.BitCast`` exports.
  // ---------------------------------------------------------------------------

  // float32 -> int32 (1-D).
  {
    NodeProto node = MakeBitCastNode(DataType::INT32);
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, -2.5f, 3.75f});
    Tensor y = k(x, DataType::INT32);
    Expect(node, {x}, {y}, "test_bitcast_float32_to_int32", {opset}, "backend-test", registry);
  }
  // int32 -> float32 (1-D).
  {
    NodeProto node = MakeBitCastNode(DataType::FLOAT);
    Tensor x = Tensor::FromInt32("", {3}, {1065353216, -1071644672, 1081081856});
    Tensor y = k(x, DataType::FLOAT);
    Expect(node, {x}, {y}, "test_bitcast_int32_to_float32", {opset}, "backend-test", registry);
  }
  // float64 -> int64.
  {
    NodeProto node = MakeBitCastNode(DataType::INT64);
    Tensor x = Tensor::FromDouble("", {3}, {1.0, -2.5, 3.75});
    Tensor y = k(x, DataType::INT64);
    Expect(node, {x}, {y}, "test_bitcast_float64_to_int64", {opset}, "backend-test", registry);
  }
  // int64 -> float64.
  {
    NodeProto node = MakeBitCastNode(DataType::DOUBLE);
    Tensor x = Tensor::FromInt64("", {3},
                                 {static_cast<int64_t>(4607182418800017408LL),
                                  static_cast<int64_t>(-4611686018427387904LL),
                                  static_cast<int64_t>(4614256656552045184LL)});
    Tensor y = k(x, DataType::DOUBLE);
    Expect(node, {x}, {y}, "test_bitcast_int64_to_float64", {opset}, "backend-test", registry);
  }
  // uint32 -> int32 (same size, different signedness).
  {
    NodeProto node = MakeBitCastNode(DataType::INT32);
    Tensor x = Tensor::FromUint32("", {3}, {4294967295u, 2147483648u, 2147483647u});
    Tensor y = k(x, DataType::INT32);
    Expect(node, {x}, {y}, "test_bitcast_uint32_to_int32", {opset}, "backend-test", registry);
  }
  // 2-D float32 -> int32.
  {
    NodeProto node = MakeBitCastNode(DataType::INT32);
    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = k(x, DataType::INT32);
    Expect(node, {x}, {y}, "test_bitcast_2d_float32_to_int32", {opset}, "backend-test", registry);
  }
  // int8 -> uint8 (same size, different signedness).
  {
    NodeProto node = MakeBitCastNode(DataType::UINT8);
    Tensor x = Tensor::FromInt8("", {4}, {-1, -128, 127, 0});
    Tensor y = k(x, DataType::UINT8);
    Expect(node, {x}, {y}, "test_bitcast_int8_to_uint8", {opset}, "backend-test", registry);
  }
  // Scalar float32 -> int32.
  {
    NodeProto node = MakeBitCastNode(DataType::INT32);
    Tensor x = Tensor::FromFloat("", {}, {1.0f});
    Tensor y = k(x, DataType::INT32);
    Expect(node, {x}, {y}, "test_bitcast_scalar_float32_to_int32", {opset}, "backend-test",
           registry);
  }
  // bool -> uint8 (same size).
  {
    NodeProto node = MakeBitCastNode(DataType::UINT8);
    Tensor x = Tensor::FromBool("", {4}, {1, 0, 1, 0});
    Tensor y = k(x, DataType::UINT8);
    Expect(node, {x}, {y}, "test_bitcast_bool_to_uint8", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
