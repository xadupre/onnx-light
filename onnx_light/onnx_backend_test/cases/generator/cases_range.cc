// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IEEE-754 binary16 encoder (round-to-nearest-even). Duplicated locally
// to keep this case file self-contained, mirroring the helpers in
// ``cases_mod`` / ``cases_attention`` / ``cases_dequantizelinear``.
uint16_t FloatToFloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t e32 = static_cast<int32_t>((u >> 23) & 0xffu);
  const uint32_t m32 = u & 0x007fffffu;
  if (e32 == 0xff) {
    return static_cast<uint16_t>(sign | 0x7c00u | (m32 != 0 ? 0x0200u : 0u));
  }
  const int32_t e = e32 - 127 + 15;
  if (e >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (e <= 0) {
    if (e < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t m = (m32 | 0x00800000u) >> static_cast<uint32_t>(1 - e);
    const uint32_t round_bit = (m >> 12) & 1u;
    const uint32_t sticky = m & 0x00000fffu;
    uint16_t h = static_cast<uint16_t>(sign | (m >> 13));
    if (round_bit && (sticky != 0 || (h & 1))) {
      h = static_cast<uint16_t>(h + 1);
    }
    return h;
  }
  const uint32_t low = m32 & 0x1fffu;
  uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m32 >> 13));
  if (low > 0x1000u || (low == 0x1000u && (h & 1u))) {
    h = static_cast<uint16_t>(h + 1);
  }
  return h;
}

// ``float`` -> ``bfloat16`` round-to-nearest-even encoder.
uint16_t FloatToBfloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
    return static_cast<uint16_t>((u >> 16) | 0x0040u);
  }
  const uint32_t rounding_bias = 0x00007fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

// Builds a FLOAT16 scalar tensor from a ``float`` sample value.
Tensor MakeFloat16Scalar(const std::string &name, float value) {
  Tensor t = Tensor::FromUint16(name, {}, {FloatToFloat16Bits(value)});
  t.data_type = static_cast<int32_t>(DataType::FLOAT16);
  return t;
}

// Builds a BFLOAT16 scalar tensor from a ``float`` sample value.
Tensor MakeBfloat16Scalar(const std::string &name, float value) {
  Tensor t = Tensor::FromUint16(name, {}, {FloatToBfloat16Bits(value)});
  t.data_type = static_cast<int32_t>(DataType::BFLOAT16);
  return t;
}

} // namespace

void RegisterRangeCases(std::vector<TestCase> &registry) {
  const OpsetId opset_v11 = DefaultOpset(11);
  const OpsetId opset_v27 = DefaultOpset(27);
  const kernel::KernelContext ctx_v11{opset_v11};
  const kernel::KernelContext ctx_v27{opset_v27};

  // Upstream test: range_float_type_positive_delta
  // start=1, limit=5, delta=2  ->  [1.0, 3.0]
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");

    const Tensor start = Tensor::FromFloat("start", {}, {1.0f});
    const Tensor limit = Tensor::FromFloat("limit", {}, {5.0f});
    const Tensor delta = Tensor::FromFloat("delta", {}, {2.0f});
    const Tensor output = kernel::Range(ctx_v11)(start, limit, delta);
    Expect(node, {start, limit, delta}, {output}, "test_range_float_type_positive_delta",
           {opset_v11}, "backend-test", registry);
  }

  // Upstream test: range_int32_type_negative_delta
  // start=10, limit=6, delta=-3  ->  [10, 7]
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");

    const Tensor start = Tensor::FromInt32("start", {}, {10});
    const Tensor limit = Tensor::FromInt32("limit", {}, {6});
    const Tensor delta = Tensor::FromInt32("delta", {}, {-3});
    const Tensor output = kernel::Range(ctx_v11)(start, limit, delta);
    Expect(node, {start, limit, delta}, {output}, "test_range_int32_type_negative_delta",
           {opset_v11}, "backend-test", registry);
  }

  // Upstream test (opset 27): range_float16_type_positive_delta
  // start=1, limit=5, delta=2  ->  [1.0, 3.0] as float16
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");

    const Tensor start = MakeFloat16Scalar("start", 1.0f);
    const Tensor limit = MakeFloat16Scalar("limit", 5.0f);
    const Tensor delta = MakeFloat16Scalar("delta", 2.0f);
    const Tensor output = kernel::Range(ctx_v27)(start, limit, delta);
    Expect(node, {start, limit, delta}, {output}, "test_range_float16_type_positive_delta",
           {opset_v27}, "backend-test", registry);
  }

  // Upstream test (opset 27): range_bfloat16_type_positive_delta
  // start=1, limit=5, delta=2  ->  [1.0, 3.0] as bfloat16
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");

    const Tensor start = MakeBfloat16Scalar("start", 1.0f);
    const Tensor limit = MakeBfloat16Scalar("limit", 5.0f);
    const Tensor delta = MakeBfloat16Scalar("delta", 2.0f);
    const Tensor output = kernel::Range(ctx_v27)(start, limit, delta);
    Expect(node, {start, limit, delta}, {output}, "test_range_bfloat16_type_positive_delta",
           {opset_v27}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
