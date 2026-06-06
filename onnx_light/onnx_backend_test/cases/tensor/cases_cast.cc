// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Cast — element-wise conversion to the dtype carried by the required
// integer attribute ``to`` (since opset 13 in the ai.onnx domain).
//
// The cases below mirror the upstream ``test_cast_<FROM>_to_<TO>`` node
// tests for every element type supported by the backend test ``Tensor``
// library and by :ref:`kernel::Cast`: the numeric types ``FLOAT``,
// ``DOUBLE``, ``INT32``, ``INT64``, ``INT8``, ``UINT8``, ``INT16``,
// ``UINT16``, ``BOOL`` and the variable-length ``STRING`` type. All
// cross-dtype permutations are registered so that ``Cast`` coverage is
// complete with respect to what the reference kernel accepts. Inputs are
// small, fully deterministic vectors so the expected outputs can be
// computed directly by :ref:`kernel::Cast`.
//
// In addition to the numeric+STRING grid, the float8 variants
// ``FLOAT8E4M3FN``, ``FLOAT8E4M3FNUZ``, ``FLOAT8E5M2`` and
// ``FLOAT8E5M2FNUZ`` are registered as ``FLOAT`` ↔ ``FLOAT8*`` pairs
// (mirroring the corresponding upstream ``test_cast`` cases — the
// upstream ``FLOAT16`` peer is intentionally omitted because the
// backend test ``Tensor`` storage does not yet support ``FLOAT16``).
// The sub-byte packed integer variants ``INT4`` / ``UINT4`` / ``INT2`` /
// ``UINT2`` are registered as ``FLOAT`` ↔ packed and packed ↔ companion
// whole-byte integer (``INT8`` / ``UINT8``) pairs, again mirroring the
// upstream ``test_cast`` coverage minus the ``FLOAT16`` peer. Upstream
// cases over ``BFLOAT16`` and ``FLOAT4E2M1`` are still intentionally
// omitted: those element types are not supported by the backend test
// ``Tensor`` storage nor by :ref:`kernel::Cast`, so they would need to
// be added at the kernel layer first.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeCastNode(int64_t to) {
  NodeProto node;
  node.set_op_type("Cast");
  node.add_input("input");
  node.add_output("output");
  AddAttribute<int64_t>(node, "to", to);
  return node;
}

struct CastDtype {
  DataType dtype;
  const char *name;
  std::function<Tensor()> make_input;
};

// Per-source-dtype deterministic inputs. Values are chosen to fit in the
// narrowest representable integer dtype (``INT8`` / ``UINT8``) so that the
// reference kernel's narrowing casts do not depend on undefined-behaviour
// overflow paths. ``BOOL`` uses {0, 1, 1, 0} so that both truthy and falsy
// elements appear in every cast.
std::vector<CastDtype> SupportedCastDtypes() {
  return {
      {DataType::FLOAT, "FLOAT",
       []() { return Tensor::FromFloat("", {4}, {-1.5f, 0.0f, 2.75f, 4.0f}); }},
      {DataType::DOUBLE, "DOUBLE",
       []() { return Tensor::FromDouble("", {4}, {-1.5, 0.0, 2.75, 4.0}); }},
      {DataType::INT32, "INT32", []() { return Tensor::FromInt32("", {4}, {-3, 0, 7, 42}); }},
      {DataType::INT64, "INT64", []() { return Tensor::FromInt64("", {4}, {-3, 0, 7, 42}); }},
      {DataType::INT8, "INT8", []() { return Tensor::FromInt8("", {4}, {-3, 0, 7, 42}); }},
      {DataType::UINT8, "UINT8", []() { return Tensor::FromUint8("", {4}, {0, 1, 7, 42}); }},
      {DataType::INT16, "INT16", []() { return Tensor::FromInt16("", {4}, {-3, 0, 7, 42}); }},
      {DataType::UINT16, "UINT16", []() { return Tensor::FromUint16("", {4}, {0, 1, 7, 42}); }},
      {DataType::BOOL, "BOOL", []() { return Tensor::FromBool("", {4}, {0, 1, 1, 0}); }},
      {DataType::STRING, "STRING",
       []() { return Tensor::FromStrings("", {4}, {"-3", "0", "7", "42"}); }},
  };
}

} // namespace

void RegisterCastCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Cast cast_kernel{ctx};

  const auto dtypes = SupportedCastDtypes();
  for (const auto &from : dtypes) {
    for (const auto &to : dtypes) {
      if (from.dtype == to.dtype) {
        // Skip identity casts: upstream ONNX node tests skip them as well.
        continue;
      }
      const int64_t to_attr = static_cast<int64_t>(to.dtype);
      NodeProto node = MakeCastNode(to_attr);
      Tensor input = from.make_input();
      Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
      Expect(node, {input}, {output}, std::string("test_cc_cast_") + from.name + "_to_" + to.name,
             {opset}, "backend-test", registry);
    }
  }

  // ---------------------------------------------------------------------
  // Float8 cases — FLOAT ↔ FLOAT8*.
  //
  // The 15-element FLOAT input mirrors the vector used by the upstream
  // ``test_cast_FLOAT_to_FLOAT8*`` node tests. It exercises the
  // round-to-nearest-even rounding paths in the normal range, the
  // subnormal underflow region, and the +/-infinity and NaN saturation
  // paths.
  // ---------------------------------------------------------------------
  const std::vector<int64_t> f8_shape = {3, 5};
  // ``std::nanf("")`` returns the canonical positive quiet NaN on every
  // IEEE 754 platform the project targets (sign 0, exponent all-ones,
  // most-significant mantissa bit set). ``kernel::Cast``'s float8
  // conversion routines only inspect the all-ones exponent and the
  // mantissa-nonzero predicate, so the resulting float8 NaN bit pattern
  // is deterministic and matches ``onnx.numpy_helper.saturate_cast``.
  const std::vector<float> f8_fp32_values = {
      0.47892547f,
      0.48033667f,
      0.49968487f,
      0.81910545f,
      0.47031248f,
      0.7229038f,
      1000000.0f,
      1e-7f,
      std::nanf(""),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -1e-7f,
      1e-7f,
      -1000000.0f,
  };
  struct Float8Variant {
    DataType dtype;
    const char *name;
  };
  const Float8Variant kFloat8Variants[] = {
      {DataType::FLOAT8E4M3FN, "FLOAT8E4M3FN"},
      {DataType::FLOAT8E4M3FNUZ, "FLOAT8E4M3FNUZ"},
      {DataType::FLOAT8E5M2, "FLOAT8E5M2"},
      {DataType::FLOAT8E5M2FNUZ, "FLOAT8E5M2FNUZ"},
  };
  for (const auto &v : kFloat8Variants) {
    // FLOAT -> FLOAT8*
    {
      const int64_t to_attr = static_cast<int64_t>(v.dtype);
      NodeProto node = MakeCastNode(to_attr);
      Tensor input = Tensor::FromFloat("", f8_shape, f8_fp32_values);
      Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
      Expect(node, {input}, {output}, std::string("test_cc_cast_FLOAT_to_") + v.name, {opset},
             "backend-test", registry);
    }
    // FLOAT8* -> FLOAT — input bytes are the saturated FLOAT8 encoding
    // of the same FP32 vector (matches the upstream behaviour where
    // ``np_from = saturate_cast(np_fp32, from_np_dtype)`` is fed into
    // the node).
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT);
      NodeProto node = MakeCastNode(to_attr);
      Tensor encoded = cast_kernel(Tensor::FromFloat("", f8_shape, f8_fp32_values),
                                   static_cast<int32_t>(v.dtype));
      Tensor input("", static_cast<int32_t>(v.dtype), f8_shape, encoded.data);
      Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
      Expect(node, {input}, {output}, std::string("test_cc_cast_") + v.name + "_to_FLOAT", {opset},
             "backend-test", registry);
    }
  }

  // ---------------------------------------------------------------------
  // Sub-byte (INT4 / UINT4 / INT2 / UINT2) cases.
  //
  // Mirrors the upstream ``test_cast_<FROM>_to_<TO>`` node tests that
  // exercise the 4-bit and 2-bit packed integer dtypes. The upstream
  // ``FLOAT16`` peer cases are intentionally omitted because the backend
  // test ``Tensor`` storage does not yet support ``FLOAT16``. Inputs use
  // the same ``np.arange`` vectors as the upstream generator so the
  // saturating-cast paths (values below/above the destination range) and
  // the typical in-range values are exercised.
  // ---------------------------------------------------------------------
  struct SubByteVariant {
    DataType dtype;
    const char *name;
    DataType wide_int_dtype; // INT8 (signed) or UINT8 (unsigned)
    const char *wide_int_name;
  };
  const SubByteVariant kInt4Variants[] = {
      {DataType::UINT4, "UINT4", DataType::UINT8, "UINT8"},
      {DataType::INT4, "INT4", DataType::INT8, "INT8"},
  };
  const SubByteVariant kInt2Variants[] = {
      {DataType::UINT2, "UINT2", DataType::UINT8, "UINT8"},
      {DataType::INT2, "INT2", DataType::INT8, "INT8"},
  };

  // INT4 / UINT4 — input shape (5, 5) with the 25-element ``np.arange(-9, 16)``
  // sweep used by the upstream generator. ``np.arange`` returns ``int64`` by
  // default; the upstream test casts the values to ``float32`` before feeding
  // them to the node, which is what ``Tensor::FromFloat`` does here.
  const std::vector<int64_t> int4_shape = {5, 5};
  std::vector<float> int4_fp32_values(25);
  for (int i = 0; i < 25; ++i) {
    int4_fp32_values[static_cast<size_t>(i)] = static_cast<float>(i - 9);
  }
  for (const auto &v : kInt4Variants) {
    // FLOAT -> sub-byte
    {
      const int64_t to_attr = static_cast<int64_t>(v.dtype);
      NodeProto node = MakeCastNode(to_attr);
      Tensor input = Tensor::FromFloat("", int4_shape, int4_fp32_values);
      Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
      Expect(node, {input}, {output}, std::string("test_cc_cast_FLOAT_to_") + v.name, {opset},
             "backend-test", registry);
    }
    // sub-byte -> FLOAT — input bytes are the saturated/packed encoding of
    // the same FP32 vector (matching upstream where the input is the result
    // of ``np_fp32.astype(sub_byte_dtype)``).
    Tensor packed_input;
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT);
      NodeProto node = MakeCastNode(to_attr);
      Tensor encoded = cast_kernel(Tensor::FromFloat("", int4_shape, int4_fp32_values),
                                   static_cast<int32_t>(v.dtype));
      packed_input = Tensor("", static_cast<int32_t>(v.dtype), int4_shape, encoded.data);
      Tensor output = cast_kernel(packed_input, static_cast<int32_t>(to_attr));
      Expect(node, {packed_input}, {output}, std::string("test_cc_cast_") + v.name + "_to_FLOAT",
             {opset}, "backend-test", registry);
    }
    // sub-byte -> companion whole-byte integer (INT4->INT8, UINT4->UINT8).
    {
      const int64_t to_attr = static_cast<int64_t>(v.wide_int_dtype);
      NodeProto node = MakeCastNode(to_attr);
      Tensor output = cast_kernel(packed_input, static_cast<int32_t>(to_attr));
      Expect(node, {packed_input}, {output},
             std::string("test_cc_cast_") + v.name + "_to_" + v.wide_int_name, {opset},
             "backend-test", registry);
    }
  }

  // INT2 / UINT2 — input shape (7, 1) with the 7-element ``np.arange(-3, 4)``
  // sweep used by the upstream generator.
  const std::vector<int64_t> int2_shape = {7, 1};
  std::vector<float> int2_fp32_values(7);
  for (int i = 0; i < 7; ++i) {
    int2_fp32_values[static_cast<size_t>(i)] = static_cast<float>(i - 3);
  }
  for (const auto &v : kInt2Variants) {
    {
      const int64_t to_attr = static_cast<int64_t>(v.dtype);
      NodeProto node = MakeCastNode(to_attr);
      Tensor input = Tensor::FromFloat("", int2_shape, int2_fp32_values);
      Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
      Expect(node, {input}, {output}, std::string("test_cc_cast_FLOAT_to_") + v.name, {opset},
             "backend-test", registry);
    }
    Tensor packed_input;
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT);
      NodeProto node = MakeCastNode(to_attr);
      Tensor encoded = cast_kernel(Tensor::FromFloat("", int2_shape, int2_fp32_values),
                                   static_cast<int32_t>(v.dtype));
      packed_input = Tensor("", static_cast<int32_t>(v.dtype), int2_shape, encoded.data);
      Tensor output = cast_kernel(packed_input, static_cast<int32_t>(to_attr));
      Expect(node, {packed_input}, {output}, std::string("test_cc_cast_") + v.name + "_to_FLOAT",
             {opset}, "backend-test", registry);
    }
    {
      const int64_t to_attr = static_cast<int64_t>(v.wide_int_dtype);
      NodeProto node = MakeCastNode(to_attr);
      Tensor output = cast_kernel(packed_input, static_cast<int32_t>(to_attr));
      Expect(node, {packed_input}, {output},
             std::string("test_cc_cast_") + v.name + "_to_" + v.wide_int_name, {opset},
             "backend-test", registry);
    }
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
