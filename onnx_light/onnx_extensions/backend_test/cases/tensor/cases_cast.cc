// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// Cast — element-wise conversion to the dtype carried by the required
// integer attribute ``to`` (since opset 13 in the ai.onnx domain).
//
// The cases below mirror the upstream ``test_cast_<FROM>_to_<TO>`` node
// tests for every element type supported by the backend test ``Tensor``
// library and by :ref:`kernel::Cast`: the numeric types ``FLOAT``,
// ``DOUBLE``, ``INT32``, ``INT64``, ``INT8``, ``UINT8``, ``INT16``,
// ``UINT16``, ``BOOL``, ``FLOAT16``, ``BFLOAT16`` and the variable-length
// ``STRING`` type. All cross-dtype permutations are registered so that
// ``Cast`` coverage is complete with respect to what the reference kernel
// accepts. Inputs are small, fully deterministic vectors so the expected
// outputs can be computed directly by :ref:`kernel::Cast`.
//
// In addition to the numeric+STRING grid, the float8 variants
// ``FLOAT8E4M3FN``, ``FLOAT8E4M3FNUZ``, ``FLOAT8E5M2`` and
// ``FLOAT8E5M2FNUZ`` are registered as ``FLOAT`` ↔ ``FLOAT8*`` and
// ``FLOAT16`` ↔ ``FLOAT8*`` pairs (mirroring the corresponding upstream
// ``test_cast`` cases). The sub-byte packed integer variants ``INT4`` /
// ``UINT4`` / ``INT2`` / ``UINT2`` are registered as ``FLOAT`` ↔ packed,
// ``FLOAT16`` ↔ packed and packed ↔ companion whole-byte integer
// (``INT8`` / ``UINT8``) pairs, again mirroring the upstream
// ``FLOAT4E2M1`` is exercised in a dedicated loop below (it only
// round-trips against ``FLOAT`` and ``FLOAT16`` / ``BFLOAT16``).
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

NodeProto MakeCastNodeNoSaturate(int64_t to) {
  NodeProto node;
  node.set_op_type("Cast");
  node.add_input("input");
  node.add_output("output");
  AddAttribute<int64_t>(node, "to", to);
  AddAttribute<int64_t>(node, "saturate", 0);
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
      // FLOAT16 inputs use a vector that is exactly representable in
      // IEEE-754 binary16 so cross-casts to integer / boolean dtypes do
      // not depend on the round-half-to-even path.
      {DataType::FLOAT16, "FLOAT16",
       []() { return MakeFloat16Tensor("", {4}, {-1.5f, 0.0f, 2.75f, 4.0f}); }},
      // BFLOAT16 inputs use ``np.arange``-style integer-valued floats so
      // the round-to-nearest-even encoder lands on an exact value.
      {DataType::BFLOAT16, "BFLOAT16",
       []() { return MakeBfloat16Tensor("", {4}, {-3.0f, 0.0f, 7.0f, 42.0f}); }},
  };
}

} // namespace

void RegisterCastCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(19);
  const OpsetId opset_v21 = DefaultOpset(21); // For FLOAT8, INT4, UINT4
  const OpsetId opset_v23 = DefaultOpset(23); // For FLOAT4E2M1
  const OpsetId opset_v25 = DefaultOpset(25); // For INT2, UINT2
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Cast cast_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    const int64_t to_attr = static_cast<int64_t>(DataType::DOUBLE);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_FLOAT_to_DOUBLE_benchmark", {opset},
           {kBenchmarkElementwiseSize}, {kBenchmarkElementwiseSize}, [cast_kernel]() -> IoData {
             Tensor input = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 2001);
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
    return;
  }

  const auto dtypes = SupportedCastDtypes();
  for (const auto &from : dtypes) {
    for (const auto &to : dtypes) {
      if (from.dtype == to.dtype) {
        // Skip identity casts: upstream ONNX node tests skip them as well.
        continue;
      }
      const int64_t to_attr = static_cast<int64_t>(to.dtype);
      Expect(registry, MakeCastNode(to_attr),
             std::string("test_cc_cast_") + from.name + "_to_" + to.name, {opset}, [=]() -> IoData {
               Tensor input = from.make_input();
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
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
  // FLOAT8 types were introduced in opset 21.
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
      Expect(registry, std::move(node), std::string("test_cc_cast_FLOAT_to_") + v.name, {opset_v21},
             [=]() -> IoData {
               Tensor input = Tensor::FromFloat("", f8_shape, f8_fp32_values);
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // FLOAT8* -> FLOAT — input bytes are the saturated FLOAT8 encoding
    // of the same FP32 vector (matches the upstream behaviour where
    // ``np_from = saturate_cast(np_fp32, from_np_dtype)`` is fed into
    // the node).
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_") + v.name + "_to_FLOAT",
             {opset_v21}, [=]() -> IoData {
               Tensor encoded = cast_kernel(Tensor::FromFloat("", f8_shape, f8_fp32_values),
                                            static_cast<int32_t>(v.dtype));
               Tensor input("", static_cast<int32_t>(v.dtype), f8_shape, encoded.data);
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // FLOAT16 -> FLOAT8*
    {
      const int64_t to_attr = static_cast<int64_t>(v.dtype);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_FLOAT16_to_") + v.name,
             {opset_v21}, [=]() -> IoData {
               Tensor input = MakeFloat16Tensor("", f8_shape, f8_fp32_values);
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // FLOAT8* -> FLOAT16 — input bytes are the saturated FLOAT8 encoding
    // of the same FP32 vector (matching the upstream behaviour).
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT16);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_") + v.name + "_to_FLOAT16",
             {opset_v21}, [=]() -> IoData {
               Tensor encoded = cast_kernel(Tensor::FromFloat("", f8_shape, f8_fp32_values),
                                            static_cast<int32_t>(v.dtype));
               Tensor input("", static_cast<int32_t>(v.dtype), f8_shape, encoded.data);
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
  }

  // ---------------------------------------------------------------------
  // Cast no_saturate cases — FLOAT/FLOAT16 → FLOAT8* with saturate=0.
  //
  // Mirrors the upstream ``test_cast_no_saturate_<FROM>_to_<TO>`` tests.
  // When saturate is 0, out-of-range values produce NaN instead of being
  // clamped to the maximum representable magnitude.
  // FLOAT8 types were introduced in opset 21.
  // ---------------------------------------------------------------------
  {
    for (const auto &v : kFloat8Variants) {
      // FLOAT -> FLOAT8* (no_saturate)
      {
        const int64_t to_attr = static_cast<int64_t>(v.dtype);
        NodeProto node = MakeCastNodeNoSaturate(to_attr);
        Expect(registry, std::move(node), std::string("test_cast_no_saturate_FLOAT_to_") + v.name,
               {opset_v21}, [=]() -> IoData {
                 Tensor input = Tensor::FromFloat("", f8_shape, f8_fp32_values);
                 Tensor output =
                     cast_kernel(input, static_cast<int32_t>(to_attr), /*saturate=*/false);
                 return IoData{{std::move(input)}, {std::move(output)}};
               });
      }
      // FLOAT16 -> FLOAT8* (no_saturate)
      {
        const int64_t to_attr = static_cast<int64_t>(v.dtype);
        NodeProto node = MakeCastNodeNoSaturate(to_attr);
        Expect(registry, std::move(node), std::string("test_cast_no_saturate_FLOAT16_to_") + v.name,
               {opset_v21}, [=]() -> IoData {
                 Tensor input = MakeFloat16Tensor("", f8_shape, f8_fp32_values);
                 Tensor output =
                     cast_kernel(input, static_cast<int32_t>(to_attr), /*saturate=*/false);
                 return IoData{{std::move(input)}, {std::move(output)}};
               });
      }
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
  // wrapping-cast paths (values outside the destination range) and
  // the typical in-range values are exercised.
  // INT4/UINT4 were introduced in opset 21, INT2/UINT2 in opset 25.
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
      Expect(registry, std::move(node), std::string("test_cc_cast_FLOAT_to_") + v.name, {opset_v21},
             [=]() -> IoData {
               Tensor input = Tensor::FromFloat("", int4_shape, int4_fp32_values);
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // sub-byte -> FLOAT — input bytes are the wrapped/packed encoding of
    // the same FP32 vector (matching upstream where the input is the result
    // of ``np_fp32.astype(sub_byte_dtype)``).
    const Tensor packed_input = [&]() {
      Tensor encoded = cast_kernel(Tensor::FromFloat("", int4_shape, int4_fp32_values),
                                   static_cast<int32_t>(v.dtype));
      return Tensor("", static_cast<int32_t>(v.dtype), int4_shape, encoded.data);
    }();
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_") + v.name + "_to_FLOAT",
             {opset_v21}, [=]() -> IoData {
               Tensor input = packed_input;
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // sub-byte -> companion whole-byte integer (INT4->INT8, UINT4->UINT8).
    {
      const int64_t to_attr = static_cast<int64_t>(v.wide_int_dtype);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node),
             std::string("test_cc_cast_") + v.name + "_to_" + v.wide_int_name, {opset_v21},
             [=]() -> IoData {
               Tensor input = packed_input;
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // FLOAT16 -> sub-byte
    {
      const int64_t to_attr = static_cast<int64_t>(v.dtype);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_FLOAT16_to_") + v.name,
             {opset_v21}, [=]() -> IoData {
               Tensor input = MakeFloat16Tensor("", int4_shape, int4_fp32_values);
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // sub-byte -> FLOAT16
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT16);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_") + v.name + "_to_FLOAT16",
             {opset_v21}, [=]() -> IoData {
               Tensor input = packed_input;
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
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
      Expect(registry, std::move(node), std::string("test_cc_cast_FLOAT_to_") + v.name, {opset_v25},
             [=]() -> IoData {
               Tensor input = Tensor::FromFloat("", int2_shape, int2_fp32_values);
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    const Tensor packed_input = [&]() {
      Tensor encoded = cast_kernel(Tensor::FromFloat("", int2_shape, int2_fp32_values),
                                   static_cast<int32_t>(v.dtype));
      return Tensor("", static_cast<int32_t>(v.dtype), int2_shape, encoded.data);
    }();
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_") + v.name + "_to_FLOAT",
             {opset_v25}, [=]() -> IoData {
               Tensor input = packed_input;
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    {
      const int64_t to_attr = static_cast<int64_t>(v.wide_int_dtype);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node),
             std::string("test_cc_cast_") + v.name + "_to_" + v.wide_int_name, {opset_v25},
             [=]() -> IoData {
               Tensor input = packed_input;
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // FLOAT16 -> INT2 / UINT2
    {
      const int64_t to_attr = static_cast<int64_t>(v.dtype);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_FLOAT16_to_") + v.name,
             {opset_v25}, [=]() -> IoData {
               Tensor input = MakeFloat16Tensor("", int2_shape, int2_fp32_values);
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    // INT2 / UINT2 -> FLOAT16
    {
      const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT16);
      NodeProto node = MakeCastNode(to_attr);
      Expect(registry, std::move(node), std::string("test_cc_cast_") + v.name + "_to_FLOAT16",
             {opset_v25}, [=]() -> IoData {
               Tensor input = packed_input;
               Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
  }

  // ---------------------------------------------------------------------
  // FLOAT4E2M1 cases — FLOAT/FLOAT16 ↔ FLOAT4E2M1.
  //
  // Mirrors the upstream ``test_cast_<FROM>_to_FLOAT4E2M1`` /
  // ``test_cast_FLOAT4E2M1_to_<TO>`` node tests. The 15-element FP32
  // vector covers the saturating-cast paths (values above the
  // representable range +/-6), the +/-0 / NaN / +/-infinity special
  // values, and a few in-range values such as the asymmetric
  // ``+/-0.5`` / ``+/-1.5`` representable points.
  // FLOAT4E2M1 was introduced in opset 23.
  // ---------------------------------------------------------------------
  const std::vector<int64_t> f4_shape = {3, 5};
  const std::vector<float> f4_fp32_values = {
      0.48f,
      0.25f,
      1.05f,
      -3.5f,
      -8.0f,
      9.0f,
      1000000.0f,
      1e-7f,
      std::nanf(""),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -4.0f,
      0.01f,
      -0.0f,
  };
  {
    // FLOAT -> FLOAT4E2M1
    const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT4E2M1);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_FLOAT_to_FLOAT4E2M1", {opset_v23},
           [=]() -> IoData {
             Tensor input = Tensor::FromFloat("", f4_shape, f4_fp32_values);
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
  const Tensor f4_packed_input = [&]() {
    Tensor encoded = cast_kernel(Tensor::FromFloat("", f4_shape, f4_fp32_values),
                                 static_cast<int32_t>(DataType::FLOAT4E2M1));
    return Tensor("", static_cast<int32_t>(DataType::FLOAT4E2M1), f4_shape, encoded.data);
  }();
  {
    // FLOAT4E2M1 -> FLOAT — input bytes are the saturated FLOAT4E2M1
    // encoding of the same FP32 vector.
    const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_FLOAT4E2M1_to_FLOAT", {opset_v23},
           [=]() -> IoData {
             Tensor input = f4_packed_input;
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
  {
    // FLOAT16 -> FLOAT4E2M1
    const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT4E2M1);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_FLOAT16_to_FLOAT4E2M1", {opset_v23},
           [=]() -> IoData {
             Tensor input = MakeFloat16Tensor("", f4_shape, f4_fp32_values);
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
  {
    // FLOAT4E2M1 -> FLOAT16
    const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT16);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_FLOAT4E2M1_to_FLOAT16", {opset_v23},
           [=]() -> IoData {
             Tensor input = f4_packed_input;
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }

  // ---------------------------------------------------------------------
  // FLOAT8E8M0 cases — FLOAT/FLOAT16 ↔ FLOAT8E8M0.
  //
  // Mirrors the upstream ``test_cast_e8m0_<FROM>_to_FLOAT8E8M0`` /
  // ``test_cast_e8m0_FLOAT8E8M0_to_<TO>`` node tests. FLOAT8E8M0 stores
  // a biased exponent only (no sign / mantissa); the default
  // ``round_mode="up"`` and ``saturate=1`` semantics are the only ones
  // implemented by :ref:`kernel::Cast`, which matches the attributes
  // used by the upstream cases.
  // FLOAT8E8M0 was introduced in opset 21.
  // ---------------------------------------------------------------------
  const std::vector<int64_t> e8m0_shape = {2, 4};
  const std::vector<float> e8m0_fp32_values = {0.0f, 0.124f, 0.25f, 0.5f, 1.1f, 2.0f, 4.0f, 8.0f};
  {
    // FLOAT -> FLOAT8E8M0
    const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT8E8M0);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_e8m0_FLOAT_to_FLOAT8E8M0", {opset_v21},
           [=]() -> IoData {
             Tensor input = Tensor::FromFloat("", e8m0_shape, e8m0_fp32_values);
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
  const Tensor e8m0_packed_input = [&]() {
    Tensor encoded = cast_kernel(Tensor::FromFloat("", e8m0_shape, e8m0_fp32_values),
                                 static_cast<int32_t>(DataType::FLOAT8E8M0));
    return Tensor("", static_cast<int32_t>(DataType::FLOAT8E8M0), e8m0_shape, encoded.data);
  }();
  {
    // FLOAT8E8M0 -> FLOAT — input bytes are the FLOAT8E8M0 encoding of
    // the same FP32 vector.
    const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_e8m0_FLOAT8E8M0_to_FLOAT", {opset_v21},
           [=]() -> IoData {
             Tensor input = e8m0_packed_input;
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
  {
    // FLOAT16 -> FLOAT8E8M0
    const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT8E8M0);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_e8m0_FLOAT16_to_FLOAT8E8M0", {opset_v21},
           [=]() -> IoData {
             Tensor input = MakeFloat16Tensor("", e8m0_shape, e8m0_fp32_values);
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
  {
    // FLOAT8E8M0 -> FLOAT16
    const int64_t to_attr = static_cast<int64_t>(DataType::FLOAT16);
    NodeProto node = MakeCastNode(to_attr);
    Expect(registry, std::move(node), "test_cc_cast_e8m0_FLOAT8E8M0_to_FLOAT16", {opset_v21},
           [=]() -> IoData {
             Tensor input = e8m0_packed_input;
             Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
