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
// CastLike — element-wise conversion to the dtype carried by the second
// input ``target_type`` (since opset 15 in the ai.onnx domain). The first
// input ``input`` carries the values; the second input is consulted only
// for its element type. Functionally equivalent to ``Cast`` with
// ``to = target_type.data_type``.
//
// The cases below mirror the upstream ``test_castlike_<FROM>_to_<TO>`` node
// tests for every element type supported by the backend test ``Tensor``
// library and by :ref:`kernel::CastLike`. Inputs are short, fully
// deterministic vectors so the expected outputs can be computed directly
// by :ref:`kernel::CastLike`.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeCastLikeNode() {
  NodeProto node;
  node.set_op_type("CastLike");
  node.add_input("input");
  node.add_input("target_type");
  node.add_output("output");
  return node;
}

struct CastLikeDtype {
  DataType dtype;
  const char *name;
  std::function<Tensor()> make_input;
};

// Per-source-dtype deterministic inputs. Values mirror those used by the
// ``Cast`` backend test cases so the two operators are exercised over the
// same conversion matrix.
std::vector<CastLikeDtype> SupportedCastLikeDtypes() {
  return {
      {DataType::FLOAT, "FLOAT",
       []() { return Tensor::FromFloat("input", {4}, {-1.5f, 0.0f, 2.75f, 4.0f}); }},
      {DataType::DOUBLE, "DOUBLE",
       []() { return Tensor::FromDouble("input", {4}, {-1.5, 0.0, 2.75, 4.0}); }},
      {DataType::INT32, "INT32", []() { return Tensor::FromInt32("input", {4}, {-3, 0, 7, 42}); }},
      {DataType::INT64, "INT64", []() { return Tensor::FromInt64("input", {4}, {-3, 0, 7, 42}); }},
      {DataType::INT8, "INT8", []() { return Tensor::FromInt8("input", {4}, {-3, 0, 7, 42}); }},
      {DataType::UINT8, "UINT8", []() { return Tensor::FromUint8("input", {4}, {0, 1, 7, 42}); }},
      {DataType::INT16, "INT16", []() { return Tensor::FromInt16("input", {4}, {-3, 0, 7, 42}); }},
      {DataType::UINT16, "UINT16",
       []() { return Tensor::FromUint16("input", {4}, {0, 1, 7, 42}); }},
      {DataType::BOOL, "BOOL", []() { return Tensor::FromBool("input", {4}, {0, 1, 1, 0}); }},
      {DataType::STRING, "STRING",
       []() { return Tensor::FromStrings("input", {4}, {"-3", "0", "7", "42"}); }},
      // FLOAT16 inputs use a vector that is exactly representable in
      // IEEE-754 binary16 so cross-casts to integer / boolean dtypes do
      // not depend on the round-half-to-even path.
      {DataType::FLOAT16, "FLOAT16",
       []() { return MakeFloat16Tensor("input", {4}, {-1.5f, 0.0f, 2.75f, 4.0f}); }},
      // BFLOAT16 inputs use ``np.arange``-style integer-valued floats so
      // the round-to-nearest-even encoder lands on an exact value.
      {DataType::BFLOAT16, "BFLOAT16",
       []() { return MakeBfloat16Tensor("input", {4}, {-3.0f, 0.0f, 7.0f, 42.0f}); }},
  };
}

// Builds a placeholder ``target_type`` input with the requested dtype. Its
// contents are unused by :ref:`kernel::CastLike` (only the dtype is read),
// but the tensor must still carry one element so the backend test harness
// can serialize it as a regular initializer.
Tensor MakeTargetTypeTensor(const CastLikeDtype &to) {
  // A single-element tensor is enough for the dtype to be propagated. The
  // value is irrelevant; pick the smallest representative value of the
  // target dtype.
  switch (to.dtype) {
  case DataType::FLOAT:
    return Tensor::FromFloat("target_type", {1}, {0.0f});
  case DataType::DOUBLE:
    return Tensor::FromDouble("target_type", {1}, {0.0});
  case DataType::INT32:
    return Tensor::FromInt32("target_type", {1}, {0});
  case DataType::INT64:
    return Tensor::FromInt64("target_type", {1}, {0});
  case DataType::INT8:
    return Tensor::FromInt8("target_type", {1}, {0});
  case DataType::UINT8:
    return Tensor::FromUint8("target_type", {1}, {0});
  case DataType::INT16:
    return Tensor::FromInt16("target_type", {1}, {0});
  case DataType::UINT16:
    return Tensor::FromUint16("target_type", {1}, {0});
  case DataType::BOOL:
    return Tensor::FromBool("target_type", {1}, {0});
  case DataType::FLOAT16:
    return MakeFloat16Tensor("target_type", {1}, {0.0f});
  case DataType::BFLOAT16:
    return MakeBfloat16Tensor("target_type", {1}, {0.0f});
  case DataType::STRING:
    return Tensor::FromStrings("target_type", {1}, {""});
  default:
    return Tensor("target_type", static_cast<int32_t>(to.dtype), {1}, std::vector<uint8_t>(0));
  }
}

} // namespace

void RegisterCastLikeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(15);
  const OpsetId opset_v21 = DefaultOpset(21); // For FLOAT8, INT4, UINT4
  const OpsetId opset_v23 = DefaultOpset(23); // For FLOAT4E2M1
  const OpsetId opset_v25 = DefaultOpset(25); // For INT2, UINT2

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeCastLikeNode();
    Expect(registry, std::move(node), "test_cc_castlike_FLOAT_to_DOUBLE_benchmark", {opset},
           {kBenchmarkElementwiseSize, 1}, {kBenchmarkElementwiseSize}, []() -> IoData {
             const OpsetId opset = DefaultOpset(15);

             const KernelContext castlike_kernel_ctx{opset};
             const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

             Tensor input = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 2001);
             Tensor target_type = Tensor::FromDouble("target_type", {1}, {0.0});
             Tensor output = castlike_kernel(input, target_type);
             return IoData{{std::move(input), std::move(target_type)}, {std::move(output)}};
           });
    return;
  }

  const auto dtypes = SupportedCastLikeDtypes();
  for (const auto &from : dtypes) {
    for (const auto &to : dtypes) {
      if (from.dtype == to.dtype) {
        // Skip identity casts: upstream ONNX node tests skip them as well.
        continue;
      }
      Expect(registry, MakeCastLikeNode(),
             std::string("test_cc_castlike_") + from.name + "_to_" + to.name, {opset},
             [from, to]() -> IoData {
               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = from.make_input();
               Tensor target_type = MakeTargetTypeTensor(to);
               Tensor output = castlike_kernel(input, target_type);
               return IoData{{std::move(input), std::move(target_type)}, {std::move(output)}};
             });
    }
  }

  // ---------------------------------------------------------------------
  // FLOAT/FLOAT16 ↔ FLOAT8* / FLOAT4E2M1 cases.
  //
  // Mirrors the upstream ``test_castlike_<FROM>_to_FLOAT8*`` /
  // ``test_castlike_<FROM>_to_FLOAT4E2M1`` node tests (and their
  // reverses). The kernel forwards directly to :ref:`kernel::Cast`,
  // so the inputs follow the same vectors used by the corresponding
  // ``Cast`` backend test cases.
  // FLOAT8 types were introduced in opset 21, FLOAT4E2M1 in opset 23.
  // ---------------------------------------------------------------------
  const std::vector<int64_t> f8_shape = {3, 5};
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
  struct LowPrecisionVariant {
    DataType dtype;
    const char *name;
    std::vector<int64_t> shape;
    std::vector<float> values;
  };
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
  const std::vector<int64_t> e8m0_shape = {2, 4};
  const std::vector<float> e8m0_fp32_values = {0.0f, 0.124f, 0.25f, 0.5f, 1.1f, 2.0f, 4.0f, 8.0f};

  // FLOAT8 variants (opset 21)
  const std::vector<LowPrecisionVariant> kFloat8Variants = {
      {DataType::FLOAT8E4M3FN, "FLOAT8E4M3FN", f8_shape, f8_fp32_values},
      {DataType::FLOAT8E4M3FNUZ, "FLOAT8E4M3FNUZ", f8_shape, f8_fp32_values},
      {DataType::FLOAT8E5M2, "FLOAT8E5M2", f8_shape, f8_fp32_values},
      {DataType::FLOAT8E5M2FNUZ, "FLOAT8E5M2FNUZ", f8_shape, f8_fp32_values},
      {DataType::FLOAT8E8M0, "FLOAT8E8M0", e8m0_shape, e8m0_fp32_values},
  };

  for (const auto &v : kFloat8Variants) {
    // Build a 1-element target_type tensor of the destination dtype. Its
    // content is irrelevant; only its data_type is read by CastLike.
    Tensor low_target("target_type", static_cast<int32_t>(v.dtype), {1},
                      std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));
    Tensor float_target = Tensor::FromFloat("target_type", {1}, {0.0f});
    Tensor float16_target = MakeFloat16Tensor("target_type", {1}, {0.0f});

    // FLOAT -> v.dtype
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_FLOAT_to_") + v.name,
             {opset_v21}, [v]() -> IoData {
               Tensor low_target(
                   "target_type", static_cast<int32_t>(v.dtype), {1},
                   std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = Tensor::FromFloat("input", v.shape, v.values);
               Tensor output = castlike_kernel(input, low_target);
               return IoData{{std::move(input), std::move(low_target)}, {std::move(output)}};
             });
    }
    // v.dtype -> FLOAT (input is the packed/encoded form of the FP32 vector).
    const auto make_packed_input = [opset, v]() {
      const KernelContext cast_kernel_ctx{opset};
      const onnx_kernels::kernel::Cast cast_kernel{cast_kernel_ctx};

      Tensor encoded =
          cast_kernel(Tensor::FromFloat("input", v.shape, v.values), static_cast<int32_t>(v.dtype));
      return Tensor("input", static_cast<int32_t>(v.dtype), v.shape, std::move(encoded.data));
    };
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_") + v.name + "_to_FLOAT",
             {opset_v21}, [make_packed_input]() -> IoData {
               Tensor float_target = Tensor::FromFloat("target_type", {1}, {0.0f});

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, float_target);
               return IoData{{std::move(packed_input), std::move(float_target)},
                             {std::move(output)}};
             });
    }
    // FLOAT16 -> v.dtype
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_FLOAT16_to_") + v.name,
             {opset_v21}, [v]() -> IoData {
               Tensor low_target(
                   "target_type", static_cast<int32_t>(v.dtype), {1},
                   std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = MakeFloat16Tensor("input", v.shape, v.values);
               Tensor output = castlike_kernel(input, low_target);
               return IoData{{std::move(input), std::move(low_target)}, {std::move(output)}};
             });
    }
    // v.dtype -> FLOAT16
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_") + v.name + "_to_FLOAT16",
             {opset_v21}, [make_packed_input]() -> IoData {
               Tensor float16_target = MakeFloat16Tensor("target_type", {1}, {0.0f});

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, float16_target);
               return IoData{{std::move(packed_input), std::move(float16_target)},
                             {std::move(output)}};
             });
    }
  }

  // FLOAT4E2M1 variant (opset 23)
  {
    const LowPrecisionVariant v = {DataType::FLOAT4E2M1, "FLOAT4E2M1", {3, 5}, f4_fp32_values};
    Tensor low_target("target_type", static_cast<int32_t>(v.dtype), {1},
                      std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));
    Tensor float_target = Tensor::FromFloat("target_type", {1}, {0.0f});
    Tensor float16_target = MakeFloat16Tensor("target_type", {1}, {0.0f});

    // FLOAT -> FLOAT4E2M1
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_FLOAT_to_") + v.name,
             {opset_v23}, [v]() -> IoData {
               Tensor low_target(
                   "target_type", static_cast<int32_t>(v.dtype), {1},
                   std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = Tensor::FromFloat("input", v.shape, v.values);
               Tensor output = castlike_kernel(input, low_target);
               return IoData{{std::move(input), std::move(low_target)}, {std::move(output)}};
             });
    }
    // FLOAT4E2M1 -> FLOAT
    const auto make_packed_input = [opset, v]() {
      const KernelContext cast_kernel_ctx{opset};
      const onnx_kernels::kernel::Cast cast_kernel{cast_kernel_ctx};

      Tensor encoded =
          cast_kernel(Tensor::FromFloat("input", v.shape, v.values), static_cast<int32_t>(v.dtype));
      return Tensor("input", static_cast<int32_t>(v.dtype), v.shape, std::move(encoded.data));
    };
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_") + v.name + "_to_FLOAT",
             {opset_v23}, [make_packed_input]() -> IoData {
               Tensor float_target = Tensor::FromFloat("target_type", {1}, {0.0f});

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, float_target);
               return IoData{{std::move(packed_input), std::move(float_target)},
                             {std::move(output)}};
             });
    }
    // FLOAT16 -> FLOAT4E2M1
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_FLOAT16_to_") + v.name,
             {opset_v23}, [v]() -> IoData {
               Tensor low_target(
                   "target_type", static_cast<int32_t>(v.dtype), {1},
                   std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = MakeFloat16Tensor("input", v.shape, v.values);
               Tensor output = castlike_kernel(input, low_target);
               return IoData{{std::move(input), std::move(low_target)}, {std::move(output)}};
             });
    }
    // FLOAT4E2M1 -> FLOAT16
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_") + v.name + "_to_FLOAT16",
             {opset_v23}, [make_packed_input]() -> IoData {
               Tensor float16_target = MakeFloat16Tensor("target_type", {1}, {0.0f});

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, float16_target);
               return IoData{{std::move(packed_input), std::move(float16_target)},
                             {std::move(output)}};
             });
    }
  }

  // ---------------------------------------------------------------------
  // Sub-byte (INT4 / UINT4 / INT2 / UINT2) cases.
  //
  // Mirrors the upstream ``test_castlike_<FROM>_to_<TO>`` node tests that
  // exercise the 4-bit and 2-bit packed integer dtypes. CastLike forwards
  // to :ref:`kernel::Cast`, so the inputs follow the same vectors used by
  // the corresponding ``Cast`` backend test cases: ``FLOAT`` ↔ packed,
  // ``FLOAT16`` ↔ packed and packed ↔ companion whole-byte integer
  // (``INT8`` / ``UINT8``) pairs.
  // ---------------------------------------------------------------------
  struct SubByteVariant {
    DataType dtype;
    const char *name;
    DataType wide_int_dtype; // INT8 (signed) or UINT8 (unsigned)
    const char *wide_int_name;
    std::function<Tensor()> make_wide_target;
  };
  const std::vector<SubByteVariant> kInt4Variants = {
      {DataType::UINT4, "UINT4", DataType::UINT8, "UINT8",
       []() { return Tensor::FromUint8("target_type", {1}, {0}); }},
      {DataType::INT4, "INT4", DataType::INT8, "INT8",
       []() { return Tensor::FromInt8("target_type", {1}, {0}); }},
  };
  const std::vector<SubByteVariant> kInt2Variants = {
      {DataType::UINT2, "UINT2", DataType::UINT8, "UINT8",
       []() { return Tensor::FromUint8("target_type", {1}, {0}); }},
      {DataType::INT2, "INT2", DataType::INT8, "INT8",
       []() { return Tensor::FromInt8("target_type", {1}, {0}); }},
  };

  // INT4 / UINT4 — input shape (5, 5) with the 25-element ``np.arange(-9, 16)``
  // sweep used by the upstream generator. INT4/UINT4 were introduced in opset 21.
  const std::vector<int64_t> int4_shape = {5, 5};
  std::vector<float> int4_fp32_values(25);
  for (int i = 0; i < 25; ++i) {
    int4_fp32_values[static_cast<size_t>(i)] = static_cast<float>(i - 9);
  }
  // INT2 / UINT2 — input shape (7, 1) with the 7-element ``np.arange(-3, 4)``
  // sweep used by the upstream generator. INT2/UINT2 were introduced in opset 25.
  const std::vector<int64_t> int2_shape = {7, 1};
  std::vector<float> int2_fp32_values(7);
  for (int i = 0; i < 7; ++i) {
    int2_fp32_values[static_cast<size_t>(i)] = static_cast<float>(i - 3);
  }

  const Tensor float_target = Tensor::FromFloat("target_type", {1}, {0.0f});
  const Tensor float16_target = MakeFloat16Tensor("target_type", {1}, {0.0f});

  // INT4/UINT4 cases (opset 21)
  for (const auto &v : kInt4Variants) {
    Tensor sub_target("target_type", static_cast<int32_t>(v.dtype), {1},
                      std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));
    Tensor wide_target = v.make_wide_target();
    // FLOAT -> sub-byte
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_FLOAT_to_") + v.name,
             {opset_v21}, [int4_shape, int4_fp32_values, v]() -> IoData {
               Tensor sub_target(
                   "target_type", static_cast<int32_t>(v.dtype), {1},
                   std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = Tensor::FromFloat("input", int4_shape, int4_fp32_values);
               Tensor output = castlike_kernel(input, sub_target);
               return IoData{{std::move(input), std::move(sub_target)}, {std::move(output)}};
             });
    }
    // sub-byte -> FLOAT (input is the packed encoding of the FP32 vector).
    const auto make_packed_input = [opset, int4_shape, int4_fp32_values, v]() {
      const KernelContext cast_kernel_ctx{opset};
      const onnx_kernels::kernel::Cast cast_kernel{cast_kernel_ctx};

      Tensor encoded = cast_kernel(Tensor::FromFloat("input", int4_shape, int4_fp32_values),
                                   static_cast<int32_t>(v.dtype));
      return Tensor("input", static_cast<int32_t>(v.dtype), int4_shape, std::move(encoded.data));
    };
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_") + v.name + "_to_FLOAT",
             {opset_v21}, [make_packed_input]() -> IoData {
               const Tensor float_target = Tensor::FromFloat("target_type", {1}, {0.0f});

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, float_target);
               return IoData{{std::move(packed_input), std::move(float_target)},
                             {std::move(output)}};
             });
    }
    // sub-byte -> companion whole-byte integer (INT4->INT8, UINT4->UINT8).
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node),
             std::string("test_cc_castlike_") + v.name + "_to_" + v.wide_int_name, {opset_v21},
             [make_packed_input, v]() -> IoData {
               Tensor wide_target = v.make_wide_target();

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, wide_target);
               return IoData{{std::move(packed_input), std::move(wide_target)},
                             {std::move(output)}};
             });
    }
    // FLOAT16 -> sub-byte
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_FLOAT16_to_") + v.name,
             {opset_v21}, [int4_shape, int4_fp32_values, v]() -> IoData {
               Tensor sub_target(
                   "target_type", static_cast<int32_t>(v.dtype), {1},
                   std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = MakeFloat16Tensor("input", int4_shape, int4_fp32_values);
               Tensor output = castlike_kernel(input, sub_target);
               return IoData{{std::move(input), std::move(sub_target)}, {std::move(output)}};
             });
    }
    // sub-byte -> FLOAT16
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_") + v.name + "_to_FLOAT16",
             {opset_v21}, [make_packed_input]() -> IoData {
               const Tensor float16_target = MakeFloat16Tensor("target_type", {1}, {0.0f});

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, float16_target);
               return IoData{{std::move(packed_input), std::move(float16_target)},
                             {std::move(output)}};
             });
    }
  }

  // INT2/UINT2 cases (opset 25)
  for (const auto &v : kInt2Variants) {
    Tensor sub_target("target_type", static_cast<int32_t>(v.dtype), {1},
                      std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));
    Tensor wide_target = v.make_wide_target();
    // FLOAT -> sub-byte
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_FLOAT_to_") + v.name,
             {opset_v25}, [int2_shape, int2_fp32_values, v]() -> IoData {
               Tensor sub_target(
                   "target_type", static_cast<int32_t>(v.dtype), {1},
                   std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = Tensor::FromFloat("input", int2_shape, int2_fp32_values);
               Tensor output = castlike_kernel(input, sub_target);
               return IoData{{std::move(input), std::move(sub_target)}, {std::move(output)}};
             });
    }
    // sub-byte -> FLOAT (input is the packed encoding of the FP32 vector).
    const auto make_packed_input = [opset, int2_shape, int2_fp32_values, v]() {
      const KernelContext cast_kernel_ctx{opset};
      const onnx_kernels::kernel::Cast cast_kernel{cast_kernel_ctx};

      Tensor encoded = cast_kernel(Tensor::FromFloat("input", int2_shape, int2_fp32_values),
                                   static_cast<int32_t>(v.dtype));
      return Tensor("input", static_cast<int32_t>(v.dtype), int2_shape, std::move(encoded.data));
    };
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_") + v.name + "_to_FLOAT",
             {opset_v25}, [make_packed_input]() -> IoData {
               const Tensor float_target = Tensor::FromFloat("target_type", {1}, {0.0f});

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, float_target);
               return IoData{{std::move(packed_input), std::move(float_target)},
                             {std::move(output)}};
             });
    }
    // sub-byte -> companion whole-byte integer (INT2->INT8, UINT2->UINT8).
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node),
             std::string("test_cc_castlike_") + v.name + "_to_" + v.wide_int_name, {opset_v25},
             [make_packed_input, v]() -> IoData {
               Tensor wide_target = v.make_wide_target();

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, wide_target);
               return IoData{{std::move(packed_input), std::move(wide_target)},
                             {std::move(output)}};
             });
    }
    // FLOAT16 -> sub-byte
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_FLOAT16_to_") + v.name,
             {opset_v25}, [int2_shape, int2_fp32_values, v]() -> IoData {
               Tensor sub_target(
                   "target_type", static_cast<int32_t>(v.dtype), {1},
                   std::vector<uint8_t>(PackedByteSize(static_cast<int32_t>(v.dtype), 1)));

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor input = MakeFloat16Tensor("input", int2_shape, int2_fp32_values);
               Tensor output = castlike_kernel(input, sub_target);
               return IoData{{std::move(input), std::move(sub_target)}, {std::move(output)}};
             });
    }
    // sub-byte -> FLOAT16
    {
      NodeProto node = MakeCastLikeNode();
      Expect(registry, std::move(node), std::string("test_cc_castlike_") + v.name + "_to_FLOAT16",
             {opset_v25}, [make_packed_input]() -> IoData {
               const Tensor float16_target = MakeFloat16Tensor("target_type", {1}, {0.0f});

               const OpsetId opset = DefaultOpset(15);

               const KernelContext castlike_kernel_ctx{opset};
               const onnx_kernels::kernel::CastLike castlike_kernel{castlike_kernel_ctx};

               Tensor packed_input = make_packed_input();
               Tensor output = castlike_kernel(packed_input, float16_target);
               return IoData{{std::move(packed_input), std::move(float16_target)},
                             {std::move(output)}};
             });
    }
  }

  // ---------------------------------------------------------------------
  // CastLike no_saturate cases — FLOAT/FLOAT16 → FLOAT8* with saturate=0.
  //
  // Mirrors the upstream ``test_castlike_no_saturate_<FROM>_to_<TO>`` tests.
  // When saturate is 0, out-of-range values produce NaN instead of being
  // clamped to the maximum representable magnitude.
  // FLOAT8 types were introduced in opset 21.
  // ---------------------------------------------------------------------
  {
    struct Float8Only {
      DataType dtype;
      const char *name;
    };
    const Float8Only kFloat8Variants[] = {
        {DataType::FLOAT8E4M3FN, "FLOAT8E4M3FN"},
        {DataType::FLOAT8E4M3FNUZ, "FLOAT8E4M3FNUZ"},
        {DataType::FLOAT8E5M2, "FLOAT8E5M2"},
        {DataType::FLOAT8E5M2FNUZ, "FLOAT8E5M2FNUZ"},
    };

    for (const auto &v : kFloat8Variants) {
      Tensor low_target("target_type", static_cast<int32_t>(v.dtype), {1},
                        std::vector<uint8_t>(1, 0u));
      // FLOAT -> FLOAT8* (no_saturate)
      {
        NodeProto node;
        node.set_op_type("CastLike");
        node.add_input("input");
        node.add_input("target_type");
        node.add_output("output");
        AddAttribute<int64_t>(node, "saturate", 0);
        Expect(registry, std::move(node),
               std::string("test_castlike_no_saturate_FLOAT_to_") + v.name, {opset_v21},
               [f8_shape, f8_fp32_values, v]() -> IoData {
                 Tensor low_target("target_type", static_cast<int32_t>(v.dtype), {1},
                                   std::vector<uint8_t>(1, 0u));

                 const OpsetId opset = DefaultOpset(15);

                 const KernelContext cast_k_ctx{opset};
                 const onnx_kernels::kernel::Cast cast_k{cast_k_ctx};

                 Tensor input = Tensor::FromFloat("input", f8_shape, f8_fp32_values);
                 Tensor output = cast_k(input, static_cast<int32_t>(v.dtype), /*saturate=*/false);
                 return IoData{{std::move(input), std::move(low_target)}, {std::move(output)}};
               });
      }
      // FLOAT16 -> FLOAT8* (no_saturate)
      {
        NodeProto node;
        node.set_op_type("CastLike");
        node.add_input("input");
        node.add_input("target_type");
        node.add_output("output");
        AddAttribute<int64_t>(node, "saturate", 0);
        Expect(registry, std::move(node),
               std::string("test_castlike_no_saturate_FLOAT16_to_") + v.name, {opset_v21},
               [f8_shape, f8_fp32_values, v]() -> IoData {
                 Tensor low_target("target_type", static_cast<int32_t>(v.dtype), {1},
                                   std::vector<uint8_t>(1, 0u));

                 const OpsetId opset = DefaultOpset(15);

                 const KernelContext cast_k_ctx{opset};
                 const onnx_kernels::kernel::Cast cast_k{cast_k_ctx};

                 Tensor input = MakeFloat16Tensor("input", f8_shape, f8_fp32_values);
                 Tensor output = cast_k(input, static_cast<int32_t>(v.dtype), /*saturate=*/false);
                 return IoData{{std::move(input), std::move(low_target)}, {std::move(output)}};
               });
      }
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
