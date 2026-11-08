// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/gguf.h"
#include "onnx_core/runtime/quantization.h"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

void PutHalf(std::string &bytes, size_t offset, uint16_t bits) {
  bytes[offset] = static_cast<char>(bits & 255);
  bytes[offset + 1] = static_cast<char>(bits >> 8);
}

std::vector<float> Values(const EncodedValueProto &payload) {
  const auto tensor = DequantizeGgufTensorPayload(payload);
  EXPECT_EQ(tensor.data_type, TensorProto::FLOAT);
  EXPECT_EQ(tensor.shape.size(), payload.logical_type().tensor_type().shape().dim().size());
  std::vector<float> values(tensor.size_bytes() / sizeof(float));
  std::memcpy(values.data(), tensor.bytes(), tensor.size_bytes());
  return values;
}

} // namespace

TEST(GgufTensorPayload, CanonicalLayoutsAndExactBytes) {
  const std::array<std::string, 5> names{"Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"};
  const std::array<size_t, 5> sizes{18, 34, 144, 176, 210};
  ASSERT_EQ(GgufBlockLayouts().size(), names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    const auto &layout = GgufLayout(names[i]);
    EXPECT_EQ(layout.type_size, sizes[i]);
    EXPECT_EQ(layout.block_size, i < 2 ? size_t{32} : size_t{256});
    std::string bytes(layout.type_size * 2, '\xff');
    for (size_t block = 0; block < 2; ++block) {
      const size_t offset = block * layout.type_size;
      if (names[i] == "Q6_K")
        PutHalf(bytes, offset + 208, 0x8000);
      else {
        PutHalf(bytes, offset, 0x8000);
        if (i >= 2)
          PutHalf(bytes, offset + 2, 0);
      }
    }
    auto payload =
        ImportGgufTensorPayload(bytes, {2, static_cast<int64_t>(layout.block_size)}, names[i]);
    EXPECT_EQ(payload.struct_type().name().value(), std::string(kGgufTypePrefix) + names[i]);
    EXPECT_EQ(ExportGgufTensorPayload(payload), bytes);
    bytes[0] = static_cast<char>(static_cast<uint8_t>(bytes[0]) ^ 1);
    EXPECT_NE(payload.raw_data(), bytes);
    EXPECT_EQ(Values(payload).size(), layout.block_size * 2);
  }
  for (const auto *name : {"q4_0", "q4_k", "Q4", "Q4_K_M", "Q4_K_S", "Q2_K", "Q3_K", "IQ4_NL",
                           "F32", "", " Q4_0", "Q8_0 "})
    EXPECT_THROW(GgufLayout(name), std::invalid_argument) << name;
}

TEST(GgufTensorPayload, Q4AndQ8SignedCodesAndNibbleOrdering) {
  std::string q4(18, '\0');
  PutHalf(q4, 0, 0xbc00); // -1 in IEEE binary16.
  for (size_t i = 0; i < 16; ++i)
    q4[2 + i] = static_cast<char>(i | ((15 - i) << 4));
  const auto four = ImportGgufTensorPayload(q4, {32}, "Q4_0");
  const auto values = Values(four);
  ASSERT_EQ(values.size(), size_t{32});
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(values[i], 8.0f - static_cast<float>(i));
    EXPECT_FLOAT_EQ(values[i + 16], static_cast<float>(i) - 7.0f);
  }
  EXPECT_EQ(ReadGgufBlockParameters(four, 0).scales, std::vector<float>{-1});
  EXPECT_TRUE(ReadGgufBlockParameters(four, 0).minima.empty());

  std::string q8(34, '\0');
  PutHalf(q8, 0, 0x3800); // 0.5.
  for (size_t i = 0; i < 32; ++i)
    q8[2 + i] = static_cast<char>(i % 2 == 0 ? 128 + i : i);
  const auto eight = ImportGgufTensorPayload(q8, {32}, "Q8_0");
  const auto decoded = Values(eight);
  ASSERT_EQ(decoded.size(), size_t{32});
  for (size_t i = 0; i < 32; ++i)
    EXPECT_FLOAT_EQ(decoded[i], 0.5f * static_cast<float>(i % 2 == 0 ? static_cast<int>(i) - 128
                                                                     : static_cast<int>(i)));
  EXPECT_EQ(ReadGgufBlockParameters(eight, 0).scales, std::vector<float>{0.5f});
}

TEST(GgufTensorPayload, Q4KAndQ5KHighScaleAndCodeBits) {
  const std::array<uint8_t, 12> packed_scales{0x81, 0x42, 0xc3, 0x04, 0x85, 0x46,
                                              0xc7, 0x08, 0xa9, 0xba, 0xcb, 0xdc};
  const std::array<int, 8> scales{1, 2, 3, 4, 41, 26, 59, 12};
  const std::array<int, 8> minima{5, 6, 7, 8, 42, 27, 60, 13};
  for (const auto *subtype : {"Q4_K", "Q5_K"}) {
    const bool q5 = std::string_view(subtype) == "Q5_K";
    std::string bytes(GgufLayout(subtype).type_size, '\0');
    PutHalf(bytes, 0, 0x3e00); // 1.5.
    PutHalf(bytes, 2, 0x3800); // 0.5.
    std::memcpy(bytes.data() + 4, packed_scales.data(), packed_scales.size());
    if (q5)
      for (size_t lane = 0; lane < 32; ++lane)
        bytes[16 + lane] = static_cast<char>(lane % 2 == 0 ? 0xaa : 0x55);
    for (size_t i = 0; i < 128; ++i)
      bytes[(q5 ? 48 : 16) + i] = static_cast<char>((i % 16) | ((15 - i % 16) << 4));
    const auto payload = ImportGgufTensorPayload(bytes, {256}, subtype);
    const auto parameters = ReadGgufBlockParameters(payload, 0);
    const auto values = Values(payload);
    ASSERT_EQ(values.size(), size_t{256});
    ASSERT_EQ(parameters.scales.size(), size_t{8});
    ASSERT_EQ(parameters.minima.size(), size_t{8});
    for (size_t group = 0; group < 8; ++group) {
      EXPECT_FLOAT_EQ(parameters.scales[group], 1.5f * scales[group]);
      EXPECT_FLOAT_EQ(parameters.minima[group], 0.5f * minima[group]);
      for (size_t lane = 0; lane < 32; ++lane) {
        int code = group % 2 == 0 ? lane % 16 : 15 - lane % 16;
        if (q5 && (group + lane) % 2 == 1)
          code += 16;
        const float expected = (1.5f * scales[group]) * code - 0.5f * minima[group];
        EXPECT_FLOAT_EQ(values[group * 32 + lane], expected) << group << ":" << lane;
      }
    }
    EXPECT_EQ(ExportGgufTensorPayload(payload), bytes);
  }
}

TEST(GgufTensorPayload, Q6KPlanesAndSignedSubgroupScales) {
  std::string bytes(210, '\0');
  for (size_t i = 0; i < 128; ++i)
    bytes[i] = static_cast<char>((i * 7 + 3) & 255);
  for (size_t i = 0; i < 64; ++i)
    bytes[128 + i] = static_cast<char>((i * 13 + 5) & 255);
  const std::array<int, 16> scales{-128, -127, -64, -1, 0,  1,   64,   127,
                                   127,  64,   1,   0,  -1, -64, -127, -128};
  for (size_t i = 0; i < scales.size(); ++i)
    bytes[192 + i] = static_cast<char>(scales[i] & 255);
  PutHalf(bytes, 208, 0x3800);
  const auto payload = ImportGgufTensorPayload(bytes, {256}, "Q6_K");
  const auto parameters = ReadGgufBlockParameters(payload, 0);
  ASSERT_EQ(parameters.scales.size(), scales.size());
  EXPECT_TRUE(parameters.minima.empty());
  for (size_t i = 0; i < scales.size(); ++i)
    EXPECT_FLOAT_EQ(parameters.scales[i], 0.5f * scales[i]);
  const auto values = Values(payload);
  ASSERT_EQ(values.size(), size_t{256});
  for (size_t half = 0; half < 2; ++half)
    for (size_t lane = 0; lane < 32; ++lane) {
      const int first = static_cast<uint8_t>(bytes[half * 64 + lane]);
      const int second = static_cast<uint8_t>(bytes[half * 64 + 32 + lane]);
      const int high = static_cast<uint8_t>(bytes[128 + half * 32 + lane]);
      const std::array<int, 4> codes{
          (first & 15) + 16 * (high & 3) - 32, (second & 15) + 16 * ((high >> 2) & 3) - 32,
          (first >> 4) + 16 * ((high >> 4) & 3) - 32, (second >> 4) + 16 * ((high >> 6) & 3) - 32};
      for (size_t quarter = 0; quarter < 4; ++quarter) {
        const size_t index = half * 128 + quarter * 32 + lane;
        EXPECT_FLOAT_EQ(values[index], (0.5f * scales[index / 16]) * codes[quarter]);
      }
    }
}

TEST(GgufTensorPayload, MetadataShapeLengthOverflowAndBudgets) {
  const std::string bytes(18, '\0');
  const auto payload = ImportGgufTensorPayload(bytes, {32}, "Q4_0");
  for (const Shape &shape : {Shape{}, Shape{0}, Shape{0, -1, 32}, Shape{1, 31}, Shape{2, 16},
                             Shape{-1, 32}, Shape{std::numeric_limits<int64_t>::max() - 31},
                             Shape{std::numeric_limits<int64_t>::max(), 32},
                             Shape{std::numeric_limits<int64_t>::max(), 0, -1, 32}})
    EXPECT_THROW(ImportGgufTensorPayload(bytes, shape, "Q4_0"), std::invalid_argument);
  EXPECT_THROW(ImportGgufTensorPayload(bytes.substr(1), {32}, "Q4_0"), std::invalid_argument);
  EXPECT_THROW(ImportGgufTensorPayload(bytes + std::string(1, '\0'), {32}, "Q4_0"),
               std::invalid_argument);
  EXPECT_THROW(ImportGgufTensorPayload(bytes, {32}, "Q4_0", "other"), std::invalid_argument);
  EXPECT_THROW(ImportGgufTensorPayload(bytes, {32}, "Q4_0", kGgufLayoutVersion, "other"),
               std::invalid_argument);
  EXPECT_NO_THROW(
      ImportGgufTensorPayload(bytes, {32}, "Q4_0", kGgufLayoutVersion, kGgufUpstreamRevision, 128));
  for (const size_t limit : {size_t{0}, size_t{17}, size_t{18}, size_t{127}}) {
    EXPECT_THROW(ImportGgufTensorPayload(bytes, {32}, "Q4_0", kGgufLayoutVersion,
                                         kGgufUpstreamRevision, limit),
                 std::invalid_argument);
    EXPECT_THROW(ExportGgufTensorPayload(payload, limit), std::invalid_argument);
    EXPECT_THROW(DequantizeGgufTensorPayload(payload, limit), std::invalid_argument);
    EXPECT_THROW(ReadGgufBlockParameters(payload, 0, limit), std::invalid_argument);
  }
  EXPECT_THROW(ReadGgufBlockParameters(payload, 1), std::invalid_argument);
  EXPECT_THROW(ReadGgufBlockParameters(payload, std::numeric_limits<size_t>::max()),
               std::invalid_argument);
  auto mutated = payload;
  for (const size_t field : {size_t{0}, size_t{1}, size_t{2}, size_t{3}}) {
    mutated = payload;
    if (field == 0)
      mutated.mutable_struct_type()->set_name("onnx_light.gguf.ggml-blocks-v2/Q4_0");
    else if (field == 1)
      mutated.mutable_struct_type()
          ->mutable_structure()
          ->mutable_field(0)
          ->mutable_constant()
          ->set_raw_data("unknown");
    else if (field == 2)
      mutated.mutable_logical_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(0)
          ->set_dim_value(31);
    else
      mutated.set_raw_data(bytes + '\0');
    EXPECT_THROW(ExportGgufTensorPayload(mutated), std::invalid_argument);
    EXPECT_THROW(DequantizeGgufTensorPayload(mutated), std::invalid_argument);
    EXPECT_THROW(ReadGgufBlockParameters(mutated, 0), std::invalid_argument);
  }
}

TEST(GgufTensorPayload, NonfiniteParametersAreRejectedEvenBehindZeroScales) {
  for (const auto &layout : GgufBlockLayouts()) {
    std::string bytes(layout.type_size * 2, '\0');
    for (const uint16_t bits : {uint16_t{0x7c00}, uint16_t{0xfc00}, uint16_t{0x7e01}}) {
      for (const size_t offset :
           layout.subtype == "Q6_K"
               ? std::vector<size_t>{208}
               : (layout.subtype == "Q4_K" || layout.subtype == "Q5_K" ? std::vector<size_t>{0, 2}
                                                                       : std::vector<size_t>{0})) {
        bytes.assign(layout.type_size * 2, '\0');
        PutHalf(bytes, layout.type_size + offset, bits);
        EXPECT_THROW(ImportGgufTensorPayload(bytes, {2, static_cast<int64_t>(layout.block_size)},
                                             layout.subtype),
                     std::invalid_argument);
        auto payload =
            ImportGgufTensorPayload(std::string(layout.type_size * 2, '\0'),
                                    {2, static_cast<int64_t>(layout.block_size)}, layout.subtype);
        payload.set_raw_data(bytes);
        EXPECT_THROW(ExportGgufTensorPayload(payload), std::invalid_argument);
        EXPECT_THROW(DequantizeGgufTensorPayload(payload), std::invalid_argument);
        EXPECT_THROW(ReadGgufBlockParameters(payload, 0), std::invalid_argument);
      }
    }

    bytes.assign(layout.type_size * 2, '\0');
    PutHalf(bytes, layout.subtype == "Q6_K" ? 208 : 0, 0x0001);
    EXPECT_NO_THROW(ImportGgufTensorPayload(bytes, {2, static_cast<int64_t>(layout.block_size)},
                                            layout.subtype));
    bytes.assign(layout.type_size * 2, '\xff');
    for (size_t block = 0; block < 2; ++block) {
      PutHalf(bytes, block * layout.type_size + (layout.subtype == "Q6_K" ? 208 : 0), 0x7bff);
      if (layout.subtype == "Q4_K" || layout.subtype == "Q5_K")
        PutHalf(bytes, block * layout.type_size + 2, 0x7bff);
    }
    const auto largest = ImportGgufTensorPayload(
        bytes, {2, static_cast<int64_t>(layout.block_size)}, layout.subtype);
    EXPECT_EQ(ExportGgufTensorPayload(largest), bytes);
    EXPECT_NO_THROW(Values(largest));
  }
}

TEST(GptqModelGgufWeights, ImportsBorrowedMatrixAndExportsExactOwningBytes) {
  for (const auto &layout : GgufBlockLayouts()) {
    std::string bytes(layout.type_size * 4, '\xff');
    for (size_t block = 0; block < 4; ++block) {
      const size_t offset = block * layout.type_size;
      PutHalf(bytes, offset + (layout.subtype == "Q6_K" ? 208 : 0), 0x3800);
      if (layout.subtype == "Q4_K" || layout.subtype == "Q5_K")
        PutHalf(bytes, offset + 2, 0x3c00);
    }
    GptqModelGgufWeights checkpoint;
    checkpoint.shape = {2, static_cast<int64_t>(layout.block_size * 2)};
    checkpoint.subtype = layout.subtype;
    checkpoint.qweight = Tensor::Borrow(
        "qweight", TensorProto::UINT8, {2, static_cast<int64_t>(layout.type_size * 2)},
        reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());
    const auto payload = ImportGptqModelGgufWeights(checkpoint);
    const auto exported = ExportGptqModelGgufWeights(payload);
    EXPECT_EQ(exported.qweight.name, "qweight");
    EXPECT_EQ(exported.qweight.data_type, TensorProto::UINT8);
    EXPECT_EQ(exported.qweight.shape, checkpoint.qweight.shape);
    EXPECT_EQ(exported.shape, checkpoint.shape);
    EXPECT_EQ(exported.subtype, checkpoint.subtype);
    EXPECT_EQ(exported.layout_version, checkpoint.layout_version);
    EXPECT_EQ(exported.upstream_revision, checkpoint.upstream_revision);
    ASSERT_EQ(exported.qweight.size_bytes(), bytes.size());
    EXPECT_EQ(std::string(reinterpret_cast<const char *>(exported.qweight.bytes()), bytes.size()),
              bytes);
    EXPECT_EQ(ImportGptqModelGgufWeights(exported).raw_data(), bytes);
    bytes[0] = static_cast<char>(static_cast<uint8_t>(bytes[0]) ^ 1);
    EXPECT_EQ(ImportGptqModelGgufWeights(exported).raw_data(), payload.raw_data());
    EXPECT_NE(payload.raw_data(), bytes);
  }
}

TEST(GptqModelGgufWeights, RejectsPaddingShapeDtypeMetadataAndLimitsBeforeCopies) {
  GptqModelGgufWeights checkpoint;
  checkpoint.shape = {2, 32};
  checkpoint.subtype = "Q4_0";
  checkpoint.qweight = Tensor("qweight", TensorProto::UINT8, {2, 18}, std::vector<uint8_t>(36));
  const auto payload = ImportGptqModelGgufWeights(checkpoint);
  for (const Shape &shape :
       {Shape{}, Shape{64}, Shape{1, 2, 32}, Shape{0, -1}, Shape{-1, 32}, Shape{2, 0}, Shape{2, 31},
        Shape{2, 33}, Shape{4, 16}, Shape{std::numeric_limits<int64_t>::max(), 32}}) {
    auto invalid = checkpoint;
    invalid.shape = shape;
    EXPECT_THROW(ImportGptqModelGgufWeights(invalid), std::invalid_argument);
  }
  for (const Shape &shape :
       {Shape{}, Shape{36}, Shape{1, 2, 18}, Shape{0, -1}, Shape{-1, 18}, Shape{2, 0}, Shape{1, 36},
        Shape{2, 19}, Shape{4, 9}, Shape{std::numeric_limits<int64_t>::max(), 18}}) {
    auto invalid = checkpoint;
    invalid.qweight.shape = shape;
    EXPECT_THROW(ImportGptqModelGgufWeights(invalid), std::invalid_argument);
  }
  for (const int32_t dtype : {TensorProto::INT8, TensorProto::FLOAT, TensorProto::INT32}) {
    auto invalid = checkpoint;
    invalid.qweight.data_type = dtype;
    EXPECT_THROW(ImportGptqModelGgufWeights(invalid), std::invalid_argument);
  }
  for (const size_t size : {size_t{0}, size_t{35}, size_t{37}}) {
    auto invalid = checkpoint;
    invalid.qweight = Tensor("qweight", TensorProto::UINT8, {2, 18}, std::vector<uint8_t>(size));
    EXPECT_THROW(ImportGptqModelGgufWeights(invalid), std::invalid_argument);
  }
  for (const size_t limit : {size_t{0}, size_t{35}, size_t{36}, size_t{255}}) {
    EXPECT_THROW(ImportGptqModelGgufWeights(checkpoint, limit), std::invalid_argument);
    EXPECT_THROW(ExportGptqModelGgufWeights(payload, limit), std::invalid_argument);
  }
  EXPECT_NO_THROW(ImportGptqModelGgufWeights(checkpoint, 256));
  EXPECT_NO_THROW(ExportGptqModelGgufWeights(payload, 256));
  auto invalid = checkpoint;
  invalid.layout_version.clear();
  EXPECT_THROW(ImportGptqModelGgufWeights(invalid), std::invalid_argument);
  invalid = checkpoint;
  invalid.upstream_revision.clear();
  EXPECT_THROW(ImportGptqModelGgufWeights(invalid), std::invalid_argument);
  invalid = checkpoint;
  invalid.subtype = "q4_0";
  EXPECT_THROW(ImportGptqModelGgufWeights(invalid), std::invalid_argument);
  auto invalid_payload = payload;
  invalid_payload.mutable_logical_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  invalid_payload.mutable_logical_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->add_dim()
      ->set_dim_value(64);
  EXPECT_THROW(ExportGptqModelGgufWeights(invalid_payload), std::invalid_argument);
  invalid.qweight = Tensor("qweight", TensorProto::UINT8, {2, 18}, std::vector<uint8_t>(36));
  invalid.subtype = "Q4_0";
  invalid.qweight.mutable_bytes()[19] = 0x7c;
  EXPECT_THROW(ImportGptqModelGgufWeights(invalid), std::invalid_argument);
  invalid_payload = payload;
  auto bad_bytes = std::string(payload.raw_data());
  bad_bytes[19] = 0x7c;
  invalid_payload.set_raw_data(bad_bytes);
  EXPECT_THROW(ExportGptqModelGgufWeights(invalid_payload), std::invalid_argument);
}

TEST(GgufTensorPayload, NativeStorageUsesStandardEncodedValuesAndRejectsPortableAliases) {
  size_t native_formats = 0;
  for (const auto format : QuantizationFormats())
    if (IsGgufQuantizationFormat(format)) {
      ++native_formats;
      EXPECT_EQ(GgufLayout(QuantizationFormatName(format)).subtype, QuantizationFormatName(format));
    }
  EXPECT_EQ(native_formats, GgufBlockLayouts().size());
  for (const auto &layout : GgufBlockLayouts()) {
    const auto format = ParseQuantizationFormat(layout.subtype);
    EXPECT_TRUE(IsGgufQuantizationFormat(format));
    auto portable_plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 1);
    portable_plan.format = format;
    EXPECT_THROW(MakeQuantizationType(portable_plan), std::invalid_argument);
    EXPECT_THROW(QuantizeTensor(Tensor::FromFloat("", {1}, {0}), portable_plan),
                 std::invalid_argument);
    for (const auto prefix : {"onnx_light.quantization.v1/", "onnx_light.quantization.v2/"}) {
      auto portable = QuantizeTensor(Tensor::FromFloat("", {1}, {0}),
                                     MakeQuantizationPlan(QuantizationFormat::kInt4, 1))
                          .Encoded();
      portable.mutable_struct_type()->set_name(std::string(prefix) + layout.subtype);
      EXPECT_THROW(DequantizeTensor(portable), std::invalid_argument);
      EXPECT_THROW(MakeSharedQuantizationType(portable.struct_type()), std::invalid_argument);
    }
    const std::string bytes(layout.type_size * 2, '\0');
    const auto encoded = ImportGgufTensorPayload(
        bytes, {2, static_cast<int64_t>(layout.block_size)}, layout.subtype);
    EncodedValueProto restored;
    ASSERT_TRUE(restored.ParseFromString(encoded.SerializeAsString()));
    EXPECT_EQ(restored.raw_data(), bytes);
    const auto direct = DequantizeGgufTensorPayload(restored);
    const auto generic = DequantizeTensor(restored);
    const auto runtime = DequantizeTensor(RuntimeValue(restored));
    const auto decoded_proto = DequantizeTensorProto(restored);
    const auto proto = TensorFromProto(decoded_proto);
    EXPECT_EQ(generic.shape, direct.shape);
    EXPECT_EQ(std::memcmp(direct.bytes(), generic.bytes(), direct.size_bytes()), 0);
    EXPECT_EQ(std::memcmp(direct.bytes(), runtime.bytes(), direct.size_bytes()), 0);
    EXPECT_EQ(std::memcmp(direct.bytes(), proto.bytes(), direct.size_bytes()), 0);
    ModelProto model;
    *model.add_struct_types() = restored.struct_type();
    model.mutable_struct_types(0)->set_type_id(17);
    auto referenced = restored;
    referenced.mutable_struct_type()->Clear();
    referenced.mutable_struct_type()->set_type_ref(17);
    StructTypeCatalogue catalogue;
    catalogue.Build(model);
    SimpleRawBufferAllocator allocator(4);
    const auto allocated = DequantizeTensor(referenced, catalogue, &allocator);
    EXPECT_EQ(allocated.allocation_owner(), &allocator);
    EXPECT_EQ(std::memcmp(direct.bytes(), allocated.bytes(), direct.size_bytes()), 0);
    EXPECT_EQ(ExportGgufTensorPayload(referenced, size_t{1} << 30, catalogue), bytes);
    EXPECT_THROW(ExportGgufTensorPayload(referenced), std::invalid_argument);
    auto invalid = restored;
    invalid.set_parameter_ref("ignored");
    EXPECT_THROW(DequantizeGgufTensorPayload(invalid), std::invalid_argument);
    invalid = restored;
    invalid.mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
    EXPECT_THROW(DequantizeTensor(invalid), std::invalid_argument);
  }
  for (const auto name :
       {"q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq4_nl", "iq1_s", "tq1_0", "tq2_0"}) {
    EXPECT_THROW(ParseQuantizationFormat(name), std::invalid_argument);
    auto invalid = QuantizeTensor(Tensor::FromFloat("", {1}, {0}),
                                  MakeQuantizationPlan(QuantizationFormat::kInt4, 1))
                       .Encoded();
    invalid.mutable_struct_type()->set_name(std::string("onnx_light.quantization.v1/") + name);
    EXPECT_THROW(DequantizeTensor(invalid), std::invalid_argument);
    invalid.mutable_struct_type()->set_name(std::string("onnx_light.quantization.v2/") + name);
    EXPECT_THROW(DequantizeTensor(invalid), std::invalid_argument);
  }
}
