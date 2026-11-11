// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

// Produced by executing the pinned PackableQuantLinear.pack_original NumPy
// packing statements and planar_pack_rows with codes (arange(32)*7+3) % (1 << bits).
// Pin: c1976918cfe3703fbcf6d1ce2440a82b26394375. The planar source executes
// through a NumPy-backed Torch tensor shim, without importing GPTQModel.
std::vector<uint32_t> UpstreamWords(uint32_t bits, bool planar) {
  switch (bits) {
  case 2:
    return {0x1b1b1b1b, 0x1b1b1b1b};
  case 3:
    return planar ? std::vector<uint32_t>{0x1b1b1b1b, 0x1b1b1b1b, 0xf0f0f0f0}
                  : std::vector<uint32_t>{0x53977053, 0x70539770, 0x97705397};
  case 4:
    return {0x4d6f81a3, 0xc5e7092b, 0x4d6f81a3, 0xc5e7092b};
  case 5:
    return {0x4d6f81a3, 0xc5e7092b, 0x4d6f81a3, 0xc5e7092b, 0xc663399c};
  case 6:
    return {0x4d6f81a3, 0xc5e7092b, 0x4d6f81a3, 0xc5e7092b, 0xa543e950, 0x503e940f};
  case 7:
    return {0x4d6f81a3, 0xc5e7092b, 0x4d6f81a3, 0xc5e7092b, 0xa543e950, 0x503e940f, 0xf803fe00};
  default:
    return {0x18110a03, 0x342d261f, 0x5049423b, 0x6c655e57,
            0x88817a73, 0xa49d968f, 0xc0b9b2ab, 0xdcd5cec7};
  }
}

// Produced by the same upstream pack_original statements with zero fields
// (arange(32)*5+3) % ((1 << bits)-1), excluding the ambiguous all-ones field.
std::vector<uint32_t> UpstreamV1Zeros(uint32_t bits) {
  switch (bits) {
  case 2:
    return {0x18618618, 0x86186186};
  case 3:
    return {0x3174298b, 0xd0a62e85, 0x98ba14c5};
  case 4:
    return {0x83d83d83, 0x3d83d83d, 0xd83d83d8, 0x83d83d83};
  default:
    return {0x120d0803, 0x26211c17, 0x3a35302b, 0x4e49443f,
            0x625d5853, 0x76716c67, 0x8a85807b, 0x9e99948f};
  }
}

void NativeWord(uint8_t *destination, const std::vector<uint32_t> &words, size_t index,
                size_t width) {
  uint64_t word = 0;
  for (size_t byte = 0; byte < width; ++byte) {
    const size_t position = index * width + byte;
    word |= uint64_t((words[position / 4] >> ((position % 4) * 8)) & 255) << (byte * 8);
  }
  if (width == 8)
    std::memcpy(destination, &word, width);
  else if (width == 4) {
    const auto value = static_cast<uint32_t>(word);
    std::memcpy(destination, &value, width);
  } else if (width == 2) {
    const auto value = static_cast<uint16_t>(word);
    std::memcpy(destination, &value, width);
  } else
    *destination = static_cast<uint8_t>(word);
}

template <typename T>
Tensor Values(const char *name, int32_t type, Shape shape, const std::vector<T> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty())
    std::memcpy(bytes.data(), values.data(), bytes.size());
  return Tensor(name, type, shape, std::move(bytes));
}

GptqWeights Fixture(uint32_t bits = 4, std::string format = "gptq_v2",
                    int32_t word_type = TensorProto::INT32) {
  GptqWeights result;
  result.format = std::move(format);
  result.in_features = 32;
  result.out_features = 32;
  result.group_size = 16;
  result.bits = bits;
  result.word_dtype = word_type;
  const bool planar = bits == 5 || bits == 6 || bits == 7 || result.format == "gptq_p";
  const auto words = UpstreamWords(bits, planar);
  const size_t width = word_type == TensorProto::INT8    ? 1
                       : word_type == TensorProto::INT16 ? 2
                       : word_type == TensorProto::INT32 ? 4
                                                         : 8;
  const size_t rows = words.size() * 4 / width;
  std::vector<uint8_t> weight(words.size() * 4 * 32), zeros(words.size() * 4 * 2);
  for (size_t row = 0; row < rows; ++row)
    for (size_t n = 0; n < 32; ++n)
      NativeWord(weight.data() + (row * 32 + n) * width, words, row, width);
  const auto zero_words = result.format == "gptq" ? UpstreamV1Zeros(bits) : words;
  for (size_t group = 0; group < 2; ++group)
    for (size_t row = 0; row < rows; ++row)
      NativeWord(zeros.data() + (group * rows + row) * width, zero_words, row, width);
  result.tensors.emplace(
      "qweight", Tensor("qweight", word_type, {static_cast<int64_t>(rows), 32}, std::move(weight)));
  result.tensors.emplace(
      "qzeros", Tensor("qzeros", word_type, {2, static_cast<int64_t>(rows)}, std::move(zeros)));
  std::vector<float> scales(64);
  for (size_t i = 0; i < scales.size(); ++i)
    scales[i] = i < 32 ? 0.5f : 1.5f;
  result.tensors.emplace("scales", MakeFloat16Tensor("scales", {2, 32}, scales));
  std::vector<int32_t> groups(32);
  for (size_t i = 0; i < groups.size(); ++i)
    groups[i] = static_cast<int32_t>(i % 2);
  result.tensors.emplace("g_idx", Values("g_idx", TensorProto::INT32, {32}, groups));
  return result;
}

template <typename T> T At(const Tensor &tensor, size_t index) {
  T value;
  std::memcpy(&value, tensor.bytes() + index * sizeof(T), sizeof(T));
  return value;
}

void EqualTensor(const Tensor &actual, const Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  if (actual.size_bytes() != 0) {
    EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  }
}

} // namespace

TEST(GptqWeights, UpstreamFixturesRoundTripUnpackAndReconstruct) {
  for (const std::string format : {"gptq", "gptq_v2", "gptq_p"})
    for (uint32_t bits = 2; bits <= 8; ++bits) {
      if (format != "gptq_p" && bits >= 5 && bits <= 7)
        continue;
      SCOPED_TRACE(format + "/" + std::to_string(bits));
      const auto checkpoint = Fixture(bits, format);
      const auto encoded = ImportGptqWeights(checkpoint);
      EncodedValueProto restored;
      ASSERT_TRUE(restored.ParseFromString(encoded.SerializeAsString()));
      const auto exported = ExportGptqWeights(restored);
      EXPECT_EQ(exported.format, format);
      EXPECT_EQ(exported.bits, bits);
      EXPECT_EQ(exported.group_axis, 1);
      EXPECT_EQ(exported.upstream_revision, checkpoint.upstream_revision);
      for (const auto &[name, tensor] : checkpoint.tensors)
        EqualTensor(exported.tensors.at(name), tensor);
      const auto codes = UnpackGptqCodes(restored);
      const auto zeros = UnpackGptqZeros(restored);
      const auto weights = ReconstructGptqWeights(restored);
      EXPECT_EQ(codes.shape, (Shape{32, 32}));
      EXPECT_EQ(codes.data_type, TensorProto::UINT8);
      EXPECT_EQ(zeros.shape, (Shape{2, 32}));
      EXPECT_EQ(zeros.data_type, TensorProto::INT16);
      const int maxq = (1 << bits) - 1;
      for (size_t n = 0; n < 32; ++n) {
        const auto zero = format == "gptq" ? static_cast<int>((n * 5 + 3) % maxq) + 1
                                           : static_cast<int>((n * 7 + 3) & maxq);
        for (size_t group = 0; group < 2; ++group)
          EXPECT_EQ(At<int16_t>(zeros, group * 32 + n), zero);
        for (size_t k = 0; k < 32; ++k) {
          EXPECT_EQ(codes.bytes()[n * 32 + k], (k * 7 + 3) & maxq);
          const float expected =
              (static_cast<int>((k * 7 + 3) & maxq) - zero) * (k % 2 == 0 ? 0.5f : 1.5f);
          EXPECT_EQ(At<float>(weights, n * 32 + k), expected);
        }
      }
      EqualTensor(DequantizeTensor(restored), weights);
      auto metadata = checkpoint;
      metadata.tensors.clear();
      const auto packed = PackGptqWeights(metadata, codes, zeros, checkpoint.tensors.at("scales"),
                                          checkpoint.tensors.at("g_idx"));
      EXPECT_EQ(packed.raw_data(), encoded.raw_data());
      EXPECT_EQ(packed.struct_type().SerializeAsString(),
                encoded.struct_type().SerializeAsString());
    }
}

TEST(GptqWeights, PreservesEverySupportedContinuousWordWidth) {
  for (const auto type : {TensorProto::INT8, TensorProto::INT16, TensorProto::INT32})
    for (uint32_t bits : {2, 4, 8}) {
      SCOPED_TRACE(std::to_string(type) + "/" + std::to_string(bits));
      const auto checkpoint = Fixture(bits, "gptq_v2", type);
      const auto encoded = ImportGptqWeights(checkpoint);
      const auto exported = ExportGptqWeights(encoded);
      EXPECT_EQ(exported.word_dtype, type);
      for (const auto &[name, tensor] : checkpoint.tensors)
        EqualTensor(exported.tensors.at(name), tensor);
      const auto codes = UnpackGptqCodes(encoded);
      for (size_t n = 0; n < 32; ++n)
        for (size_t k = 0; k < 32; ++k)
          EXPECT_EQ(codes.bytes()[n * 32 + k], (k * 7 + 3) % (1 << bits));
      auto metadata = checkpoint;
      metadata.tensors.clear();
      EXPECT_EQ(PackGptqWeights(metadata, codes, UnpackGptqZeros(encoded),
                                checkpoint.tensors.at("scales"), checkpoint.tensors.at("g_idx"))
                    .raw_data(),
                encoded.raw_data());
    }
}

TEST(GptqWeights, UpstreamSparseCrossingFixtures) {
  // Outputs of the pinned pack_original NumPy stages with only x[10] and
  // x[21] equal to maxq, all remaining logical fields zero.
  for (uint32_t bits : {2, 3, 4, 8}) {
    const std::vector<uint32_t> words =
        bits == 2   ? std::vector<uint32_t>{0x00300000, 0x00000c00}
        : bits == 3 ? std::vector<uint32_t>{0xc0000000, 0x80000001, 0x00000003}
        : bits == 4 ? std::vector<uint32_t>{0, 0x00000f00, 0x00f00000, 0}
                    : std::vector<uint32_t>{0, 0, 0x00ff0000, 0, 0, 0x0000ff00, 0, 0};
    auto checkpoint = Fixture(bits);
    auto &weight = checkpoint.tensors.at("qweight");
    auto &zero = checkpoint.tensors.at("qzeros");
    for (size_t row = 0; row < words.size(); ++row) {
      for (size_t n = 0; n < 32; ++n)
        NativeWord(weight.mutable_bytes() + (row * 32 + n) * 4, words, row, 4);
      for (size_t group = 0; group < 2; ++group)
        NativeWord(zero.mutable_bytes() + (group * words.size() + row) * 4, words, row, 4);
    }
    const auto encoded = ImportGptqWeights(checkpoint);
    const auto codes = UnpackGptqCodes(encoded);
    const auto zeros = UnpackGptqZeros(encoded);
    for (size_t n = 0; n < 32; ++n) {
      EXPECT_EQ(At<int16_t>(zeros, n), n == 10 || n == 21 ? (1 << bits) - 1 : 0);
      for (size_t k = 0; k < 32; ++k)
        EXPECT_EQ(codes.bytes()[n * 32 + k], k == 10 || k == 21 ? (1 << bits) - 1 : 0);
    }
  }
}

TEST(GptqWeights, UpstreamPlanarFixturesKeepDistinctAdjacentBlocks) {
  for (uint32_t bits : {3, 5, 6, 7}) {
    // The second block is a separate pinned planar_pack_rows execution with
    // (arange(32)*3+1) % (1 << bits); the first uses UpstreamWords' input.
    const std::vector<uint32_t> second =
        bits == 3 ? std::vector<uint32_t>{0xb1b1b1b1, 0xb1b1b1b1, 0x96969696}
        : bits == 5
            ? std::vector<uint32_t>{0x630da741, 0xeb852fc9, 0x630da741, 0xeb852fc9, 0xf81f07e0}
        : bits == 6 ? std::vector<uint32_t>{0x630da741, 0xeb852fc9, 0x630da741,
                                            0xeb852fc9, 0xaa955400, 0x554003ff}
                    : std::vector<uint32_t>{0x630da741, 0xeb852fc9, 0x630da741, 0xeb852fc9,
                                            0xaa955400, 0x554003ff, 0xffe00000};
    auto words = UpstreamWords(bits, true);
    words.insert(words.end(), second.begin(), second.end());
    GptqWeights checkpoint;
    checkpoint.format = "gptq_p";
    checkpoint.bits = bits;
    checkpoint.in_features = checkpoint.out_features = 64;
    checkpoint.group_size = 16;
    std::vector<uint32_t> weight(words.size() * 64), zeros(words.size() * 4);
    for (size_t row = 0; row < words.size(); ++row) {
      for (size_t n = 0; n < 64; ++n)
        weight[row * 64 + n] = words[row];
      for (size_t group = 0; group < 4; ++group)
        zeros[group * words.size() + row] = words[row];
    }
    checkpoint.tensors.emplace("qweight", Values("qweight", TensorProto::INT32,
                                                 {static_cast<int64_t>(words.size()), 64}, weight));
    checkpoint.tensors.emplace("qzeros", Values("qzeros", TensorProto::INT32,
                                                {4, static_cast<int64_t>(words.size())}, zeros));
    checkpoint.tensors.emplace("scales",
                               MakeFloat16Tensor("scales", {4, 64}, std::vector<float>(256, 0.5f)));
    std::vector<int32_t> groups(64);
    for (size_t k = 0; k < 64; ++k)
      groups[k] = static_cast<int32_t>(k % 4);
    checkpoint.tensors.emplace("g_idx", Values("g_idx", TensorProto::INT32, {64}, groups));
    const auto encoded = ImportGptqWeights(checkpoint);
    const auto codes = UnpackGptqCodes(encoded);
    const auto logical_zeros = UnpackGptqZeros(encoded);
    auto expected = [&](size_t i) {
      return static_cast<int>((i < 32 ? i * 7 + 3 : (i - 32) * 3 + 1) % (1 << bits));
    };
    for (size_t n = 0; n < 64; ++n) {
      EXPECT_EQ(At<int16_t>(logical_zeros, n), expected(n));
      for (size_t k = 0; k < 64; ++k)
        EXPECT_EQ(codes.bytes()[n * 64 + k], expected(k));
    }
    auto metadata = checkpoint;
    metadata.tensors.clear();
    EXPECT_EQ(PackGptqWeights(metadata, codes, logical_zeros, checkpoint.tensors.at("scales"),
                              checkpoint.tensors.at("g_idx"))
                  .raw_data(),
              encoded.raw_data());
  }
}

TEST(GptqWeights, PreservesPartialWordsPaddingAndPartialGroups) {
  GptqWeights checkpoint;
  checkpoint.in_features = checkpoint.out_features = 3;
  checkpoint.group_size = 16;
  checkpoint.tensors.emplace("qweight", Values<uint32_t>("qweight", TensorProto::INT32, {1, 3},
                                                         {0xdeadf210, 0xdeadf210, 0xdeadf210}));
  checkpoint.tensors.emplace("qzeros",
                             Values<uint32_t>("qzeros", TensorProto::INT32, {1, 1}, {0xabcdf210}));
  checkpoint.tensors.emplace("scales", MakeFloat16Tensor("scales", {1, 3}, {1, 1, 1}));
  checkpoint.tensors.emplace("g_idx", Values<int32_t>("g_idx", TensorProto::INT32, {3}, {0, 0, 0}));
  const auto encoded = ImportGptqWeights(checkpoint);
  const auto exported = ExportGptqWeights(encoded);
  for (const auto &[name, tensor] : checkpoint.tensors)
    EqualTensor(exported.tensors.at(name), tensor);
  const auto codes = UnpackGptqCodes(encoded);
  const auto zeros = UnpackGptqZeros(encoded);
  for (size_t n = 0; n < 3; ++n)
    for (size_t k = 0; k < 3; ++k)
      EXPECT_EQ(codes.bytes()[n * 3 + k], k);
  EXPECT_EQ(At<int16_t>(zeros, 2), 2);
  checkpoint.group_size = -1;
  EXPECT_NO_THROW(ImportGptqWeights(checkpoint));
  checkpoint.format = "gptq";
  EXPECT_NO_THROW(ImportGptqWeights(checkpoint));
}

TEST(GptqWeights, RejectsAmbiguousV1ExtremaAndUnrepresentableZeros) {
  for (uint32_t bits : {2, 3, 4, 8}) {
    auto checkpoint = Fixture(bits, "gptq");
    auto &packed_zeros = checkpoint.tensors.at("qzeros");
    std::memset(packed_zeros.mutable_bytes(), 0xff, packed_zeros.size_bytes());
    EXPECT_ANY_THROW(ImportGptqWeights(checkpoint));
    checkpoint = Fixture(bits, "gptq");
    const auto encoded = ImportGptqWeights(checkpoint);
    auto ambiguous = encoded;
    std::memset(ambiguous.mutable_raw_data()->data() +
                    checkpoint.tensors.at("qweight").size_bytes(),
                0xff, checkpoint.tensors.at("qzeros").size_bytes());
    EXPECT_ANY_THROW(ExportGptqWeights(ambiguous));
    EXPECT_ANY_THROW(UnpackGptqCodes(ambiguous));
    EXPECT_ANY_THROW(UnpackGptqZeros(ambiguous));
    EXPECT_ANY_THROW(ReconstructGptqWeights(ambiguous));
    auto zeros = UnpackGptqZeros(encoded);
    auto metadata = checkpoint;
    metadata.tensors.clear();
    const auto codes = UnpackGptqCodes(encoded);
    EXPECT_NO_THROW(PackGptqWeights(metadata, codes, zeros, checkpoint.tensors.at("scales"),
                                    checkpoint.tensors.at("g_idx")));
    for (size_t i = 0; i < 64; ++i) {
      const auto endpoint = static_cast<int16_t>(1 << bits);
      std::memcpy(zeros.mutable_bytes() + i * 2, &endpoint, 2);
    }
    EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, zeros, checkpoint.tensors.at("scales"),
                                     checkpoint.tensors.at("g_idx")));
    metadata.format = "gptq_v2";
    EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, zeros, checkpoint.tensors.at("scales"),
                                     checkpoint.tensors.at("g_idx")));
    for (size_t i = 0; i < 64; ++i) {
      const auto largest = static_cast<int16_t>((1 << bits) - 1);
      std::memcpy(zeros.mutable_bytes() + i * 2, &largest, 2);
    }
    metadata.format = "gptq";
    EXPECT_NO_THROW(PackGptqWeights(metadata, codes, zeros, checkpoint.tensors.at("scales"),
                                    checkpoint.tensors.at("g_idx")));
    std::memset(zeros.mutable_bytes(), 0, zeros.size_bytes());
    metadata.format = "gptq";
    EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, zeros, checkpoint.tensors.at("scales"),
                                     checkpoint.tensors.at("g_idx")));
    metadata.format = "gptq_v2";
    EXPECT_NO_THROW(PackGptqWeights(metadata, codes, zeros, checkpoint.tensors.at("scales"),
                                    checkpoint.tensors.at("g_idx")));
  }
}

TEST(GptqWeights, ResolvesCatalogueAndRejectsPortableSchemas) {
  const auto encoded = ImportGptqWeights(Fixture());
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(92);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  auto referenced = encoded;
  referenced.mutable_struct_type()->clear_structure();
  referenced.mutable_struct_type()->clear_name();
  referenced.mutable_struct_type()->set_type_ref(92);
  EXPECT_EQ(ExportGptqWeights(referenced, size_t{1} << 30, catalogue).format, "gptq_v2");
  EqualTensor(DequantizeTensor(referenced, catalogue), ReconstructGptqWeights(encoded));
  const auto tensor = Tensor::FromFloat("weight", {2}, {0, 1});
  const auto portable = QuantizeTensor(tensor, MakeQuantizationPlan(QuantizationFormat::kGptq, 2));
  EXPECT_ANY_THROW(ExportGptqWeights(portable.Encoded()));
  auto parameter = encoded;
  parameter.set_parameter_ref("shared");
  EXPECT_ANY_THROW(ExportGptqWeights(parameter));
}

TEST(GptqWeights, RejectsMalformedMetadataTensorsAndResourceLimits) {
  const auto good = Fixture();
  const auto encoded = ImportGptqWeights(good);
  EXPECT_ANY_THROW(ImportGptqWeights(good, 0));
  EXPECT_ANY_THROW(ImportGptqWeights(good, 4095));
  EXPECT_NO_THROW(ImportGptqWeights(good, 4096));
  EXPECT_ANY_THROW(ExportGptqWeights(encoded, 4095));
  for (const auto bits : {0, 1, 9}) {
    auto bad = good;
    bad.bits = bits;
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
  }
  for (const int64_t group : {int64_t{0}, int64_t{-2}}) {
    auto bad = good;
    bad.group_size = group;
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
  }
  auto bad = good;
  bad.in_features = -1;
  EXPECT_ANY_THROW(ImportGptqWeights(bad));
  bad = good;
  bad.in_features = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ImportGptqWeights(bad, std::numeric_limits<size_t>::max()));
  bad = good;
  bad.out_features = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ImportGptqWeights(bad, std::numeric_limits<size_t>::max()));
  bad = good;
  bad.group_axis = 0;
  EXPECT_ANY_THROW(ImportGptqWeights(bad));
  bad = good;
  bad.word_dtype = TensorProto::UINT32;
  EXPECT_ANY_THROW(ImportGptqWeights(bad));
  bad.word_dtype = TensorProto::INT64;
  EXPECT_ANY_THROW(ImportGptqWeights(bad));
  bad = good;
  bad.upstream_revision = "future";
  EXPECT_ANY_THROW(ImportGptqWeights(bad));
  bad = good;
  bad.format = "portable";
  EXPECT_ANY_THROW(ImportGptqWeights(bad));
  for (const std::string name : {"qweight", "qzeros", "scales", "g_idx"}) {
    bad = good;
    bad.tensors.erase(name);
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
    bad = good;
    bad.tensors.at(name).shape = {1};
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
    bad = good;
    bad.tensors.at(name) = Tensor(name, bad.tensors.at(name).data_type, bad.tensors.at(name).shape,
                                  std::vector<uint8_t>{1});
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
  }
  bad = good;
  bad.tensors.emplace("bias", Tensor{});
  EXPECT_ANY_THROW(ImportGptqWeights(bad));
  for (int32_t index : {-1, 2, std::numeric_limits<int32_t>::max()}) {
    bad = good;
    std::memcpy(bad.tensors.at("g_idx").mutable_bytes(), &index, 4);
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
  }
  for (uint16_t scale :
       {uint16_t{0}, uint16_t{0x8000}, uint16_t{0xbc00}, uint16_t{0x7c00}, uint16_t{0x7e00}}) {
    bad = good;
    std::memcpy(bad.tensors.at("scales").mutable_bytes(), &scale, 2);
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
  }
  for (uint32_t bits : {3, 5, 6, 7}) {
    bad = Fixture(bits, "gptq_p");
    bad.in_features = 31;
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
    bad = Fixture(bits, "gptq_p");
    bad.word_dtype = TensorProto::INT16;
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
  }
  for (uint32_t bits : {2, 4, 8}) {
    bad = Fixture(bits, "gptq_p", TensorProto::INT16);
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
  }
  for (uint32_t bits : {5, 6, 7}) {
    bad = Fixture(bits, "gptq");
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
    bad.format = "gptq_v2";
    EXPECT_ANY_THROW(ImportGptqWeights(bad));
  }
}

TEST(GptqWeights, RejectsMalformedEncodedSchemaAndLogicalType) {
  const auto good = ImportGptqWeights(Fixture());
  auto bad = good;
  bad.mutable_raw_data()->push_back(0);
  EXPECT_ANY_THROW(UnpackGptqCodes(bad));
  bad = good;
  bad.mutable_raw_data()->resize(bad.raw_data().size() - 1);
  EXPECT_ANY_THROW(ExportGptqWeights(bad));
  bad = good;
  bad.mutable_struct_type()->set_name("onnx_light.gptq_weights.v2");
  EXPECT_ANY_THROW(ReconstructGptqWeights(bad));
  bad = good;
  bad.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(5)
      ->mutable_constant()
      ->ref_int64_data()[0] = 0;
  EXPECT_ANY_THROW(ExportGptqWeights(bad));
  bad = good;
  bad.mutable_struct_type()->mutable_structure()->mutable_field(8)->set_name("renamed");
  EXPECT_ANY_THROW(ExportGptqWeights(bad));
  bad = good;
  bad.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(7)
      ->mutable_constant()
      ->set_raw_data("future");
  EXPECT_ANY_THROW(ExportGptqWeights(bad));
  bad = good;
  bad.mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
  EXPECT_ANY_THROW(ExportGptqWeights(bad));
  bad = good;
  bad.mutable_logical_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(
      31);
  EXPECT_ANY_THROW(ExportGptqWeights(bad));
}

TEST(GptqWeights, ValidatesPackingInputsBeforeLosslessEncoding) {
  auto metadata = Fixture();
  const auto encoded = ImportGptqWeights(metadata);
  auto codes = UnpackGptqCodes(encoded);
  auto zeros = UnpackGptqZeros(encoded);
  const auto scales = metadata.tensors.at("scales");
  const auto groups = metadata.tensors.at("g_idx");
  EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, zeros, scales, groups));
  metadata.tensors.clear();
  codes.mutable_bytes()[0] = 16;
  EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, zeros, scales, groups));
  codes.mutable_bytes()[0] = 0;
  auto bad_zeros = zeros;
  bad_zeros.data_type = TensorProto::UINT8;
  EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, bad_zeros, scales, groups));
  const int16_t negative = -1;
  std::memcpy(zeros.mutable_bytes(), &negative, 2);
  EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, zeros, scales, groups));
  zeros = UnpackGptqZeros(encoded);
  EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, zeros, scales, groups, 4095));
}

TEST(GptqWeights, PreservesScaleTypesAndRejectsFloatReconstructionOverflow) {
  for (const auto type : {TensorProto::BFLOAT16, TensorProto::FLOAT}) {
    auto checkpoint = Fixture();
    if (type == TensorProto::FLOAT)
      checkpoint.tensors.at("scales") =
          Tensor::FromFloat("scales", {2, 32}, std::vector<float>(64, 0.5f));
    else {
      auto &scales = checkpoint.tensors.at("scales");
      for (size_t i = 0; i < 64; ++i) {
        const auto bits = FloatToBfloat16Bits(0.5f);
        std::memcpy(scales.mutable_bytes() + i * 2, &bits, 2);
      }
      scales.data_type = type;
    }
    const auto encoded = ImportGptqWeights(checkpoint);
    EqualTensor(ExportGptqWeights(encoded).tensors.at("scales"), checkpoint.tensors.at("scales"));
    EXPECT_NO_THROW(ReconstructGptqWeights(encoded));
    if (type == TensorProto::FLOAT) {
      const float largest = std::numeric_limits<float>::max();
      std::memcpy(checkpoint.tensors.at("scales").mutable_bytes(), &largest, 4);
    } else {
      const uint16_t largest = 0x7f7f;
      std::memcpy(checkpoint.tensors.at("scales").mutable_bytes(), &largest, 2);
    }
    EXPECT_ANY_THROW(ImportGptqWeights(checkpoint));
  }
}

TEST(GptqWeights, SupportsEmptyLogicalDimensionsWithExplicitGroups) {
  for (const int64_t k : {0, 32})
    for (const int64_t n : {0, 32}) {
      if (k != 0 && n != 0)
        continue;
      GptqWeights metadata;
      metadata.in_features = k;
      metadata.out_features = n;
      metadata.group_size = 16;
      Tensor codes("codes", TensorProto::UINT8, {n, k}, {});
      Tensor zeros("zeros", TensorProto::INT16, {k / 16, n}, {});
      Tensor scales("scales", TensorProto::FLOAT16, {k / 16, n}, {});
      auto groups = Values<int32_t>("g_idx", TensorProto::INT32, {k},
                                    std::vector<int32_t>(static_cast<size_t>(k), 0));
      const auto encoded = PackGptqWeights(metadata, codes, zeros, scales, groups);
      EXPECT_EQ(ReconstructGptqWeights(encoded).size_bytes(), size_t{0});
      EXPECT_EQ(UnpackGptqCodes(encoded).shape, (Shape{n, k}));
      EXPECT_EQ(ExportGptqWeights(encoded).in_features, k);
    }
  GptqWeights metadata;
  metadata.bits = 2;
  metadata.out_features = std::numeric_limits<int64_t>::max();
  Tensor codes("codes", TensorProto::UINT8, {metadata.out_features, 0}, {});
  Tensor zeros("zeros", TensorProto::INT16, {0, metadata.out_features}, {});
  Tensor scales("scales", TensorProto::FLOAT16, {0, metadata.out_features}, {});
  Tensor groups("g_idx", TensorProto::INT32, {0}, {});
  const auto encoded = PackGptqWeights(metadata, codes, zeros, scales, groups, 1);
  EXPECT_EQ(encoded.raw_data().size(), size_t{1});
  EXPECT_EQ(ReconstructGptqWeights(encoded, 1).size_bytes(), size_t{0});
  EXPECT_EQ(UnpackGptqCodes(encoded, 1).size_bytes(), size_t{0});
  auto bad_marker = encoded;
  (*bad_marker.mutable_raw_data())[0] = 1;
  EXPECT_ANY_THROW(ExportGptqWeights(bad_marker));
  metadata.group_size = -1;
  EXPECT_ANY_THROW(PackGptqWeights(metadata, codes, zeros, scales, groups));
}
