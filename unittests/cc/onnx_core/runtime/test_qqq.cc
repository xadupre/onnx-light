// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

template <typename T>
Tensor Values(const char *name, int32_t type, Shape shape, const std::vector<T> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty())
    std::memcpy(bytes.data(), values.data(), bytes.size());
  return Tensor(name, type, std::move(shape), std::move(bytes));
}

template <typename T> T At(const Tensor &tensor, size_t index) {
  T result;
  std::memcpy(&result, tensor.bytes() + index * sizeof(T), sizeof(T));
  return result;
}

uint8_t Expected(size_t n, size_t k) {
  return static_cast<uint8_t>((n * 7 + k * 3 + k / 16 + k / 64) % 16);
}

float ChannelScale(size_t n) { return static_cast<float>(n % 31 + 1) / 32; }
float GroupScale(size_t group, size_t n) { return static_cast<float>((group + n) % 17) / 2; }

QqqWeights Fixture(int64_t k, int64_t n, int64_t group) {
  QqqWeights result;
  result.in_features = k;
  result.out_features = n;
  result.group_size = group;
  const bool grouped = group == 128 && k != 128;
  // Source _get_perms followed by reshape/permute/gather in pack(), not
  // the production decoder's inverse indexing.
  std::vector<size_t> perm;
  for (size_t i = 0; i < 32; ++i) {
    std::vector<size_t> first;
    for (size_t block : {0, 1})
      for (size_t row = 4 * (i % 4); row < 4 * (i % 4) + 4; ++row)
        first.push_back(16 * row + i / 4 + 8 * block);
    for (size_t j = 0; j < 4; ++j)
      for (size_t p : first)
        perm.push_back(p + 256 * j);
  }
  const std::vector<size_t> interleave = grouped ? std::vector<size_t>{0, 2, 4, 6, 1, 3, 5, 7}
                                                 : std::vector<size_t>{4, 0, 5, 1, 6, 2, 7, 3};
  auto original = perm;
  for (size_t start = 0; start < perm.size(); start += 8)
    for (size_t lane = 0; lane < 8; ++lane)
      perm[start + lane] = original[start + interleave[lane]];
  std::vector<uint8_t> tiled(static_cast<size_t>(k * n));
  size_t offset = 0;
  for (size_t kb = 0; kb < static_cast<size_t>(k / 16); ++kb)
    for (size_t nb = 0; nb < static_cast<size_t>(n / 16); ++nb)
      for (size_t row = 0; row < 16; ++row)
        for (size_t col = 0; col < 16; ++col)
          tiled[offset++] = Expected(nb * 16 + col, kb * 16 + row);
  std::vector<uint32_t> words(tiled.size() / 8);
  for (size_t block = 0; block < tiled.size(); block += 1024)
    for (size_t i = 0; i < 1024; ++i)
      words[(block + i) / 8] |= uint32_t(tiled[block + perm[i]]) << (4 * (i % 8));
  result.tensors.emplace("B", Values("B", TensorProto::INT32, {k / 16, 2 * n}, words));
  std::vector<size_t> scale_single;
  for (size_t i = 0; i < 4; ++i)
    for (size_t j : {0, 1, 8, 9, 16, 17, 24, 25})
      scale_single.push_back(2 * i + j);
  std::vector<float> channel(static_cast<size_t>(n));
  for (size_t i = 0; i < channel.size(); ++i)
    channel[i] = ChannelScale((i / 32) * 32 + scale_single[i % 32]);
  result.tensors.emplace("s_channel", Values("s_channel", TensorProto::FLOAT, {1, n}, channel));
  std::vector<float> scales(grouped ? static_cast<size_t>(k / 128 * n) : 0);
  std::vector<size_t> scale_perm;
  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 8; ++j)
      scale_perm.push_back(i + 8 * j);
  for (size_t i = 0; i < scales.size(); ++i) {
    const size_t source = (i / 64) * 64 + scale_perm[i % 64];
    scales[i] = GroupScale(source / static_cast<size_t>(n), source % static_cast<size_t>(n));
  }
  result.tensors.emplace(
      "s_group", MakeFloat16Tensor("s_group", grouped ? Shape{k / 128, n} : Shape{0}, scales));
  return result;
}

void EqualTensor(const Tensor &actual, const Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  if (actual.size_bytes()) {
    EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  }
}

QqqWeights Metadata(const QqqWeights &checkpoint) {
  auto result = checkpoint;
  result.tensors.clear();
  return result;
}

} // namespace

TEST(QqqWeights, SourceDerivedPermutationsRoundTripAndReconstruct) {
  for (const int64_t k : {64, 128, 256, 384})
    for (const int64_t n : {64, 128, 320})
      for (const int64_t group : {-1, 128}) {
        if (group == 128 && k == 64)
          continue;
        const auto checkpoint = Fixture(k, n, group);
        const bool grouped = group == 128 && k != 128;
        const auto &b = checkpoint.tensors.at("B");
        EXPECT_EQ(At<uint32_t>(b, 0), grouped ? 0x1b93e860u : 0x916e3b08u);
        EXPECT_EQ(At<uint32_t>(b, 31), grouped ? 0xc64e931bu : 0x4c19e6b3u);
        EXPECT_EQ(At<uint32_t>(b, 127), grouped ? 0x60e83db5u : 0xe6b3805du);
        const auto encoded = ImportQqqWeights(checkpoint);
        EncodedValueProto restored;
        restored.ParseFromString(encoded.SerializeAsString());
        const auto exported = ExportQqqWeights(restored);
        EXPECT_EQ(exported.in_features, k);
        EXPECT_EQ(exported.out_features, n);
        EXPECT_EQ(exported.group_size, group);
        EXPECT_EQ(exported.bits, 4u);
        EXPECT_EQ(exported.upstream_revision, checkpoint.upstream_revision);
        for (const auto &[name, tensor] : checkpoint.tensors)
          EqualTensor(exported.tensors.at(name), tensor);
        const auto codes = UnpackQqqCodes(restored);
        const auto weights = ReconstructQqqWeights(restored);
        ASSERT_EQ(codes.shape, (Shape{n, k}));
        ASSERT_EQ(codes.data_type, TensorProto::UINT8);
        ASSERT_EQ(weights.shape, codes.shape);
        ASSERT_EQ(weights.data_type, TensorProto::FLOAT);
        for (size_t col = 0; col < static_cast<size_t>(n); ++col)
          for (size_t row = 0; row < static_cast<size_t>(k); ++row) {
            const int code = Expected(col, row);
            EXPECT_EQ(codes.bytes()[col * static_cast<size_t>(k) + row], code);
            const float integer =
                grouped ? std::clamp(std::nearbyint(float(code - 8) * GroupScale(row / 128, col)),
                                     -128.0f, 127.0f)
                        : float((code >= 8 ? code - 16 : code) * 16);
            EXPECT_EQ(At<float>(weights, col * static_cast<size_t>(k) + row),
                      integer * ChannelScale(col));
          }
        EXPECT_EQ(PackQqqWeights(Metadata(checkpoint), codes, checkpoint.tensors.at("s_channel"),
                                 checkpoint.tensors.at("s_group"))
                      .raw_data(),
                  encoded.raw_data());
      }
}

TEST(QqqWeights, TiesSaturationZeroGroupAndFenvIndependentFloatRounding) {
  auto checkpoint = Fixture(256, 64, 128);
  auto channel = Values("s_channel", TensorProto::FLOAT, {1, 64},
                        std::vector<float>(64, std::bit_cast<float>(0x3f800001u)));
  Tensor codes("codes", TensorProto::UINT8, {64, 256}, std::vector<uint8_t>(64 * 256));
  for (size_t n = 0; n < 64; ++n)
    for (size_t k = 0; k < 256; ++k)
      codes.mutable_bytes()[n * 256 + k] = static_cast<uint8_t>(k % 16);
  const int saved = std::fegetround();
  for (float scale : {0.0f, -0.0f, 0x1p-24f, 0.5f, 1.5f, 32.0f, 65504.0f}) {
    const auto group = MakeFloat16Tensor("s_group", {2, 64}, std::vector<float>(128, scale));
    std::fesetround(FE_TONEAREST);
    const auto baseline = PackQqqWeights(Metadata(checkpoint), codes, channel, group);
    const auto expected = ReconstructQqqWeights(baseline);
    for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
      std::fesetround(mode);
      const auto encoded = PackQqqWeights(Metadata(checkpoint), codes, channel, group);
      EXPECT_EQ(encoded.raw_data(), baseline.raw_data());
      EqualTensor(ReconstructQqqWeights(encoded), expected);
    }
    std::fesetround(FE_TONEAREST);
    if (scale == 0.5f) {
      EXPECT_EQ(At<float>(expected, 9), 0);
      EXPECT_EQ(At<float>(expected, 11), 2 * std::bit_cast<float>(0x3f800001u));
      EXPECT_EQ(At<float>(expected, 7), 0);
      EXPECT_EQ(At<float>(expected, 5), -2 * std::bit_cast<float>(0x3f800001u));
    }
    if (scale == 1.5f) {
      EXPECT_EQ(std::bit_cast<uint32_t>(At<float>(expected, 10)), 0x40400002u);
    }
    if (scale == 32.0f) {
      EXPECT_EQ(At<float>(expected, 0), -128 * std::bit_cast<float>(0x3f800001u));
      EXPECT_EQ(At<float>(expected, 15), 127 * std::bit_cast<float>(0x3f800001u));
    }
  }
  checkpoint = Fixture(64, 64, -1);
  codes = Tensor("codes", TensorProto::UINT8, {64, 64}, std::vector<uint8_t>(64 * 64, 3));
  const auto group = checkpoint.tensors.at("s_group");
  for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
    std::fesetround(mode);
    const auto encoded = PackQqqWeights(Metadata(checkpoint), codes, channel, group);
    EXPECT_EQ(std::bit_cast<uint32_t>(At<float>(ReconstructQqqWeights(encoded), 0)), 0x42400002u);
  }
  std::fesetround(saved);
  const auto tiny_channel =
      Values("s_channel", TensorProto::FLOAT, {1, 64},
             std::vector<float>(64, std::numeric_limits<float>::denorm_min()));
  const auto tiny = PackQqqWeights(Metadata(checkpoint), codes, tiny_channel, group);
  EXPECT_EQ(std::bit_cast<uint32_t>(At<float>(ReconstructQqqWeights(tiny), 0)), 48u);
  checkpoint = Fixture(256, 64, 128);
  const auto huge_channel = Values("s_channel", TensorProto::FLOAT, {1, 64},
                                   std::vector<float>(64, std::numeric_limits<float>::max()));
  codes = Tensor("codes", TensorProto::UINT8, {64, 256}, std::vector<uint8_t>(64 * 256, 8));
  const auto zero =
      PackQqqWeights(Metadata(checkpoint), codes, huge_channel, checkpoint.tensors.at("s_group"));
  EXPECT_EQ(At<float>(ReconstructQqqWeights(zero), 0), 0);
}

TEST(QqqWeights, CatalogueDispatchAndCanonicalSchema) {
  const auto encoded = ImportQqqWeights(Fixture(128, 64, 128));
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(94);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  auto referenced = encoded;
  referenced.mutable_struct_type()->clear_structure();
  referenced.mutable_struct_type()->clear_name();
  referenced.mutable_struct_type()->set_type_ref(94);
  EXPECT_EQ(ExportQqqWeights(referenced, size_t{1} << 30, catalogue).group_size, 128);
  EqualTensor(DequantizeTensor(referenced, catalogue), ReconstructQqqWeights(encoded));
  for (int field = 0; field < 8; ++field) {
    auto bad = encoded;
    bad.mutable_struct_type()->mutable_structure()->mutable_field(field)->set_name("wrong");
    EXPECT_ANY_THROW(ExportQqqWeights(bad));
  }
  auto bad = encoded;
  bad.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(0)
      ->mutable_constant()
      ->set_data_type(TensorProto::UINT64);
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  bad = encoded;
  bad.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(4)
      ->mutable_constant()
      ->set_raw_data("future");
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  bad = encoded;
  bad.mutable_struct_type()->set_name("onnx_light.qqq_weights.v2");
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  bad = encoded;
  bad.mutable_struct_type()->mutable_structure()->add_field();
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  bad = encoded;
  bad.mutable_logical_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(
      1);
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  bad = encoded;
  bad.set_parameter_ref("shared");
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  bad = encoded;
  bad.mutable_raw_data()->resize(bad.raw_data().size() - 1);
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  bad = encoded;
  bad.mutable_raw_data()->resize(bad.raw_data().size() + 1);
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  bad = encoded;
  const size_t channel_offset = 128 * 64 / 2;
  std::fill(bad.mutable_raw_data()->data() + channel_offset,
            bad.mutable_raw_data()->data() + channel_offset + 4, uint8_t{0});
  bad.mutable_raw_data()->data()[channel_offset + 2] = 0x80;
  bad.mutable_raw_data()->data()[channel_offset + 3] = 0xbf;
  EXPECT_ANY_THROW(ExportQqqWeights(bad));
  EXPECT_ANY_THROW(UnpackQqqCodes(bad));
  EXPECT_ANY_THROW(ReconstructQqqWeights(bad));
  const auto portable = QuantizeTensor(Tensor::FromFloat("weight", {2}, {0, 1}),
                                       MakeQuantizationPlan(QuantizationFormat::kGptq, 2));
  EXPECT_ANY_THROW(ExportQqqWeights(portable.Encoded()));
}

TEST(QqqWeights, RejectsInvalidGeometryTensorsScalesAndLimits) {
  const auto good = Fixture(256, 64, 128);
  const auto encoded = ImportQqqWeights(good);
  constexpr size_t limit = 256 * 64 * sizeof(float);
  EXPECT_NO_THROW(ImportQqqWeights(good, limit));
  EXPECT_ANY_THROW(ImportQqqWeights(good, limit - 1));
  EXPECT_ANY_THROW(ImportQqqWeights(good, 0));
  EXPECT_ANY_THROW(ExportQqqWeights(encoded, limit - 1));
  EXPECT_ANY_THROW(UnpackQqqCodes(encoded, limit - 1));
  EXPECT_ANY_THROW(ReconstructQqqWeights(encoded, limit - 1));
  for (int64_t dim : {int64_t{-1}, int64_t{0}, int64_t{63}, int64_t{65},
                      std::numeric_limits<int64_t>::max() - 63}) {
    auto bad = good;
    bad.in_features = dim;
    EXPECT_ANY_THROW(ImportQqqWeights(bad, std::numeric_limits<size_t>::max()));
    bad = good;
    bad.out_features = dim;
    EXPECT_ANY_THROW(ImportQqqWeights(bad, std::numeric_limits<size_t>::max()));
  }
  for (int64_t group : {-2, 0, 64, 256}) {
    auto bad = good;
    bad.group_size = group;
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
  }
  for (int64_t k : {64, 192}) {
    auto bad = good;
    bad.in_features = k;
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
  }
  for (uint32_t bits : {0u, 3u, 8u, std::numeric_limits<uint32_t>::max()}) {
    auto bad = good;
    bad.bits = bits;
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
  }
  auto bad = good;
  bad.upstream_revision = "future";
  EXPECT_ANY_THROW(ImportQqqWeights(bad));
  for (const std::string name : {"B", "s_channel", "s_group"}) {
    bad = good;
    bad.tensors.erase(name);
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
    bad = good;
    bad.tensors.at(name).shape = {1};
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
    bad = good;
    bad.tensors.at(name).data_type = TensorProto::UINT8;
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
    bad = good;
    bad.tensors.at(name) = Tensor(name, good.tensors.at(name).data_type,
                                  good.tensors.at(name).shape, std::vector<uint8_t>{1});
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
  }
  bad = good;
  bad.tensors.emplace("bias", Tensor{});
  EXPECT_ANY_THROW(ImportQqqWeights(bad));
  for (float scale : {-1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max()}) {
    bad = good;
    std::memcpy(bad.tensors.at("s_channel").mutable_bytes(), &scale, sizeof(scale));
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
  }

  for (uint16_t scale : {uint16_t{0xbc00}, uint16_t{0x7c00}, uint16_t{0x7e00}}) {
    bad = good;
    std::memcpy(bad.tensors.at("s_group").mutable_bytes(), &scale, sizeof(scale));
    EXPECT_ANY_THROW(ImportQqqWeights(bad));
  }
  auto codes = UnpackQqqCodes(encoded);
  codes.mutable_bytes()[0] = 16;
  EXPECT_ANY_THROW(PackQqqWeights(Metadata(good), codes, good.tensors.at("s_channel"),
                                  good.tensors.at("s_group")));
  codes.mutable_bytes()[0] = 0;
  EXPECT_ANY_THROW(
      PackQqqWeights(good, codes, good.tensors.at("s_channel"), good.tensors.at("s_group")));
  EXPECT_ANY_THROW(PackQqqWeights(Metadata(good), codes, good.tensors.at("s_channel"),
                                  good.tensors.at("s_group"), limit - 1));
  codes.shape = {256, 64};
  EXPECT_ANY_THROW(PackQqqWeights(Metadata(good), codes, good.tensors.at("s_channel"),
                                  good.tensors.at("s_group")));
  const auto empty_codes = Tensor("codes", TensorProto::UINT8, {64, 256}, std::vector<uint8_t>{});
  EXPECT_ANY_THROW(PackQqqWeights(Metadata(good), empty_codes, good.tensors.at("s_channel"),
                                  good.tensors.at("s_group")));
}

TEST(QqqWeights, ZeroChannelScalesPreserveBytesAndSignedZeroWeights) {
  for (bool grouped : {false, true}) {
    const auto checkpoint = Fixture(grouped ? 256 : 64, 64, grouped ? 128 : -1);
    const auto metadata = Metadata(checkpoint);
    const int64_t k = metadata.in_features;
    Tensor codes("codes", TensorProto::UINT8, {64, k},
                 std::vector<uint8_t>(static_cast<size_t>(64 * k)));
    for (size_t i = 0; i < codes.size_bytes(); ++i)
      codes.mutable_bytes()[i] = static_cast<uint8_t>(i % 16);
    const auto group = MakeFloat16Tensor("s_group", grouped ? Shape{2, 64} : Shape{0},
                                         std::vector<float>(grouped ? 128 : 0, 1));
    for (float zero : {0.0f, -0.0f}) {
      const auto channel =
          Values("s_channel", TensorProto::FLOAT, {1, 64}, std::vector<float>(64, zero));
      const auto encoded = PackQqqWeights(metadata, codes, channel, group);
      const auto exported = ExportQqqWeights(encoded);
      EqualTensor(exported.tensors.at("s_channel"), channel);
      EqualTensor(exported.tensors.at("s_group"), group);
      EXPECT_EQ(ImportQqqWeights(exported).raw_data(), encoded.raw_data());
      EqualTensor(UnpackQqqCodes(encoded), codes);
      const auto weights = ReconstructQqqWeights(encoded);
      for (size_t i = 0; i < codes.size_bytes(); ++i) {
        const bool negative = grouped ? codes.bytes()[i] < 8 : codes.bytes()[i] >= 8;
        const uint32_t expected = negative != std::signbit(zero) ? 0x80000000u : 0u;
        EXPECT_EQ(std::bit_cast<uint32_t>(At<float>(weights, i)), expected);
      }
    }
  }
}
