// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

uint8_t Expected(size_t n, size_t k) {
  return static_cast<uint8_t>((n * 7 + k * 3 + k / 16 + k / 64) % 16);
}

Tensor Words(const char *name, int32_t type, Shape shape, const std::vector<uint32_t> &words) {
  const size_t width = type == TensorProto::INT16 ? 2 : 4;
  std::vector<uint8_t> bytes(words.size() * width);
  for (size_t i = 0; i < words.size(); ++i) {
    if (width == 4)
      std::memcpy(bytes.data() + i * width, &words[i], width);
    else {
      const auto word = static_cast<uint16_t>(words[i]);
      std::memcpy(bytes.data() + i * width, &word, width);
    }
  }
  return Tensor(name, type, shape, std::move(bytes));
}

AwqWeights Fixture(const std::string &variant, int64_t group_size = 64) {
  AwqWeights result;
  result.variant = variant;
  result.in_features = 128;
  result.out_features = 8;
  result.group_size = group_size;
  const size_t group = group_size == -1 ? 128 : static_cast<size_t>(group_size);
  const size_t groups = 128 / group;
  const size_t multiplier = group >= 128 ? 1 : 128 / group;
  const size_t padded = multiplier * 8;
  if (variant == "gemm") {
    // GPTQModel pack_awq_columns gathers AWQ_ORDER before shifting each nibble.
    constexpr size_t order[] = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<uint32_t> words(128), zeros(groups);
    for (size_t k = 0; k < 128; ++k)
      for (size_t j = 0; j < 8; ++j)
        words[k] |= uint32_t(Expected(order[j], k)) << (4 * j);
    for (size_t g = 0; g < groups; ++g)
      for (size_t j = 0; j < 8; ++j)
        zeros[g] |= uint32_t((order[j] + g) % 16) << (4 * j);
    result.tensors.emplace("qweight", Words("qweight", TensorProto::INT32, {128, 1}, words));
    result.tensors.emplace(
        "qzeros", Words("qzeros", TensorProto::INT32, {static_cast<int64_t>(groups), 1}, zeros));
    std::vector<float> scales(groups * 8);
    for (size_t g = 0; g < groups; ++g)
      for (size_t n = 0; n < 8; ++n)
        scales[g * 8 + n] = static_cast<float>(g + 1) * 0.25f;
    result.tensors.emplace("scales",
                           MakeFloat16Tensor("scales", {static_cast<int64_t>(groups), 8}, scales));
  } else if (variant == "gemv") {
    std::vector<uint32_t> words(8 * 16), zeros(8 * padded / 8, 0xfedcba98u);
    for (size_t n = 0; n < 8; ++n) {
      for (size_t k = 0; k < 128; ++k)
        words[n * 16 + k / 8] |= uint32_t(Expected(n, k)) << (4 * (k % 8));
      for (size_t g = 0; g < groups; ++g) {
        auto &word = zeros[n * (padded / 8) + g / 8];
        word = (word & ~(15u << (4 * (g % 8)))) | (uint32_t((n + g) % 16) << (4 * (g % 8)));
      }
    }
    result.tensors.emplace("qweight", Words("qweight", TensorProto::INT32, {8, 16}, words));
    result.tensors.emplace("qzeros", Words("qzeros", TensorProto::INT32,
                                           {8, static_cast<int64_t>(padded / 8)}, zeros));
    std::vector<float> scales(8 * padded, -3);
    for (size_t n = 0; n < 8; ++n)
      for (size_t g = 0; g < groups; ++g)
        scales[n * padded + g] = static_cast<float>(g + 1) * 0.25f;
    result.tensors.emplace("scales",
                           MakeFloat16Tensor("scales", {8, static_cast<int64_t>(padded)}, scales));
  } else {
    // Source forward reshape/permute chain, not the decoder's index inverse.
    std::vector<uint8_t> permuted(8 * 128);
    for (size_t n = 0; n < 8; ++n)
      for (size_t block = 0; block < 4; ++block) {
        uint8_t first[32];
        for (size_t a = 0; a < 4; ++a)
          for (size_t b = 0; b < 4; ++b)
            for (size_t c = 0; c < 2; ++c)
              first[(a * 4 + b) * 2 + c] = Expected(n, block * 32 + (b * 4 + a) * 2 + c);
        for (size_t a = 0; a < 4; ++a)
          for (size_t b = 0; b < 2; ++b)
            for (size_t c = 0; c < 4; ++c)
              permuted[n * 128 + block * 32 + a * 8 + b * 4 + c] = first[a * 8 + c * 2 + b];
      }
    std::vector<uint32_t> words(2 * 128);
    for (size_t nb = 0; nb < 2; ++nb)
      for (size_t kb = 0; kb < 2; ++kb)
        for (size_t j = 0; j < 64; ++j)
          for (size_t lane = 0; lane < 4; ++lane) {
            const size_t flat = j * 4 + lane;
            words[nb * 128 + kb * 64 + j] |=
                uint32_t(permuted[(nb * 4 + flat / 64) * 128 + kb * 64 + flat % 64]) << (4 * lane);
          }
    result.tensors.emplace("qweight", Words("qweight", TensorProto::INT16, {2, 128}, words));
    std::vector<float> scales(padded * 8, -3), zeros(padded * 8, 7);
    for (size_t g = 0; g < groups; ++g)
      for (size_t n = 0; n < 8; ++n) {
        scales[g * 8 + n] = static_cast<float>(g + 1) * 0.25f;
        zeros[g * 8 + n] = -static_cast<float>((n + g) % 16) * scales[g * 8 + n];
      }
    const char *zero_name = variant == "llm-awq" ? "scaled_zeros" : "qzeros";
    result.tensors.emplace(zero_name,
                           MakeFloat16Tensor(zero_name, {static_cast<int64_t>(padded), 8}, zeros));
    result.tensors.emplace("scales",
                           MakeFloat16Tensor("scales", {static_cast<int64_t>(padded), 8}, scales));
  }
  return result;
}

} // namespace

TEST(AwqWeights, SourceDerivedLayoutsRoundTripAndReconstruct) {
  for (const std::string variant : {"gemm", "gemv", "gemv_fast", "llm-awq"})
    for (int64_t group_size : {-1, 64, 128}) {
      const auto checkpoint = Fixture(variant, group_size);
      auto encoded = ImportAwqWeights(checkpoint);
      EncodedValueProto restored;
      restored.ParseFromString(encoded.SerializeAsString());
      const auto exported = ExportAwqWeights(restored);
      EXPECT_EQ(exported.variant, variant);
      EXPECT_EQ(exported.group_size, group_size);
      EXPECT_EQ(exported.upstream_revision, checkpoint.upstream_revision);
      for (const auto &[name, tensor] : checkpoint.tensors) {
        const auto &actual = exported.tensors.at(name);
        EXPECT_EQ(actual.data_type, tensor.data_type);
        EXPECT_EQ(actual.shape, tensor.shape);
        ASSERT_EQ(actual.size_bytes(), tensor.size_bytes());
        EXPECT_EQ(std::memcmp(actual.bytes(), tensor.bytes(), tensor.size_bytes()), 0);
      }
      const auto codes = UnpackAwqCodes(restored);
      const auto weights = ReconstructAwqWeights(restored);
      ASSERT_EQ(codes.shape, (Shape{8, 128}));
      ASSERT_EQ(codes.data_type, TensorProto::UINT8);
      ASSERT_EQ(weights.shape, codes.shape);
      ASSERT_EQ(weights.data_type, TensorProto::FLOAT);
      for (size_t n = 0; n < 8; ++n)
        for (size_t k = 0; k < 128; ++k) {
          EXPECT_EQ(codes.bytes()[n * 128 + k], Expected(n, k)) << variant;
          const size_t g = k / (group_size == -1 ? 128 : group_size);
          const float expected = (static_cast<float>(Expected(n, k)) - float((n + g) % 16)) *
                                 (static_cast<float>(g + 1) * 0.25f);
          float actual;
          std::memcpy(&actual, weights.bytes() + (n * 128 + k) * 4, 4);
          EXPECT_EQ(actual, expected) << variant << " n=" << n << " k=" << k;
        }
      const auto generic = DequantizeTensor(restored);
      EXPECT_EQ(std::memcmp(generic.bytes(), weights.bytes(), weights.size_bytes()), 0);
    }
}

TEST(AwqWeights, SourceFixtureContainsExpectedPackedWords) {
  const auto gemm = Fixture("gemm");
  uint32_t word;
  std::memcpy(&word, gemm.tensors.at("qweight").bytes(), 4);
  EXPECT_EQ(word, 0x1357ace0u);
  std::memcpy(&word, gemm.tensors.at("qzeros").bytes(), 4);
  EXPECT_EQ(word, 0x75316420u);
  const auto gemv = Fixture("gemv");
  std::memcpy(&word, gemv.tensors.at("qweight").bytes(), 4);
  EXPECT_EQ(word, 0x52fc9630u);
  const auto fast = Fixture("gemv_fast");
  uint16_t half;
  std::memcpy(&half, fast.tensors.at("qweight").bytes(), 2);
  EXPECT_EQ(half, 0x9180u);
}

TEST(AwqWeights, RejectsInvalidMetadataPayloadAndSchema) {
  auto checkpoint = Fixture("gemm");
  auto encoded = ImportAwqWeights(checkpoint);
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint, 0));
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint, 4095));
  EXPECT_NO_THROW(ImportAwqWeights(checkpoint, 4096));
  EXPECT_ANY_THROW(ExportAwqWeights(encoded, 4095));
  checkpoint.bits = 8;
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemm");
  checkpoint.group_axis = 0;
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemm");
  checkpoint.group_size = 0;
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemm");
  checkpoint.in_features = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint, std::numeric_limits<size_t>::max()));
  checkpoint = Fixture("gemm");
  checkpoint.upstream_revision = "future";
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemm");
  checkpoint.tensors.at("qweight").shape = {1, 128};
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemm");
  checkpoint.tensors.at("qweight").data_type = TensorProto::UINT32;
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemm");
  checkpoint.tensors.at("qzeros") =
      Tensor("qzeros", TensorProto::INT32, {2, 1}, std::vector<uint8_t>{0});
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemm");
  const uint16_t infinity = 0x7c00;
  std::memcpy(checkpoint.tensors.at("scales").mutable_bytes(), &infinity, 2);
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  encoded.mutable_raw_data()->push_back(0);
  EXPECT_ANY_THROW(UnpackAwqCodes(encoded));
  encoded = ImportAwqWeights(Fixture("gemm"));
  encoded.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(5)
      ->mutable_constant()
      ->ref_int64_data()[0] = 0;
  EXPECT_ANY_THROW(ExportAwqWeights(encoded));
  encoded = ImportAwqWeights(Fixture("gemm"));
  encoded.mutable_struct_type()->mutable_structure()->mutable_field(8)->set_name("scaled_zeros");
  EXPECT_ANY_THROW(ReconstructAwqWeights(encoded));
}

TEST(AwqWeights, ResolvesCatalogueAndRejectsRenamedFastZeroTensor) {
  auto encoded = ImportAwqWeights(Fixture("gemv_fast", 16));
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(91);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  auto referenced = encoded;
  referenced.mutable_struct_type()->clear_structure();
  referenced.mutable_struct_type()->clear_name();
  referenced.mutable_struct_type()->set_type_ref(91);
  EXPECT_EQ(ExportAwqWeights(referenced, size_t{1} << 30, catalogue).variant, "gemv_fast");
  const auto actual = DequantizeTensor(referenced, catalogue);
  const auto expected = ReconstructAwqWeights(encoded);
  EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  auto checkpoint = Fixture("llm-awq");
  checkpoint.tensors.emplace("qzeros", std::move(checkpoint.tensors.at("scaled_zeros")));
  checkpoint.tensors.erase("scaled_zeros");
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
}

TEST(AwqWeights, PreservesBfloat16AndChecksReconstructionOverflow) {
  auto checkpoint = Fixture("gemm");
  auto &scales = checkpoint.tensors.at("scales");
  for (size_t i = 0; i < scales.size_bytes() / 2; ++i) {
    uint16_t half;
    std::memcpy(&half, scales.bytes() + i * 2, 2);
    const uint16_t bits = FloatToBfloat16Bits(Float16BitsToFloat(half));
    std::memcpy(scales.mutable_bytes() + i * 2, &bits, 2);
  }
  scales.data_type = TensorProto::BFLOAT16;
  const auto encoded = ImportAwqWeights(checkpoint);
  const auto exported = ExportAwqWeights(encoded);
  EXPECT_EQ(exported.tensors.at("scales").data_type, TensorProto::BFLOAT16);
  EXPECT_EQ(std::memcmp(exported.tensors.at("scales").bytes(), scales.bytes(), scales.size_bytes()),
            0);
  const uint16_t largest = 0x7f7f;
  std::memcpy(scales.mutable_bytes(), &largest, 2);
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  auto malformed = encoded;
  const size_t scale_offset = 128 * 4 + 2 * 4;
  (*malformed.mutable_raw_data())[scale_offset] = static_cast<char>(largest);
  (*malformed.mutable_raw_data())[scale_offset + 1] = static_cast<char>(largest >> 8);
  EXPECT_ANY_THROW(UnpackAwqCodes(malformed));
  EXPECT_ANY_THROW(ExportAwqWeights(malformed));
  EXPECT_ANY_THROW(ReconstructAwqWeights(malformed));
}

TEST(AwqWeights, ValidatesFloat16ParametersWithoutReconstructionPass) {
  for (const std::string variant : {"gemm", "gemv", "gemv_fast", "llm-awq"}) {
    SCOPED_TRACE(variant);
    auto checkpoint = Fixture(variant);
    auto &scales = checkpoint.tensors.at("scales");
    const uint16_t largest = 0x7bff;
    for (size_t i = 0; i < scales.size_bytes(); i += 2)
      std::memcpy(scales.mutable_bytes() + i, &largest, 2);
    const char *zero_name = variant == "llm-awq" ? "scaled_zeros" : "qzeros";
    if (variant == "gemv_fast" || variant == "llm-awq") {
      auto &zeros = checkpoint.tensors.at(zero_name);
      const uint16_t smallest = 0xfbff;
      for (size_t i = 0; i < zeros.size_bytes(); i += 2)
        std::memcpy(zeros.mutable_bytes() + i, &smallest, 2);
    }

    auto encoded = ImportAwqWeights(checkpoint);
    EXPECT_NO_THROW(ExportAwqWeights(encoded));
    EXPECT_NO_THROW(UnpackAwqCodes(encoded));
    const Tensor reconstructed = ReconstructAwqWeights(encoded);
    for (size_t i = 0; i < reconstructed.size_bytes(); i += sizeof(float)) {
      float value;
      std::memcpy(&value, reconstructed.bytes() + i, sizeof(float));
      EXPECT_TRUE(std::isfinite(value));
    }

    if (variant == "gemv_fast" || variant == "llm-awq") {
      auto invalid_zero = encoded;
      const size_t zero_offset = checkpoint.tensors.at("qweight").size_bytes();
      (*invalid_zero.mutable_raw_data())[zero_offset] = 0;
      (*invalid_zero.mutable_raw_data())[zero_offset + 1] = '\x7c';
      EXPECT_THROW(ExportAwqWeights(invalid_zero), std::invalid_argument);
      EXPECT_THROW(UnpackAwqCodes(invalid_zero), std::invalid_argument);
      EXPECT_THROW(ReconstructAwqWeights(invalid_zero), std::invalid_argument);
    }

    const size_t scale_offset = checkpoint.tensors.at("qweight").size_bytes() +
                                checkpoint.tensors.at(zero_name).size_bytes();
    (*encoded.mutable_raw_data())[scale_offset] = 0;
    (*encoded.mutable_raw_data())[scale_offset + 1] = '\x7c';
    EXPECT_THROW(ExportAwqWeights(encoded), std::invalid_argument);
    EXPECT_THROW(UnpackAwqCodes(encoded), std::invalid_argument);
    EXPECT_THROW(ReconstructAwqWeights(encoded), std::invalid_argument);
  }
}

TEST(AwqWeights, SupportsSmallGroupPaddingAndRejectsImpossiblePackingDimensions) {
  for (const std::string variant : {"gemm", "gemv", "gemv_fast", "llm-awq"})
    for (int64_t group : {16, 32}) {
      const auto encoded = ImportAwqWeights(Fixture(variant, group));
      const auto codes = UnpackAwqCodes(encoded);
      for (size_t n = 0; n < 8; ++n)
        for (size_t k = 0; k < 128; ++k)
          EXPECT_EQ(codes.bytes()[n * 128 + k], Expected(n, k));
    }
  auto checkpoint = Fixture("gemm");
  checkpoint.out_features = 7;
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemv_fast");
  checkpoint.out_features = 7;
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
  checkpoint = Fixture("gemv_fast");
  checkpoint.in_features = 96;
  checkpoint.group_size = 32;
  EXPECT_ANY_THROW(ImportAwqWeights(checkpoint));
}

TEST(AwqWeights, PreservesGroup256AndEquivalentFullWidthSentinel) {
  for (const std::string variant : {"gemm", "gemv", "gemv_fast", "llm-awq"})
    for (int64_t grouping : {256, -1}) {
      AwqWeights checkpoint;
      checkpoint.variant = variant;
      checkpoint.in_features = 256;
      checkpoint.out_features = 8;
      checkpoint.group_size = grouping;
      if (variant == "gemm") {
        checkpoint.tensors.emplace("qweight", Words("qweight", TensorProto::INT32, {256, 1},
                                                    std::vector<uint32_t>(256, 0xffffffffu)));
        checkpoint.tensors.emplace("qzeros",
                                   Words("qzeros", TensorProto::INT32, {1, 1}, {0x22222222u}));
        checkpoint.tensors.emplace(
            "scales", MakeFloat16Tensor("scales", {1, 8}, std::vector<float>(8, 1.25f)));
      } else if (variant == "gemv") {
        checkpoint.tensors.emplace("qweight", Words("qweight", TensorProto::INT32, {8, 32},
                                                    std::vector<uint32_t>(256, 0xffffffffu)));
        checkpoint.tensors.emplace("qzeros", Words("qzeros", TensorProto::INT32, {8, 1},
                                                   std::vector<uint32_t>(8, 0x22222222u)));
        checkpoint.tensors.emplace(
            "scales", MakeFloat16Tensor("scales", {8, 8}, std::vector<float>(64, 1.25f)));
      } else {
        checkpoint.tensors.emplace("qweight", Words("qweight", TensorProto::INT16, {2, 256},
                                                    std::vector<uint32_t>(512, 0xffffu)));
        const char *name = variant == "llm-awq" ? "scaled_zeros" : "qzeros";
        checkpoint.tensors.emplace(name,
                                   MakeFloat16Tensor(name, {8, 8}, std::vector<float>(64, -2.5f)));
        checkpoint.tensors.emplace(
            "scales", MakeFloat16Tensor("scales", {8, 8}, std::vector<float>(64, 1.25f)));
      }
      const auto encoded = ImportAwqWeights(checkpoint);
      EXPECT_EQ(ExportAwqWeights(encoded).group_size, grouping);
      const auto codes = UnpackAwqCodes(encoded);
      const auto weights = ReconstructAwqWeights(encoded);
      for (size_t i = 0; i < 8 * 256; ++i) {
        EXPECT_EQ(codes.bytes()[i], 15);
        float actual;
        std::memcpy(&actual, weights.bytes() + i * 4, 4);
        EXPECT_EQ(actual, 16.25f);
      }
    }
}
