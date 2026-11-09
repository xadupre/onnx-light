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

Exl3Weights Fixture(uint32_t bits = 3, uint32_t book = 0) {
  Exl3Weights result;
  result.tensors.emplace("trellis", Tensor("trellis", TensorProto::INT16, {8, 8, 16 * bits},
                                           std::vector<uint8_t>(2048 * bits, 0)));
  result.tensors.emplace("suh", MakeFloat16Tensor("suh", {128}, std::vector<float>(128, -0.5f)));
  result.tensors.emplace("svh", MakeFloat16Tensor("svh", {128}, std::vector<float>(128, 2)));
  if (book) {
    const uint32_t marker = book == 1 ? 0xcbac1fedu : 0x83dcd12du;
    std::vector<uint8_t> bytes(4);
    for (size_t i = 0; i < 4; ++i)
      bytes[i] = static_cast<uint8_t>(marker >> (8 * i));
    const char *name = book == 1 ? "mcg" : "mul1";
    result.tensors.emplace(name, Tensor(name, TensorProto::INT32, {}, std::move(bytes)));
  }
  return result;
}

} // namespace

TEST(Exl3Weights, NativeRatesAndCodebooksRoundTrip) {
  // State zero from GPTQModel's binary16 decoder: 3inst, mcg, mul1 respectively.
  constexpr float decoded[] = {0.76806640625f, 1.84375f, -3.453125f};
  for (uint32_t bits = 1; bits <= 8; ++bits)
    for (uint32_t book = 0; book <= 2; ++book) {
      auto checkpoint = Fixture(bits, book);
      auto encoded = ImportExl3Weights(checkpoint);
      EncodedValueProto restored;
      restored.ParseFromString(encoded.SerializeAsString());
      const auto exported = ExportExl3Weights(restored);
      ASSERT_EQ(exported.tensors.size(), checkpoint.tensors.size());
      for (const auto &[name, tensor] : checkpoint.tensors) {
        const auto &actual = exported.tensors.at(name);
        EXPECT_EQ(actual.data_type, tensor.data_type);
        EXPECT_EQ(actual.shape, tensor.shape);
        ASSERT_EQ(actual.size_bytes(), tensor.size_bytes());
        EXPECT_EQ(std::memcmp(actual.bytes(), tensor.bytes(), tensor.size_bytes()), 0);
      }
      const auto actual = ReconstructExl3Weights(restored);
      ASSERT_EQ(actual.shape, (Shape{128, 128}));
      const auto *values = reinterpret_cast<const float *>(actual.bytes());
      EXPECT_NEAR(values[0], -decoded[book] * 128, 0.0001f);
      for (size_t i = 1; i < 128 * 128; ++i)
        EXPECT_EQ(values[i], 0);
      const auto generic = DequantizeTensor(restored);
      EXPECT_EQ(std::memcmp(generic.bytes(), actual.bytes(), actual.size_bytes()), 0);
      const auto proto = DequantizeTensorProto(restored);
      EXPECT_EQ(proto.data_type(), TensorProto::FLOAT);
    }
}

TEST(Exl3Weights, RejectsMalformedInputsBeforeAllocation) {
  auto checkpoint = Fixture();
  EXPECT_ANY_THROW(ImportExl3Weights(checkpoint, 0));
  EXPECT_ANY_THROW(ImportExl3Weights(checkpoint, 65535));
  auto encoded = ImportExl3Weights(checkpoint);
  EXPECT_ANY_THROW(ExportExl3Weights(encoded, 65535));
  EXPECT_ANY_THROW(ReconstructExl3Weights(encoded, 65535));
  checkpoint.upstream_revision = "future";
  EXPECT_ANY_THROW(ImportExl3Weights(checkpoint));
  checkpoint = Fixture();
  checkpoint.tensors.at("trellis").shape[0] = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ImportExl3Weights(checkpoint));
  checkpoint = Fixture();
  checkpoint.tensors.at("trellis").shape[0] = 4;
  EXPECT_ANY_THROW(ImportExl3Weights(checkpoint));
  checkpoint = Fixture();
  checkpoint.tensors.at("trellis").shape[2] = 17;
  EXPECT_ANY_THROW(ImportExl3Weights(checkpoint));
  checkpoint = Fixture();
  checkpoint.tensors.at("suh") = MakeFloat16Tensor(
      "suh", {128}, std::vector<float>(128, std::numeric_limits<float>::infinity()));
  EXPECT_ANY_THROW(ImportExl3Weights(checkpoint));
  checkpoint = Fixture(3, 1);
  checkpoint.tensors.at("mcg").mutable_bytes()[0] ^= 1;
  EXPECT_ANY_THROW(ImportExl3Weights(checkpoint));
  encoded.mutable_raw_data()->resize(encoded.raw_data().size() - 1);
  EXPECT_ANY_THROW(ExportExl3Weights(encoded));
  EXPECT_ANY_THROW(DequantizeTensor(encoded));
  encoded = ImportExl3Weights(Fixture());
  encoded.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(0)
      ->mutable_constant()
      ->ref_int64_data()[0] = 0;
  EXPECT_ANY_THROW(ExportExl3Weights(encoded));
  encoded = ImportExl3Weights(Fixture());
  encoded.mutable_logical_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(0)
      ->set_dim_value(std::numeric_limits<int64_t>::max() - 127);
  EXPECT_ANY_THROW(ReconstructExl3Weights(encoded, std::numeric_limits<size_t>::max()));
}

TEST(Exl3Weights, ResolvesSharedTypeCatalogueWithoutChangingPayload) {
  auto encoded = ImportExl3Weights(Fixture(5, 2));
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(86);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  auto referenced = encoded;
  referenced.mutable_struct_type()->clear_structure();
  referenced.mutable_struct_type()->clear_name();
  referenced.mutable_struct_type()->set_type_ref(86);
  EXPECT_EQ(ExportExl3Weights(referenced, size_t{1} << 30, catalogue).tensors.size(), 4);
  const auto states = UnpackExl3States(referenced, size_t{1} << 30, catalogue);
  EXPECT_EQ(states.shape, (Shape{8, 8, 256}));
  const auto actual = DequantizeTensor(referenced, catalogue);
  const auto expected = ReconstructExl3Weights(encoded);
  EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
}
