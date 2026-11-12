// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

Tensor FloatTensor(const std::string &name, const Shape &shape, const std::vector<float> &values) {
  TensorProto proto;
  proto.set_name(name);
  proto.set_data_type(TensorProto::FLOAT);
  for (const auto dim : shape)
    proto.add_dims(dim);
  auto &raw = *proto.mutable_raw_data();
  for (const float value : values) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    for (size_t byte = 0; byte < 4; ++byte)
      raw.push_back(static_cast<char>(bits >> (byte * 8)));
  }
  return TensorFromProto(proto).ToOwned();
}

float FloatValue(const Tensor &tensor, size_t index) {
  uint32_t bits = 0;
  for (size_t byte = 0; byte < 4; ++byte)
    bits |= uint32_t(tensor.bytes()[index * 4 + byte]) << (byte * 8);
  return std::bit_cast<float>(bits);
}

BitsAndBytesWeights Fixture(bool nested = false) {
  BitsAndBytesWeights weights;
  std::vector<uint8_t> codes(17);
  for (size_t i = 0; i < codes.size(); ++i)
    codes[i] = static_cast<uint8_t>(((2 * i % 16) << 4) | ((2 * i + 1) % 16));
  codes.back() = 0x0d; // Nonzero padding must survive export, but not become a logical code.
  weights.tensors.emplace("weight",
                          Tensor("weight", TensorProto::UINT8, {17, 1}, std::move(codes)));
  const auto table = MakeQuantizationPlan(QuantizationFormat::kNf4, 16).runs[0].blocks[0].codebook;
  weights.tensors.emplace("weight_quant_map",
                          FloatTensor("map", {16}, std::vector<float>(table.begin(), table.end())));
  std::string json = R"({"quant_type":"nf4","blocksize":32,"dtype":"float16","shape":[3,11])";
  if (nested) {
    json += R"(,"nested_blocksize":256,"nested_dtype":"float32","nested_offset":2.0)";
    weights.tensors.emplace(
        "weight_absmax", Tensor("absmax", TensorProto::UINT8, {2}, std::vector<uint8_t>{0, 255}));
    weights.tensors.emplace("weight_nested_absmax", FloatTensor("max", {1}, {0.5f}));
    std::vector<float> map(256, 0);
    map.front() = -1;
    map.back() = 1;
    weights.tensors.emplace("weight_nested_quant_map", FloatTensor("nested_map", {256}, map));
  } else
    weights.tensors.emplace("weight_absmax", FloatTensor("absmax", {2}, {2.f, 0.f}));
  json += "}";
  weights.tensors.emplace("weight_quant_state",
                          Tensor("json", TensorProto::UINT8, {static_cast<int64_t>(json.size())},
                                 std::vector<uint8_t>(json.begin(), json.end())));
  return weights;
}

void SameCheckpoint(const BitsAndBytesWeights &expected, const BitsAndBytesWeights &actual) {
  ASSERT_EQ(expected.tensors.size(), actual.tensors.size());
  for (const auto &[name, tensor] : expected.tensors) {
    const auto &restored = actual.tensors.at(name);
    EXPECT_EQ(restored.data_type, tensor.data_type);
    EXPECT_EQ(restored.shape, tensor.shape);
    ASSERT_EQ(restored.size_bytes(), tensor.size_bytes());
    EXPECT_EQ(std::memcmp(restored.bytes(), tensor.bytes(), tensor.size_bytes()), 0);
  }
}

} // namespace

TEST(BitsAndBytesWeights, NativeNibblesPartialBlocksAndNestedState) {
  // get_4bit_type("nf4") at bitsandbytes 08a9956b; native flattened index order.
  constexpr float expected[] = {-1.f,
                                -0.6961928009986877f,
                                -0.5250730514526367f,
                                -0.39491748809814453f,
                                -0.28444138169288635f,
                                -0.18477343022823334f,
                                -0.09105003625154495f,
                                0.f,
                                0.07958029955625534f,
                                0.16093020141124725f,
                                0.24611230194568634f,
                                0.33791524171829224f,
                                0.44070982933044434f,
                                0.5626170039176941f,
                                0.7229568362236023f,
                                1.f};
  for (bool nested : {false, true}) {
    const auto weights = Fixture(nested);
    const auto encoded = ImportBitsAndBytesWeights(weights);
    EncodedValueProto restored;
    restored.ParseFromString(encoded.SerializeAsString());
    SameCheckpoint(weights, ExportBitsAndBytesWeights(restored));
    const auto codes = UnpackBitsAndBytesCodes(restored);
    ASSERT_EQ(codes.shape, (Shape{3, 11}));
    for (size_t i = 0; i < 33; ++i)
      EXPECT_EQ(codes.bytes()[i], i % 16);
    const auto decoded = ReconstructBitsAndBytesWeights(restored);
    for (size_t i = 0; i < 33; ++i)
      EXPECT_EQ(FloatValue(decoded, i),
                expected[i % 16] * (i < 32 ? (nested ? 1.5f : 2.f) : (nested ? 2.5f : 0.f)));
    const auto generic = DequantizeTensorProto(restored);
    EXPECT_EQ(generic.raw_data().size(), decoded.size_bytes());
    EXPECT_EQ(std::memcmp(generic.raw_data().data(), decoded.bytes(), decoded.size_bytes()), 0);
  }
}

TEST(BitsAndBytesWeights, Int8SignedCodesAndRowScales) {
  BitsAndBytesWeights weights;
  weights.tensors.emplace("weight", Tensor("weight", TensorProto::INT8, {2, 3},
                                           std::vector<uint8_t>{128, 129, 255, 0, 1, 127}));
  weights.tensors.emplace("weight_scb", FloatTensor("scb", {2}, {127.f, 254.f}));
  const auto encoded = ImportBitsAndBytesWeights(weights);
  SameCheckpoint(weights, ExportBitsAndBytesWeights(encoded));
  const auto codes = UnpackBitsAndBytesCodes(encoded);
  EXPECT_EQ(codes.data_type, TensorProto::INT8);
  EXPECT_EQ(std::memcmp(codes.bytes(), weights.tensors.at("weight").bytes(), 6), 0);
  const auto decoded = ReconstructBitsAndBytesWeights(encoded);
  const auto expected = FloatTensor("expected", {2, 3}, {-128, -127, -1, 0, 2, 254});
  EXPECT_EQ(std::memcmp(decoded.bytes(), expected.bytes(), expected.size_bytes()), 0);
}

TEST(BitsAndBytesWeights, PreservesLittleEndianProtoBytes) {
  TensorProto proto;
  proto.set_name("weight_scb");
  proto.set_data_type(TensorProto::FLOAT);
  proto.add_dims(1);
  const std::string scale("\x00\x00\xfe\x42", 4); // 127.f in canonical little-endian order.
  proto.set_raw_data(scale);
  BitsAndBytesWeights weights;
  weights.tensors.emplace("weight",
                          Tensor("weight", TensorProto::INT8, {1, 1}, std::vector<uint8_t>{1}));
  weights.tensors.emplace("weight_scb", TensorFromProto(proto));
  const auto encoded = ImportBitsAndBytesWeights(weights);
  EXPECT_EQ(encoded.raw_data(), std::string("\x01", 1) + scale);
  SameCheckpoint(weights, ExportBitsAndBytesWeights(encoded));
  for (const auto &[dtype, expected] : std::vector<std::pair<int32_t, std::string>>{
           {TensorProto::FLOAT, std::string("\x00\x00\x80\x3f", 4)},
           {TensorProto::FLOAT16, std::string("\x00\x3c", 2)},
           {TensorProto::BFLOAT16, std::string("\x80\x3f", 2)}}) {
    const auto decoded = ReconstructBitsAndBytesWeights(encoded, size_t{1} << 30, {}, dtype);
    EXPECT_EQ(std::string(reinterpret_cast<const char *>(decoded.bytes()), decoded.size_bytes()),
              expected);
    const auto generic = DequantizeTensor(encoded, {}, nullptr, dtype);
    EXPECT_EQ(std::string(reinterpret_cast<const char *>(generic.bytes()), generic.size_bytes()),
              expected);
  }
  EXPECT_EQ(DequantizeTensorProto(encoded).raw_data(), std::string("\x00\x00\x80\x3f", 4));
  for (const bool nested : {false, true}) {
    const auto checkpoint = Fixture(nested);
    const auto value = ImportBitsAndBytesWeights(checkpoint);
    SameCheckpoint(checkpoint, ExportBitsAndBytesWeights(value));
    std::string expected;
    for (const char *name : {"weight", "weight_absmax", "weight_quant_map", "weight_nested_absmax",
                             "weight_nested_quant_map", "weight_quant_state"}) {
      const auto found = checkpoint.tensors.find(name);
      if (found != checkpoint.tensors.end())
        expected.append(reinterpret_cast<const char *>(found->second.bytes()),
                        found->second.size_bytes());
    }
    EXPECT_EQ(value.raw_data(), expected);
  }
}

TEST(BitsAndBytesWeights, RejectsMalformedStateAndOverflowBeforeAllocation) {
  auto weights = Fixture();
  for (const char *name : {"weight", "weight_absmax", "weight_quant_map", "weight_quant_state"}) {
    auto invalid = Fixture();
    invalid.tensors.erase(name);
    EXPECT_ANY_THROW(ImportBitsAndBytesWeights(invalid));
  }
  weights.upstream_revision = "future";
  EXPECT_ANY_THROW(ImportBitsAndBytesWeights(weights));
  weights = Fixture();
  weights.tensors.at("weight").shape[0] = 18;
  EXPECT_ANY_THROW(ImportBitsAndBytesWeights(weights));
  weights = Fixture();
  weights.tensors.at("weight_absmax") = FloatTensor("scales", {2}, {-1.f, 0.f});
  EXPECT_ANY_THROW(ImportBitsAndBytesWeights(weights));
  weights = Fixture(true);
  weights.tensors.at("weight_nested_absmax") =
      FloatTensor("scales", {1}, {std::numeric_limits<float>::infinity()});
  EXPECT_ANY_THROW(ImportBitsAndBytesWeights(weights));
  auto encoded = ImportBitsAndBytesWeights(Fixture(true));
  for (size_t limit : {size_t{0}, size_t{131}}) {
    EXPECT_ANY_THROW(ImportBitsAndBytesWeights(Fixture(), limit));
    EXPECT_ANY_THROW(ExportBitsAndBytesWeights(encoded, limit));
    EXPECT_ANY_THROW(UnpackBitsAndBytesCodes(encoded, limit));
    EXPECT_ANY_THROW(ReconstructBitsAndBytesWeights(encoded, limit));
  }
  encoded.mutable_raw_data()->resize(encoded.raw_data().size() - 1);
  EXPECT_ANY_THROW(ExportBitsAndBytesWeights(encoded));
  EXPECT_ANY_THROW(DequantizeTensor(encoded));
  encoded = ImportBitsAndBytesWeights(Fixture());
  encoded.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(1)
      ->mutable_constant()
      ->ref_int64_data()[0] = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ExportBitsAndBytesWeights(encoded, std::numeric_limits<size_t>::max()));
  EXPECT_ANY_THROW(ReconstructBitsAndBytesWeights(encoded, std::numeric_limits<size_t>::max()));
  encoded = ImportBitsAndBytesWeights(Fixture());
  encoded.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(5)
      ->mutable_constant()
      ->ref_int64_data()[0] = (int64_t{1} << 32) + TensorProto::FLOAT16;
  EXPECT_ANY_THROW(ExportBitsAndBytesWeights(encoded));
}

TEST(BitsAndBytesWeights, ResolvesCatalogueAndKeepsPortableNf4Separate) {
  const auto encoded = ImportBitsAndBytesWeights(Fixture(true));
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(86);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  auto referenced = encoded;
  referenced.mutable_struct_type()->clear_structure();
  referenced.mutable_struct_type()->clear_name();
  referenced.mutable_struct_type()->set_type_ref(86);
  SameCheckpoint(Fixture(true), ExportBitsAndBytesWeights(referenced, size_t{1} << 30, catalogue));
  const auto actual = DequantizeTensor(referenced, catalogue);
  const auto expected = ReconstructBitsAndBytesWeights(encoded);
  EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  const auto plan = MakeQuantizationPlan(QuantizationFormat::kNf4, 2);
  const auto portable = QuantizeTensor(Tensor::FromFloat("x", {2}, {-1.f, 1.f}), plan);
  EXPECT_ANY_THROW(ExportBitsAndBytesWeights(portable.Encoded()));
  EXPECT_NE(portable.Encoded().struct_type().name().value(), encoded.struct_type().name().value());
}
