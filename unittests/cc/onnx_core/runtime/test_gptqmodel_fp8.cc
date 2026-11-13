// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/quantization.h"
#include <bit>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <tuple>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

Tensor Scales(Shape shape, const std::vector<float> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  for (size_t i = 0; i < values.size(); ++i) {
    const uint32_t bits = std::bit_cast<uint32_t>(values[i]);
    for (size_t j = 0; j < sizeof(float); ++j)
      bytes[i * sizeof(float) + j] = static_cast<uint8_t>(bits >> (8 * j));
  }
  return Tensor("weight_scale_inv", TensorProto::FLOAT, std::move(shape), std::move(bytes));
}

float At(const Tensor &tensor, size_t index) {
  uint32_t bits = 0;
  for (size_t j = 0; j < sizeof(float); ++j)
    bits |= uint32_t(tensor.bytes()[index * sizeof(float) + j]) << (8 * j);
  return std::bit_cast<float>(bits);
}

void EqualTensor(const Tensor &a, const Tensor &b) {
  EXPECT_EQ(a.data_type, b.data_type);
  EXPECT_EQ(a.shape, b.shape);
  ASSERT_EQ(a.size_bytes(), b.size_bytes());
  EXPECT_EQ(std::memcmp(a.bytes(), b.bytes(), a.size_bytes()), 0);
}

GptqModelFp8Weights Fixture(std::string method = "row", Shape block = {}) {
  GptqModelFp8Weights result;
  result.in_features = 4;
  result.out_features = 4;
  result.weight_scale_method = method;
  result.weight_block_size = std::move(block);
  // Pinned TorchFP8Linear storage: [N,K], no transpose or packed words.
  const std::vector<uint8_t> row{0x38, 0xbc, 0x40, 0xb0}; // 1, -1.5, 2, -0.5
  std::vector<uint8_t> codes;
  for (int i = 0; i < 4; ++i)
    codes.insert(codes.end(), row.begin(), row.end());
  result.tensors.emplace("weight",
                         Tensor("weight", TensorProto::FLOAT8E4M3FN, {4, 4}, std::move(codes)));
  if (method == "tensor")
    result.tensors.emplace("weight_scale_inv", Scales({}, {0.3f}));
  else if (method == "row")
    result.tensors.emplace("weight_scale_inv", Scales({4}, {0.3f, 1.3f, 3.f, 7.f}));
  else {
    const int64_t rows = 4 / result.weight_block_size[0];
    const int64_t cols = 4 / result.weight_block_size[1];
    std::vector<float> scales(static_cast<size_t>(rows * cols));
    for (size_t i = 0; i < scales.size(); ++i)
      scales[i] = static_cast<float>(i + 1) * 0.3f;
    result.tensors.emplace("weight_scale_inv", Scales({rows, cols}, scales));
  }
  return result;
}

} // namespace

TEST(GptqModelFp8, ExactBytesInverseScalesAndTwoAxisGrouping) {
  for (const auto &[method, block] :
       std::vector<std::pair<std::string, Shape>>{{"tensor", {}},
                                                  {"row", {}},
                                                  {"block", {1, 2}},
                                                  {"block", {2, 1}},
                                                  {"block", {2, 2}},
                                                  {"block", {4, 4}}}) {
    SCOPED_TRACE(method);
    const auto checkpoint = Fixture(method, block);
    const auto encoded = ImportGptqModelFp8Weights(checkpoint);
    EncodedValueProto restored;
    restored.ParseFromString(encoded.SerializeAsString());
    const auto exported = ExportGptqModelFp8Weights(restored);
    EXPECT_EQ(exported.format, checkpoint.format);
    EXPECT_EQ(exported.weight_scale_method, method);
    EXPECT_EQ(exported.weight_scale_semantics, "inverse");
    EXPECT_EQ(exported.weight_block_size, block);
    EXPECT_EQ(exported.upstream_revision, checkpoint.upstream_revision);
    EXPECT_EQ(exported.in_features, 4);
    EXPECT_EQ(exported.out_features, 4);
    for (const auto &[name, tensor] : checkpoint.tensors)
      EqualTensor(exported.tensors.at(name), tensor);
    const auto reconstructed = ReconstructGptqModelFp8Weights(restored);
    const auto &scales = checkpoint.tensors.at("weight_scale_inv");
    const float decoded[] = {1.f, -1.5f, 2.f, -0.5f};
    for (size_t n = 0; n < 4; ++n)
      for (size_t k = 0; k < 4; ++k) {
        const size_t index = method == "tensor" ? 0
                             : method == "row"  ? n
                                                : (n / static_cast<size_t>(block[0])) *
                                                          (4 / static_cast<size_t>(block[1])) +
                                                      k / static_cast<size_t>(block[1]);
        // fp8.py's explicit inverse path: cast weight and scales to FLOAT, then divide.
        const float expected = decoded[k] / At(scales, index);
        EXPECT_EQ(std::bit_cast<uint32_t>(At(reconstructed, n * 4 + k)),
                  std::bit_cast<uint32_t>(expected));
      }
    EqualTensor(DequantizeTensor(restored), reconstructed);
  }
}

TEST(GptqModelFp8, PreservesSubtypesAndSignedZero) {
  struct Case {
    const char *format;
    int32_t type;
    std::vector<uint8_t> codes;
    std::vector<float> decoded;
  };
  // Scalar ABI fixtures, independent of the production FP8 decoder.
  for (const auto &test : std::vector<Case>{{"float8_e4m3fn",
                                             TensorProto::FLOAT8E4M3FN,
                                             {0x38, 0xbc, 0x00, 0x80},
                                             {1.f, -1.5f, 0.f, -0.f}},
                                            {"float8_e4m3fnuz",
                                             TensorProto::FLOAT8E4M3FNUZ,
                                             {0x40, 0xc4, 0x00, 0x38},
                                             {1.f, -1.5f, 0.f, 0.5f}},
                                            {"float8_e5m2",
                                             TensorProto::FLOAT8E5M2,
                                             {0x3c, 0xbe, 0x00, 0x80},
                                             {1.f, -1.5f, 0.f, -0.f}},
                                            {"float8_e5m2fnuz",
                                             TensorProto::FLOAT8E5M2FNUZ,
                                             {0x40, 0xc2, 0x00, 0x3c},
                                             {1.f, -1.5f, 0.f, 0.5f}},
                                            {"float8_e8m0fnu",
                                             TensorProto::FLOAT8E8M0,
                                             {0x7f, 0x80, 0x7e, 0x00},
                                             {1.f, 2.f, 0.5f, 0x1p-127f}}}) {
    auto checkpoint = Fixture("tensor");
    checkpoint.in_features = 4;
    checkpoint.out_features = 1;
    checkpoint.format = test.format;
    checkpoint.tensors.at("weight") = Tensor("weight", test.type, {1, 4}, test.codes);
    const auto encoded = ImportGptqModelFp8Weights(checkpoint);
    const auto exported = ExportGptqModelFp8Weights(encoded);
    EXPECT_EQ(exported.format, test.format);
    EqualTensor(exported.tensors.at("weight"), checkpoint.tensors.at("weight"));
    const auto reconstructed = ReconstructGptqModelFp8Weights(encoded);
    for (size_t i = 0; i < 4; ++i) {
      const float expected = test.decoded[i] / 0.3f;
      EXPECT_EQ(std::bit_cast<uint32_t>(At(reconstructed, i)), std::bit_cast<uint32_t>(expected));
    }
  }
}

TEST(GptqModelFp8, RetainsNonReversibleInverseParameters) {
  auto checkpoint = Fixture("tensor");
  const float inverse = std::bit_cast<float>(uint32_t{0x3e350913});
  ASSERT_NE(std::bit_cast<uint32_t>(1.f / (1.f / inverse)), std::bit_cast<uint32_t>(inverse));
  checkpoint.tensors.at("weight_scale_inv") = Scales({}, {inverse});
  EqualTensor(ExportGptqModelFp8Weights(ImportGptqModelFp8Weights(checkpoint))
                  .tensors.at("weight_scale_inv"),
              checkpoint.tensors.at("weight_scale_inv"));
}

TEST(GptqModelFp8, UsesLittleEndianTensorAndWireBytes) {
  auto checkpoint = Fixture("tensor");
  checkpoint.tensors.at("weight_scale_inv") = Scales({}, {0.5f});
  const auto encoded = ImportGptqModelFp8Weights(checkpoint);
  const std::string scale_bytes("\x00\x00\x00\x3f", 4);
  EXPECT_EQ(std::memcmp(encoded.raw_data().data() + 16, scale_bytes.data(), 4), 0);
  const auto exported = ExportGptqModelFp8Weights(encoded);
  EXPECT_EQ(std::memcmp(exported.tensors.at("weight_scale_inv").bytes(), scale_bytes.data(), 4), 0);

  const std::string floats("\x00\x00\x00\x40\x00\x00\x40\xc0", 8);
  const std::string halves("\x00\x40\x00\xc2", 4);
  const std::string doubles("\x00\x00\x00\x00\x00\x00\x00\x40"
                            "\x00\x00\x00\x00\x00\x00\x08\xc0",
                            16);
  const auto reconstructed = ReconstructGptqModelFp8Weights(encoded);
  EXPECT_EQ(std::memcmp(reconstructed.bytes(), floats.data(), floats.size()), 0);
  const auto generic = DequantizeTensor(encoded);
  EXPECT_EQ(std::memcmp(generic.bytes(), floats.data(), floats.size()), 0);
  const auto generic_half = DequantizeTensor(encoded, {}, nullptr, TensorProto::FLOAT16);
  EXPECT_EQ(std::memcmp(generic_half.bytes(), halves.data(), halves.size()), 0);
  const auto generic_double = DequantizeTensor(encoded, {}, nullptr, TensorProto::DOUBLE);
  EXPECT_EQ(std::memcmp(generic_double.bytes(), doubles.data(), doubles.size()), 0);
  const auto proto = DequantizeTensorProto(encoded);
  EXPECT_EQ(std::memcmp(proto.raw_data().data(), floats.data(), floats.size()), 0);
}

TEST(GptqModelFp8, RejectsInvalidCheckpointGeometryDtypesExtentsAndLimits) {
  const auto good = Fixture();
  auto rejects = [&](GptqModelFp8Weights bad) { EXPECT_ANY_THROW(ImportGptqModelFp8Weights(bad)); };
  auto bad = good;
  bad.tensors.erase("weight_scale_inv");
  rejects(bad);
  bad = good;
  bad.tensors.emplace("bias", Scales({}, {1.f}));
  rejects(bad);
  bad = good;
  bad.upstream_revision = "unverified";
  rejects(bad);
  bad = good;
  bad.format = "mxfp8";
  rejects(bad);
  bad = good;
  bad.weight_scale_method = "channel";
  rejects(bad);
  bad = good;
  bad.weight_scale_semantics = "heuristic";
  rejects(bad);
  bad = good;
  bad.weight_scale_method = "block";
  for (const Shape &block : std::vector<Shape>{{}, {2}, {0, 2}, {-1, 2}, {3, 2}, {2, 3}}) {
    bad.weight_block_size = block;
    rejects(bad);
  }
  for (int64_t dim : {int64_t{0}, int64_t{-1}, std::numeric_limits<int64_t>::max()}) {
    bad = good;
    bad.in_features = dim;
    rejects(bad);
  }
  bad = good;
  bad.in_features = bad.out_features = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ImportGptqModelFp8Weights(bad, std::numeric_limits<size_t>::max()));
  bad = good;
  bad.tensors.at("weight").data_type = TensorProto::UINT8;
  rejects(bad);
  bad = good;
  bad.tensors.at("weight").shape = {2, 8};
  rejects(bad);
  bad = good;
  bad.tensors.at("weight") =
      Tensor("weight", TensorProto::FLOAT8E4M3FN, {4, 4}, std::vector<uint8_t>(15));
  rejects(bad);
  bad = good;
  bad.tensors.at("weight_scale_inv").data_type = TensorProto::FLOAT16;
  rejects(bad);
  bad = good;
  bad.tensors.at("weight_scale_inv") = Scales({4, 1}, {1, 1, 1, 1});
  rejects(bad);
  bad = good;
  bad.tensors.at("weight_scale_inv") = Scales({4}, {1});
  rejects(bad);
  EXPECT_ANY_THROW(ImportGptqModelFp8Weights(good, 0));
  EXPECT_ANY_THROW(ImportGptqModelFp8Weights(good, 63));
  EXPECT_NO_THROW(ImportGptqModelFp8Weights(good, 64));
}

TEST(GptqModelFp8, RejectsNonfiniteParametersCodesAndReconstruction) {
  for (float scale :
       {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::denorm_min()}) {
    auto checkpoint = Fixture("tensor");
    checkpoint.tensors.at("weight_scale_inv") = Scales({}, {scale});
    EXPECT_ANY_THROW(ImportGptqModelFp8Weights(checkpoint));
  }
  for (const auto &[format, type, code] : std::vector<std::tuple<std::string, int32_t, uint8_t>>{
           {"float8_e4m3fn", TensorProto::FLOAT8E4M3FN, 0x7f},
           {"float8_e4m3fnuz", TensorProto::FLOAT8E4M3FNUZ, 0x80},
           {"float8_e5m2", TensorProto::FLOAT8E5M2, 0x7c},
           {"float8_e5m2", TensorProto::FLOAT8E5M2, 0x7f},
           {"float8_e5m2fnuz", TensorProto::FLOAT8E5M2FNUZ, 0x80},
           {"float8_e8m0fnu", TensorProto::FLOAT8E8M0, 0xff}}) {
    auto checkpoint = Fixture("tensor");
    checkpoint.format = format;
    checkpoint.tensors.at("weight").data_type = type;
    checkpoint.tensors.at("weight").mutable_bytes()[0] = code;
    EXPECT_ANY_THROW(ImportGptqModelFp8Weights(checkpoint));
  }
}

TEST(GptqModelFp8, ValidatesEncodedSchemaAndResolvesCatalogue) {
  const auto encoded = ImportGptqModelFp8Weights(Fixture());
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(92);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  auto referenced = encoded;
  referenced.mutable_struct_type()->Clear();
  referenced.mutable_struct_type()->set_type_ref(92);
  EqualTensor(ReconstructGptqModelFp8Weights(referenced, 64, catalogue),
              ReconstructGptqModelFp8Weights(encoded));
  EXPECT_EQ(ExportGptqModelFp8Weights(referenced, 64, catalogue).format, "float8_e4m3fn");
  for (size_t size : {size_t{0}, encoded.raw_data().size() - 1, encoded.raw_data().size() + 1}) {
    auto bad = encoded;
    bad.mutable_raw_data()->resize(size);
    EXPECT_ANY_THROW(ExportGptqModelFp8Weights(bad));
    EXPECT_ANY_THROW(ReconstructGptqModelFp8Weights(bad));
  }
  auto bad = encoded;
  bad.mutable_struct_type()->set_name("onnx_light.gptqmodel_fp8_weights.v2");
  EXPECT_ANY_THROW(ExportGptqModelFp8Weights(bad));
  bad = encoded;
  bad.mutable_struct_type()->mutable_structure()->mutable_field(8)->set_name("weight_scale");
  EXPECT_ANY_THROW(ExportGptqModelFp8Weights(bad));
  bad = encoded;
  bad.mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT16);
  EXPECT_ANY_THROW(ExportGptqModelFp8Weights(bad));
  bad = encoded;
  // The first inverse scale follows the 16 weight bytes, little-endian.
  std::memset(bad.mutable_raw_data()->data() + 16, 0, sizeof(float));
  EXPECT_ANY_THROW(ExportGptqModelFp8Weights(bad));
  EXPECT_ANY_THROW(ExportGptqModelFp8Weights(encoded, 63));
  const auto portable =
      QuantizeTensor(Scales({}, {1.f}), MakeQuantizationPlan(QuantizationFormat::kFp8E4m3, 1));
  EXPECT_ANY_THROW(ExportGptqModelFp8Weights(portable.Encoded()));
}
