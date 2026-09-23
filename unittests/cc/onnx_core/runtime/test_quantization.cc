// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

constexpr auto kQuantizationFormats = QuantizationFormats();
static_assert(kQuantizationFormats.size() == 40);
static_assert(kQuantizationFormats.front() == "int8");
static_assert(kQuantizationFormats.back() == "column_major");
static_assert([] {
  for (size_t i = 0; i < kQuantizationFormats.size(); ++i) {
    if (kQuantizationFormats[i].empty())
      return false;
    for (size_t j = 0; j < i; ++j)
      if (kQuantizationFormats[i] == kQuantizationFormats[j])
        return false;
  }
  return true;
}());

void ExpectValues(const Tensor &tensor, const std::vector<float> &expected, float tolerance = 0) {
  ASSERT_EQ(tensor.data_type, TensorProto::FLOAT);
  ASSERT_EQ(tensor.element_count(), static_cast<int64_t>(expected.size()));
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(tensor.AsFloat()[i], expected[i], tolerance) << i;
}

RuntimeValue WireRoundTrip(const RuntimeValue &value) {
  EncodedValueProto parsed;
  parsed.ParseFromString(value.Encoded().SerializeAsString());
  return RuntimeValue(std::move(parsed));
}

QuantizationPlan WithTables(const std::string &format, size_t count) {
  auto plan = MakeQuantizationPlan(format, count, 4);
  for (auto &block : plan.blocks) {
    if (block.method == QuantizationMethod::kCodebook && block.codebook.empty()) {
      block.bits = 2;
      block.entries = 4;
      block.vector_size = 2;
      block.codebook.resize(block.books * block.entries * block.vector_size);
      for (size_t book = 0; book < block.books; ++book)
        for (size_t entry = 0; entry < block.entries; ++entry)
          for (size_t j = 0; j < block.vector_size; ++j)
            block.codebook[(book * block.entries + entry) * block.vector_size + j] =
                book == 0 ? double(entry) - 1 : 0;
    }
  }
  if (format == "quarot" || format == "quip_sharp" || format == "smoothquant") {
    plan.transform_size = 2;
    plan.forward = {1, 0, 0, 1};
    plan.inverse = plan.forward;
  }
  return plan;
}

} // namespace

TEST(Quantization, AffineGoldenBytesRoundingClippingAndOwnership) {
  auto source = Tensor::FromFloat("weights", {7}, {-9, -7.5f, -0.5f, 0.5f, 1.5f, 6.5f, 8});
  auto plan = MakeQuantizationPlan("int4", 7);
  auto encoded = QuantizeTensor(source, plan);
  source.AsFloat()[0] = 99;
  const auto raw = std::string(encoded.Encoded().raw_data());
  ASSERT_EQ(raw.size(), 1u + 24u + 4u);
  EXPECT_EQ(static_cast<uint8_t>(raw[raw.size() - 4]), 0x88);
  EXPECT_EQ(static_cast<uint8_t>(raw[raw.size() - 3]), 0);
  EXPECT_EQ(static_cast<uint8_t>(raw[raw.size() - 2]), 0x62);
  EXPECT_EQ(static_cast<uint8_t>(raw[raw.size() - 1]), 7);
  const auto decoded = DequantizeTensor(WireRoundTrip(encoded));
  EXPECT_EQ(decoded.name, "weights");
  ExpectValues(decoded, {-8, -8, 0, 0, 2, 6, 7});
}

TEST(Quantization, EveryCatalogueProfileProducesSelfContainedValues) {
  const auto tensor = Tensor::FromFloat("", {8}, {-1, -1, 0, 0, 1, 1, 1, 1});
  TensorProto proto;
  proto.set_data_type(TensorProto::FLOAT);
  proto.add_dims(8);
  for (float value : {-1, -1, 0, 0, 1, 1, 1, 1})
    proto.add_float_data(value);
  EXPECT_EQ(QuantizationFormats().size(), 40u);
  for (const auto &format : QuantizationFormats()) {
    SCOPED_TRACE(format);
    auto plan = WithTables(std::string(format), 8);
    if (format == "iq4_nl")
      plan.blocks[0].scale = plan.blocks[1].scale = 0.01;
    const auto encoded = QuantizeTensor(tensor, plan);
    const auto decoded = DequantizeTensor(WireRoundTrip(encoded));
    const auto encoded_proto = QuantizeTensorProto(proto, plan);
    EXPECT_EQ(encoded_proto.raw_data(), encoded.Encoded().raw_data());
    EXPECT_EQ(encoded_proto.struct_type().SerializeAsString(),
              encoded.Encoded().struct_type().SerializeAsString());
    const auto decoded_proto = DequantizeTensorProto(encoded_proto);
    const auto decoded_direct = DequantizeTensor(encoded_proto);
    const auto proto_tensor = TensorFromProto(decoded_proto);
    EXPECT_EQ(std::memcmp(decoded.bytes(), proto_tensor.bytes(), decoded.size_bytes()), 0);
    EXPECT_EQ(std::memcmp(decoded.bytes(), decoded_direct.bytes(), decoded.size_bytes()), 0);
    ASSERT_EQ(decoded.shape, tensor.shape);
    for (int i = 0; i < 8; ++i) {
      EXPECT_TRUE(std::isfinite(decoded.AsFloat()[i]));
      EXPECT_NEAR(decoded.AsFloat()[i], tensor.AsFloat()[i],
                  format == "binary"   ? 1
                  : format == "iq4_nl" ? 0.14
                                       : 1e-6);
    }
  }
}

TEST(Quantization, MixedPrecisionOffsetsAndIndependentBlockScales) {
  auto plan = MakeQuantizationPlan("exl2", 6, 3);
  plan.blocks[0].bits = 2;
  plan.blocks[0].scale = 0.25;
  plan.blocks[0].offset = 1.125;
  plan.blocks[1].bits = 5;
  plan.blocks[1].scale = 2;
  const auto tensor = Tensor::FromFloat("", {2, 3}, {0.625, 1.125, 1.375, -32, 0, 30});
  ExpectValues(DequantizeTensor(WireRoundTrip(QuantizeTensor(tensor, plan))),
               {0.625, 1.125, 1.375, -32, 0, 30});
}

TEST(Quantization, ConstantProfileTablesRemainIndependent) {
  const std::vector<double> expected{0,  0.125, -0.125, 0.25, -0.25, 0.5, -0.5, 1,
                                     -1, 2,     -2,     4,    -4,    8,   -8};
  auto plan = MakeQuantizationPlan("log", 2, 1);
  ASSERT_EQ(plan.blocks.size(), 2u);
  EXPECT_EQ(plan.blocks[0].codebook, expected);
  EXPECT_EQ(plan.blocks[1].codebook, expected);
  plan.blocks[0].codebook[0] = 99;
  EXPECT_EQ(plan.blocks[1].codebook, expected);
  EXPECT_EQ(MakeQuantizationPlan("log", 1).blocks[0].codebook, expected);
  ExpectValues(DequantizeTensor(QuantizeTensor(Tensor::FromFloat("", {4}, {0.125f, -0.25f, 4, -8}),
                                               MakeQuantizationPlan("log", 4))),
               {0.125f, -0.25f, 4, -8});
}

TEST(Quantization, Base3GoldenBytesAndShortTail) {
  auto plan = MakeQuantizationPlan("tq1_0", 7);
  auto encoded = QuantizeTensor(Tensor::FromFloat("", {7}, {-1, 0, 1, -1, 1, 1, 0}), plan);
  const auto raw = std::string(encoded.Encoded().raw_data());
  EXPECT_EQ(static_cast<uint8_t>(raw[raw.size() - 2]), 183);
  EXPECT_EQ(static_cast<uint8_t>(raw.back()), 5);
  ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), {-1, 0, 1, -1, 1, 1, 0});
}

TEST(Quantization, AdditiveVectorCodebooksAndPartialVectors) {
  auto plan = MakeQuantizationPlan("aqlm", 3);
  auto &block = plan.blocks[0];
  block.entries = 2;
  block.bits = 1;
  block.vector_size = 2;
  block.codebook = {0, 0, 10, 20, 0, 0, 1, 2};
  const auto source = Tensor::FromFloat("", {3}, {11, 22, 1});
  auto encoded = QuantizeTensor(source, plan);
  EXPECT_EQ(static_cast<uint8_t>(std::string(encoded.Encoded().raw_data()).back()), 11);
  ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), {11, 22, 1});
}

TEST(Quantization, PermutationRotationAndSparseOutliers) {
  auto plan = MakeQuantizationPlan("quarot", 4);
  plan.permutation = {2, 3, 0, 1};
  plan.transform_size = 2;
  plan.forward = {1, 1, 1, -1};
  plan.inverse = {0.5, 0.5, 0.5, -0.5};
  plan.outliers = {1};
  const auto source = Tensor::FromFloat("", {2, 2}, {1, 1000, 2, 3});
  auto encoded = QuantizeTensor(source, plan);
  plan = {};
  ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), {1, 1000, 2, 3});
}

TEST(Quantization, FloatProfilesUseExpectedRepresentableValues) {
  for (const std::string format : {"mxfp4", "nvfp4", "mxfp6", "fp6_llm", "fp8_e4m3"}) {
    SCOPED_TRACE(format);
    auto plan = MakeQuantizationPlan(format, 6);
    plan.blocks[0].scale = 2;
    const auto encoded = QuantizeTensor(Tensor::FromFloat("", {6}, {-12, -3, -1, 0, 1, 12}), plan);
    ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), {-12, -3, -1, 0, 1, 12});
  }
}

TEST(Quantization, PreservesScalarEmptyTensorAndSourceDtypes) {
  for (const Shape &shape : {Shape{}, Shape{0}, Shape{2, 0}}) {
    const size_t count = shape.empty() ? 1 : 0;
    auto plan = MakeQuantizationPlan("int8", count);
    auto source =
        Tensor::FromFloat("", shape, count ? std::vector<float>{2} : std::vector<float>{});
    auto decoded = DequantizeTensor(WireRoundTrip(QuantizeTensor(source, plan)));
    EXPECT_EQ(decoded.shape, shape);
    EXPECT_EQ(decoded.element_count(), static_cast<int64_t>(count));
  }
  auto plan = MakeQuantizationPlan("int8", 3);
  for (int32_t dtype :
       {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    SCOPED_TRACE(dtype);
    Tensor source;
    if (dtype == TensorProto::FLOAT)
      source = Tensor::FromFloat("", {3}, {-2, 0, 2});
    else if (dtype == TensorProto::DOUBLE)
      source = Tensor::FromDouble("", {3}, {-2, 0, 2});
    else if (dtype == TensorProto::FLOAT16)
      source = MakeFloat16Tensor("", {3}, {-2, 0, 2});
    else
      source = MakeBfloat16Tensor("", {3}, {-2, 0, 2});
    const auto encoded = WireRoundTrip(QuantizeTensor(source, plan));
    const auto result = DequantizeTensor(encoded);
    EXPECT_EQ(result.data_type, dtype);
    EXPECT_EQ(result.size_bytes(), source.size_bytes());
    EXPECT_EQ(std::memcmp(result.bytes(), source.bytes(), source.size_bytes()), 0);
    const auto proto = DequantizeTensorProto(encoded.Encoded());
    const auto proto_tensor = TensorFromProto(proto);
    EXPECT_EQ(proto.data_type(), dtype);
    EXPECT_EQ(proto_tensor.size_bytes(), source.size_bytes());
    EXPECT_EQ(std::memcmp(proto_tensor.bytes(), source.bytes(), source.size_bytes()), 0);
  }
}

TEST(Quantization, ProtoRoundTripPreservesNameShapeAndDocumentation) {
  TensorProto tensor;
  tensor.set_name("weight");
  tensor.set_doc_string("example");
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.add_dims(2);
  tensor.add_dims(2);
  for (float value : {-2, -1, 1, 2})
    tensor.add_float_data(value);
  const auto original = tensor.SerializeAsString();
  auto encoded = QuantizeTensorProto(tensor, MakeQuantizationPlan("int8", 4));
  EncodedValueProto parsed;
  parsed.ParseFromString(encoded.SerializeAsString());
  const auto decoded = DequantizeTensorProto(parsed);
  EXPECT_EQ(tensor.SerializeAsString(), original);
  EXPECT_EQ(decoded.name(), tensor.name());
  EXPECT_EQ(decoded.doc_string(), tensor.doc_string());
  EXPECT_EQ(decoded.dims().size(), 2u);
  EXPECT_EQ(decoded.dims(0), 2);
  ExpectValues(TensorFromProto(decoded), {-2, -1, 1, 2});
}

TEST(Quantization, ResolvesRootCatalogueReferenceAndUsesAllocator) {
  auto encoded =
      QuantizeTensor(Tensor::FromFloat("", {2}, {1, 2}), MakeQuantizationPlan("int8", 2)).Encoded();
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(17);
  encoded.mutable_struct_type()->Clear();
  encoded.mutable_struct_type()->set_type_ref(17);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  SimpleRawBufferAllocator allocator(4);
  auto decoded = DequantizeTensor(encoded, catalogue, &allocator);
  ExpectValues(decoded, {1, 2});
  EXPECT_GE(allocator.TotalAllocatedSize(), 2 * sizeof(float));
  EXPECT_THROW(DequantizeTensor(RuntimeValue(encoded)), std::invalid_argument);
}

TEST(Quantization, DirectMessageDecodingOwnsOutputsAfterSourceRelease) {
  auto encoded =
      QuantizeTensor(Tensor::FromFloat("", {3}, {-1, 0, 1}), MakeQuantizationPlan("int4", 3))
          .Encoded();
  const auto *payload = encoded.raw_data().data();
  const auto original = encoded.SerializeAsString();
  auto tensor = DequantizeTensor(encoded);
  auto proto = DequantizeTensorProto(encoded);
  EXPECT_EQ(encoded.raw_data().data(), payload);
  EXPECT_EQ(encoded.SerializeAsString(), original);
  EXPECT_FALSE(proto.raw_data().is_borrowed());
  encoded.Clear();
  ExpectValues(tensor, {-1, 0, 1});
  ExpectValues(TensorFromProto(proto), {-1, 0, 1});
}

TEST(Quantization, RejectsIncompleteAndInvalidPlans) {
  const auto source = Tensor::FromFloat("", {2}, {1, 2});
  EXPECT_THROW(MakeQuantizationPlan("unknown", 2), std::invalid_argument);
  EXPECT_THROW(MakeQuantizationPlan("int4", 2, 0), std::invalid_argument);
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan("aqlm", 2)), std::invalid_argument);
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan("quarot", 2)), std::invalid_argument);
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan("int4", 1)), std::invalid_argument);
  for (double scale : {0., -1., std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    auto plan = MakeQuantizationPlan("int4", 2);
    plan.blocks[0].scale = scale;
    EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  }
  auto plan = MakeQuantizationPlan("int4", 2);
  plan.permutation = {0, 0};
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.permutation.clear();
  plan.outliers = {2};
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.outliers.clear();
  plan.blocks[0].bits = 0;
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.blocks[0].bits = 4;
  plan.blocks[0].zero_point = 0.5;
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.blocks[0].zero_point = 0;
  plan.transform_size = 2;
  plan.forward = {1, 0, 0, 2};
  plan.inverse = {1, 0, 0, 1};
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
}

TEST(Quantization, RejectsUnknownProfilesInMutablePlansAndEncodedLayouts) {
  const auto source = Tensor::FromFloat("", {2}, {1, 2});
  const auto valid = QuantizeTensor(source, MakeQuantizationPlan("int4", 2)).Encoded();
  for (const std::string name : {"", "unknown", "QUAROT"}) {
    SCOPED_TRACE(name);
    auto plan = MakeQuantizationPlan("quarot", 2);
    plan.format = name;
    EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
    auto encoded = valid;
    encoded.mutable_struct_type()->set_name("onnx_light.quantization.v1/" + name);
    EXPECT_THROW(DequantizeTensor(RuntimeValue(encoded)), std::invalid_argument);
  }
  auto renamed = valid;
  renamed.mutable_struct_type()->set_name("onnx_light.quantization.v1/quarot");
  EXPECT_THROW(DequantizeTensor(RuntimeValue(renamed)), std::invalid_argument);
}

TEST(Quantization, OptionalNamesAndDocumentationMayBeAbsent) {
  TensorProto source;
  source.set_data_type(TensorProto::FLOAT);
  source.add_dims(2);
  source.add_float_data(1);
  source.add_float_data(2);
  ASSERT_FALSE(source.has_name());
  ASSERT_FALSE(source.has_doc_string());
  auto encoded = QuantizeTensorProto(source, MakeQuantizationPlan("int4", 2));
  encoded.clear_name();
  encoded.clear_doc_string();
  EncodedValueProto loaded;
  loaded.ParseFromString(encoded.SerializeAsString());
  ASSERT_FALSE(loaded.has_name());
  ASSERT_FALSE(loaded.has_doc_string());
  auto tensor = DequantizeTensor(RuntimeValue(loaded));
  EXPECT_TRUE(tensor.name.empty());
  ExpectValues(tensor, {1, 2});
  const auto decoded = DequantizeTensorProto(loaded);
  EXPECT_TRUE(decoded.name().empty());
  EXPECT_TRUE(decoded.doc_string().empty());
  ExpectValues(TensorFromProto(decoded), {1, 2});
}

TEST(Quantization, AcceptsLoadedExternalRawDataWithoutChangingMetadata) {
  TensorProto external;
  external.set_data_type(TensorProto::FLOAT);
  external.add_dims(2);
  external.set_data_location(TensorProto::EXTERNAL);
  auto *location = external.add_external_data();
  location->set_key("location");
  location->set_value("weights.bin");
  external.set_raw_data(std::string("\0\0\x80\x3f\0\0\0\x40", 8));
  const auto original = external.SerializeAsString();
  const auto encoded = QuantizeTensorProto(external, MakeQuantizationPlan("int4", 2));
  ExpectValues(DequantizeTensor(RuntimeValue(encoded)), {1, 2});
  EXPECT_EQ(external.SerializeAsString(), original);
  external.set_raw_data(std::string("\0", 1));
  EXPECT_THROW(QuantizeTensorProto(external, MakeQuantizationPlan("int4", 2)),
               std::invalid_argument);
}

TEST(Quantization, RejectsCorruptPayloadAndSchema) {
  const auto source = Tensor::FromFloat("", {3}, {1, 0, -1});
  auto valid = QuantizeTensor(source, MakeQuantizationPlan("int4", 3)).Encoded();
  auto corrupt = valid;
  auto raw = std::string(valid.raw_data());
  raw.pop_back();
  corrupt.set_raw_data(raw);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(corrupt)), std::invalid_argument);
  corrupt = valid;
  raw = std::string(valid.raw_data());
  raw.back() = static_cast<char>(0xff);
  corrupt.set_raw_data(raw);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(corrupt)), std::invalid_argument);
  corrupt = valid;
  corrupt.mutable_struct_type()->set_name("unknown");
  EXPECT_THROW(DequantizeTensor(RuntimeValue(corrupt)), std::invalid_argument);
  corrupt = valid;
  corrupt.mutable_struct_type()->mutable_structure()->mutable_field(0)->set_name("changed");
  EXPECT_THROW(DequantizeTensor(RuntimeValue(corrupt)), std::invalid_argument);
  valid = QuantizeTensor(source, MakeQuantizationPlan("tq1_0", 3)).Encoded();
  raw = std::string(valid.raw_data());
  raw.back() = static_cast<char>(243);
  valid.set_raw_data(raw);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(valid)), std::invalid_argument);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(source)), std::invalid_argument);
}

TEST(Quantization, RejectsNonfiniteInputsAndUnloadedExternalData) {
  const auto source = Tensor::FromFloat("", {1}, {std::numeric_limits<float>::quiet_NaN()});
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan("int4", 1)), std::invalid_argument);
  TensorProto external;
  external.set_data_type(TensorProto::FLOAT);
  external.add_dims(1);
  external.set_data_location(TensorProto::EXTERNAL);
  EXPECT_THROW(QuantizeTensorProto(external, MakeQuantizationPlan("int4", 1)),
               std::invalid_argument);
}

TEST(Quantization, EveryIndexWidthPacksAcrossByteBoundaries) {
  for (uint32_t bits = 1; bits <= 16; ++bits) {
    SCOPED_TRACE(bits);
    auto plan = MakeQuantizationPlan("exl2", 19);
    auto &block = plan.blocks[0];
    block.bits = bits;
    block.scale = 0.25;
    std::vector<float> numbers;
    std::vector<uint32_t> expected_codes;
    const int32_t lower = -(int32_t{1} << (bits - 1));
    for (int32_t i = 0; i < 19; ++i) {
      const int32_t code = lower + i % (int32_t{1} << bits);
      numbers.push_back(static_cast<float>(code) * 0.25f);
      expected_codes.push_back(static_cast<uint32_t>(code) & ((uint32_t{1} << bits) - 1));
    }
    auto encoded = QuantizeTensor(Tensor::FromFloat("", {19}, numbers), plan);
    const auto raw = std::string(encoded.Encoded().raw_data());
    const size_t bytes = (19 * bits + 7) / 8;
    ASSERT_EQ(raw.size(), 25 + bytes);
    for (size_t index = 0; index < expected_codes.size(); ++index) {
      uint32_t actual = 0;
      for (uint32_t bit = 0; bit < bits; ++bit) {
        const size_t position = index * bits + bit;
        actual |= ((static_cast<uint8_t>(raw[25 + position / 8]) >> (position % 8)) & 1) << bit;
      }
      EXPECT_EQ(actual, expected_codes[index]);
    }
    ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), numbers);
  }
}

TEST(Quantization, ScalarCodebookGoldenIndicesAndTieBreaking) {
  auto plan = MakeQuantizationPlan("nf4", 3);
  const auto encoded = QuantizeTensor(Tensor::FromFloat("", {3}, {-1, 0, 1}), plan);
  const auto raw = std::string(encoded.Encoded().raw_data());
  EXPECT_EQ(static_cast<uint8_t>(raw[raw.size() - 2]), 0x70);
  EXPECT_EQ(static_cast<uint8_t>(raw.back()), 0x0f);
  plan = MakeQuantizationPlan("binary", 4);
  const auto binary = QuantizeTensor(Tensor::FromFloat("", {4}, {-1, 1, 0, 1}), plan);
  EXPECT_EQ(static_cast<uint8_t>(std::string(binary.Encoded().raw_data()).back()), 0x0a);
  ExpectValues(DequantizeTensor(binary), {-1, 1, -1, 1});
}

TEST(Quantization, LearnedProfilesAcceptTheirDefaultTableDimensions) {
  for (const std::string format : {"stq1_0", "iq1_s", "aqlm", "quip_sharp"}) {
    SCOPED_TRACE(format);
    auto plan = MakeQuantizationPlan(format, 17);
    auto &block = plan.blocks[0];
    block.codebook.resize(block.books * block.entries * block.vector_size);
    for (uint32_t book = 0; book < block.books; ++book)
      for (uint32_t entry = 0; entry < block.entries; ++entry)
        for (uint32_t j = 0; j < block.vector_size; ++j)
          block.codebook[(book * block.entries + entry) * block.vector_size + j] =
              book == 0 ? double(entry) : 0;
    if (format == "quip_sharp") {
      plan.transform_size = 1;
      plan.forward = {1};
      plan.inverse = {1};
    }
    std::vector<float> values(17, 7);
    ExpectValues(
        DequantizeTensor(WireRoundTrip(QuantizeTensor(Tensor::FromFloat("", {17}, values), plan))),
        values);
  }
}

TEST(Quantization, CastProfilesRetainRequestedStoragePrecision) {
  for (int32_t type :
       {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT, TensorProto::DOUBLE}) {
    auto plan = MakeQuantizationPlan("tiled_float", 4);
    plan.blocks[0].cast_type = type;
    const auto input = Tensor::FromFloat("", {4}, {-2, 0.5, 1.5, 4});
    ExpectValues(DequantizeTensor(WireRoundTrip(QuantizeTensor(input, plan))), {-2, 0.5, 1.5, 4});
  }
  auto plan = MakeQuantizationPlan("tiled_float", 1);
  plan.blocks[0].cast_type = TensorProto::FLOAT16;
  EXPECT_THROW(QuantizeTensor(Tensor::FromFloat("", {1}, {1e10f}), plan), std::invalid_argument);
}

TEST(Quantization, RejectsOutOfRangeIndicesAndAlteredDescriptorGeometry) {
  auto valid =
      QuantizeTensor(Tensor::FromFloat("", {1}, {1}), MakeQuantizationPlan("tq2_0", 1)).Encoded();
  auto raw = std::string(valid.raw_data());
  raw.back() = 3;
  valid.set_raw_data(raw);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(valid)), std::invalid_argument);
  valid =
      QuantizeTensor(Tensor::FromFloat("", {1}, {1}), MakeQuantizationPlan("int4", 1)).Encoded();
  valid.mutable_logical_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(0)
      ->set_dim_value(2);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(valid)), std::invalid_argument);
}

TEST(Quantization, RepeatedBlockLayoutsShareOneDescriptor) {
  auto small_plan = MakeQuantizationPlan("int4", 128, 128);
  auto large_plan = MakeQuantizationPlan("int4", 128 * 100, 128);
  for (size_t i = 0; i < large_plan.blocks.size(); ++i)
    large_plan.blocks[i].scale = double(i + 1);
  auto small = QuantizeTensor(Tensor::FromFloat("", {128}, std::vector<float>(128, 0)), small_plan);
  auto large = QuantizeTensor(Tensor::FromFloat("", {128 * 100}, std::vector<float>(128 * 100, 0)),
                              large_plan);
  EXPECT_LT(large.Encoded().struct_type().SerializeAsString().size(),
            small.Encoded().struct_type().SerializeAsString().size() + 32);
  const auto &runs = large.Encoded().struct_type().structure().field(6).type().struct_type();
  ASSERT_EQ(runs.structure().field().size(), 2u);
  EXPECT_EQ(runs.structure().field(1).type().struct_type().array().dimension(), 100u);
  auto retained = std::move(large).Retain();
  ExpectValues(DequantizeTensor(WireRoundTrip(retained)), std::vector<float>(128 * 100, 0));
}
