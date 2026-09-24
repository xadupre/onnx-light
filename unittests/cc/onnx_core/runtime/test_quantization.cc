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
static_assert(kQuantizationFormats.size() == 43);
static_assert(kQuantizationFormats.front() == QuantizationFormat::kInt8);
static_assert(kQuantizationFormats.back() == QuantizationFormat::kOrtMatmulnbitsInt8);
static_assert([] {
  for (size_t i = 0; i < kQuantizationFormats.size(); ++i) {
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

QuantizationPlan WithTables(QuantizationFormat format, size_t count) {
  if (format >= QuantizationFormat::kOrtMatmulnbitsInt2)
    return MakeMatMulNBitsPlan(format, count / 2, 2, 16);
  auto plan = MakeQuantizationPlan(format, count, 4);
  for (auto &run : plan.runs) {
    auto &layout = run.layout;
    for (auto &block : run.blocks) {
      if (layout.method == QuantizationMethod::kCodebook && block.codebook.empty()) {
        layout.bits = 2;
        layout.entries = 4;
        layout.vector_size = 2;
        block.codebook.resize(layout.books * layout.entries * layout.vector_size);
        for (size_t book = 0; book < layout.books; ++book)
          for (size_t entry = 0; entry < layout.entries; ++entry)
            for (size_t j = 0; j < layout.vector_size; ++j)
              block.codebook[(book * layout.entries + entry) * layout.vector_size + j] =
                  book == 0 ? double(entry) - 1 : 0;
      }
    }
  }
  if (format == QuantizationFormat::kQuarot || format == QuantizationFormat::kQuipSharp ||
      format == QuantizationFormat::kSmoothquant) {
    plan.transform_size = 2;
    plan.forward = {1, 0, 0, 1};
    plan.inverse = plan.forward;
  }
  return plan;
}

} // namespace

TEST(Quantization, AffineGoldenBytesRoundingClippingAndOwnership) {
  auto source = Tensor::FromFloat("weights", {7}, {-9, -7.5f, -0.5f, 0.5f, 1.5f, 6.5f, 8});
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 7);
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
  const auto tensor = Tensor::FromFloat("", {4, 2}, {-1, -1, 0, 0, 1, 1, 1, 1});
  TensorProto proto;
  proto.set_data_type(TensorProto::FLOAT);
  proto.add_dims(4);
  proto.add_dims(2);
  for (float value : {-1, -1, 0, 0, 1, 1, 1, 1})
    proto.add_float_data(value);
  EXPECT_EQ(QuantizationFormats().size(), 43u);
  for (const auto &format : QuantizationFormats()) {
    SCOPED_TRACE(QuantizationFormatName(format));
    auto plan = WithTables(format, 8);
    if (format == QuantizationFormat::kIq4Nl)
      plan.runs[0].blocks[0].scale = plan.runs[0].blocks[1].scale = 0.01;
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
                  format == QuantizationFormat::kBinary  ? 1
                  : format == QuantizationFormat::kIq4Nl ? 0.14
                                                         : 1e-6);
    }
  }
}

TEST(Quantization, MixedPrecisionOffsetsAndIndependentBlockScales) {
  auto plan = MakeQuantizationPlan(QuantizationFormat::kExl2, 6, 3);
  auto second = plan.runs[0];
  plan.runs[0].blocks.resize(1);
  second.blocks.resize(1);
  plan.runs.push_back(std::move(second));
  plan.runs[0].layout.bits = 2;
  plan.runs[0].blocks[0].scale = 0.25;
  plan.runs[0].blocks[0].offset = 1.125;
  plan.runs[1].layout.bits = 5;
  plan.runs[1].blocks[0].scale = 2;
  const auto tensor = Tensor::FromFloat("", {2, 3}, {0.625, 1.125, 1.375, -32, 0, 30});
  ExpectValues(DequantizeTensor(WireRoundTrip(QuantizeTensor(tensor, plan))),
               {0.625, 1.125, 1.375, -32, 0, 30});
}

TEST(Quantization, ConstantProfileTablesRemainIndependent) {
  const std::vector<double> expected{0,  0.125, -0.125, 0.25, -0.25, 0.5, -0.5, 1,
                                     -1, 2,     -2,     4,    -4,    8,   -8};
  auto plan = MakeQuantizationPlan(QuantizationFormat::kLog, 2, 1);
  ASSERT_EQ(plan.runs[0].blocks.size(), 2u);
  EXPECT_EQ(plan.runs[0].blocks[0].codebook, expected);
  EXPECT_EQ(plan.runs[0].blocks[1].codebook, expected);
  plan.runs[0].blocks[0].codebook[0] = 99;
  EXPECT_EQ(plan.runs[0].blocks[1].codebook, expected);
  EXPECT_EQ(MakeQuantizationPlan(QuantizationFormat::kLog, 1).runs[0].blocks[0].codebook, expected);
  ExpectValues(DequantizeTensor(QuantizeTensor(Tensor::FromFloat("", {4}, {0.125f, -0.25f, 4, -8}),
                                               MakeQuantizationPlan(QuantizationFormat::kLog, 4))),
               {0.125f, -0.25f, 4, -8});
}

TEST(Quantization, Base3GoldenBytesAndShortTail) {
  auto plan = MakeQuantizationPlan(QuantizationFormat::kTq10, 7);
  auto encoded = QuantizeTensor(Tensor::FromFloat("", {7}, {-1, 0, 1, -1, 1, 1, 0}), plan);
  const auto raw = std::string(encoded.Encoded().raw_data());
  EXPECT_EQ(static_cast<uint8_t>(raw[raw.size() - 2]), 183);
  EXPECT_EQ(static_cast<uint8_t>(raw.back()), 5);
  ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), {-1, 0, 1, -1, 1, 1, 0});
}

TEST(Quantization, AdditiveVectorCodebooksAndPartialVectors) {
  auto plan = MakeQuantizationPlan(QuantizationFormat::kAqlm, 3);
  auto &block = plan.runs[0].blocks[0];
  plan.runs[0].layout.entries = 2;
  plan.runs[0].layout.bits = 1;
  plan.runs[0].layout.vector_size = 2;
  block.codebook = {0, 0, 10, 20, 0, 0, 1, 2};
  const auto source = Tensor::FromFloat("", {3}, {11, 22, 1});
  auto encoded = QuantizeTensor(source, plan);
  EXPECT_EQ(static_cast<uint8_t>(std::string(encoded.Encoded().raw_data()).back()), 11);
  ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), {11, 22, 1});
}

TEST(Quantization, PermutationRotationAndSparseOutliers) {
  auto plan = MakeQuantizationPlan(QuantizationFormat::kQuarot, 4);
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
  for (const auto format :
       {QuantizationFormat::kMxfp4, QuantizationFormat::kNvfp4, QuantizationFormat::kMxfp6,
        QuantizationFormat::kFp6Llm, QuantizationFormat::kFp8E4m3}) {
    SCOPED_TRACE(QuantizationFormatName(format));
    auto plan = MakeQuantizationPlan(format, 6);
    plan.runs[0].blocks[0].scale = 2;
    const auto encoded = QuantizeTensor(Tensor::FromFloat("", {6}, {-12, -3, -1, 0, 1, 12}), plan);
    ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), {-12, -3, -1, 0, 1, 12});
  }
}

TEST(Quantization, PreservesScalarEmptyTensorAndSourceDtypes) {
  for (const Shape &shape : {Shape{}, Shape{0}, Shape{2, 0}}) {
    const size_t count = shape.empty() ? 1 : 0;
    auto plan = MakeQuantizationPlan(QuantizationFormat::kInt8, count);
    auto source =
        Tensor::FromFloat("", shape, count ? std::vector<float>{2} : std::vector<float>{});
    auto decoded = DequantizeTensor(WireRoundTrip(QuantizeTensor(source, plan)));
    EXPECT_EQ(decoded.shape, shape);
    EXPECT_EQ(decoded.element_count(), static_cast<int64_t>(count));
  }
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt8, 3);
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
  auto encoded = QuantizeTensorProto(tensor, MakeQuantizationPlan(QuantizationFormat::kInt8, 4));
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
  auto encoded = QuantizeTensor(Tensor::FromFloat("", {2}, {1, 2}),
                                MakeQuantizationPlan(QuantizationFormat::kInt8, 2))
                     .Encoded();
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
  auto encoded = QuantizeTensor(Tensor::FromFloat("", {3}, {-1, 0, 1}),
                                MakeQuantizationPlan(QuantizationFormat::kInt4, 3))
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
  EXPECT_THROW(MakeQuantizationPlan(static_cast<QuantizationFormat>(-1), 2), std::invalid_argument);
  EXPECT_THROW(MakeQuantizationPlan(QuantizationFormat::kInt4, 2, 0), std::invalid_argument);
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kAqlm, 2)),
               std::invalid_argument);
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kQuarot, 2)),
               std::invalid_argument);
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kInt4, 1)),
               std::invalid_argument);
  for (double scale : {0., -1., std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 2);
    plan.runs[0].blocks[0].scale = scale;
    EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  }
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 2);
  plan.permutation = {0, 0};
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.permutation.clear();
  plan.outliers = {2};
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.outliers.clear();
  plan.runs[0].layout.bits = 0;
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.runs[0].layout.bits = 4;
  plan.runs[0].blocks[0].zero_point = 0.5;
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.runs[0].blocks[0].zero_point = 0;
  plan.transform_size = 2;
  plan.forward = {1, 0, 0, 2};
  plan.inverse = {1, 0, 0, 1};
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
}

TEST(Quantization, RejectsUnknownProfilesInMutablePlansAndEncodedLayouts) {
  const auto source = Tensor::FromFloat("", {2}, {1, 2});
  const auto valid =
      QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kInt4, 2)).Encoded();
  for (const std::string name : {"", "unknown", "QUAROT"}) {
    SCOPED_TRACE(name);
    EXPECT_THROW(ParseQuantizationFormat(name), std::invalid_argument);
    auto encoded = valid;
    encoded.mutable_struct_type()->set_name("onnx_light.quantization.v1/" + name);
    EXPECT_THROW(DequantizeTensor(RuntimeValue(encoded)), std::invalid_argument);
  }
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 2);
  plan.format = static_cast<QuantizationFormat>(43);
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
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
  auto encoded = QuantizeTensorProto(source, MakeQuantizationPlan(QuantizationFormat::kInt4, 2));
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
  const auto encoded =
      QuantizeTensorProto(external, MakeQuantizationPlan(QuantizationFormat::kInt4, 2));
  ExpectValues(DequantizeTensor(RuntimeValue(encoded)), {1, 2});
  EXPECT_EQ(external.SerializeAsString(), original);
  external.set_raw_data(std::string("\0", 1));
  EXPECT_THROW(QuantizeTensorProto(external, MakeQuantizationPlan(QuantizationFormat::kInt4, 2)),
               std::invalid_argument);
}

TEST(Quantization, RejectsCorruptPayloadAndSchema) {
  const auto source = Tensor::FromFloat("", {3}, {1, 0, -1});
  auto valid = QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kInt4, 3)).Encoded();
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
  valid = QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kTq10, 3)).Encoded();
  raw = std::string(valid.raw_data());
  raw.back() = static_cast<char>(243);
  valid.set_raw_data(raw);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(valid)), std::invalid_argument);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(source)), std::invalid_argument);
}

TEST(Quantization, RejectsNonfiniteInputsAndUnloadedExternalData) {
  const auto source = Tensor::FromFloat("", {1}, {std::numeric_limits<float>::quiet_NaN()});
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kInt4, 1)),
               std::invalid_argument);
  TensorProto external;
  external.set_data_type(TensorProto::FLOAT);
  external.add_dims(1);
  external.set_data_location(TensorProto::EXTERNAL);
  EXPECT_THROW(QuantizeTensorProto(external, MakeQuantizationPlan(QuantizationFormat::kInt4, 1)),
               std::invalid_argument);
}

TEST(Quantization, EveryIndexWidthPacksAcrossByteBoundaries) {
  for (uint32_t bits = 1; bits <= 16; ++bits) {
    SCOPED_TRACE(bits);
    auto plan = MakeQuantizationPlan(QuantizationFormat::kExl2, 19);
    auto &block = plan.runs[0].blocks[0];
    plan.runs[0].layout.bits = bits;
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
  auto plan = MakeQuantizationPlan(QuantizationFormat::kNf4, 3);
  const auto encoded = QuantizeTensor(Tensor::FromFloat("", {3}, {-1, 0, 1}), plan);
  const auto raw = std::string(encoded.Encoded().raw_data());
  EXPECT_EQ(static_cast<uint8_t>(raw[raw.size() - 2]), 0x70);
  EXPECT_EQ(static_cast<uint8_t>(raw.back()), 0x0f);
  plan = MakeQuantizationPlan(QuantizationFormat::kBinary, 4);
  const auto binary = QuantizeTensor(Tensor::FromFloat("", {4}, {-1, 1, 0, 1}), plan);
  EXPECT_EQ(static_cast<uint8_t>(std::string(binary.Encoded().raw_data()).back()), 0x0a);
  ExpectValues(DequantizeTensor(binary), {-1, 1, -1, 1});
}

TEST(Quantization, LearnedProfilesAcceptTheirDefaultTableDimensions) {
  for (const auto format : {QuantizationFormat::kStq10, QuantizationFormat::kIq1S,
                            QuantizationFormat::kAqlm, QuantizationFormat::kQuipSharp}) {
    SCOPED_TRACE(QuantizationFormatName(format));
    auto plan = MakeQuantizationPlan(format, 17);
    auto &block = plan.runs[0].blocks[0];
    const auto &layout = plan.runs[0].layout;
    block.codebook.resize(layout.books * layout.entries * layout.vector_size);
    for (uint32_t book = 0; book < layout.books; ++book)
      for (uint32_t entry = 0; entry < layout.entries; ++entry)
        for (uint32_t j = 0; j < layout.vector_size; ++j)
          block.codebook[(book * layout.entries + entry) * layout.vector_size + j] =
              book == 0 ? double(entry) : 0;
    if (format == QuantizationFormat::kQuipSharp) {
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
    auto plan = MakeQuantizationPlan(QuantizationFormat::kTiledFloat, 4);
    plan.runs[0].layout.cast_type = type;
    const auto input = Tensor::FromFloat("", {4}, {-2, 0.5, 1.5, 4});
    ExpectValues(DequantizeTensor(WireRoundTrip(QuantizeTensor(input, plan))), {-2, 0.5, 1.5, 4});
  }
  auto plan = MakeQuantizationPlan(QuantizationFormat::kTiledFloat, 1);
  plan.runs[0].layout.cast_type = TensorProto::FLOAT16;
  EXPECT_THROW(QuantizeTensor(Tensor::FromFloat("", {1}, {1e10f}), plan), std::invalid_argument);
}

TEST(Quantization, RejectsOutOfRangeIndicesAndAlteredDescriptorGeometry) {
  auto valid = QuantizeTensor(Tensor::FromFloat("", {1}, {1}),
                              MakeQuantizationPlan(QuantizationFormat::kTq20, 1))
                   .Encoded();
  auto raw = std::string(valid.raw_data());
  raw.back() = 3;
  valid.set_raw_data(raw);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(valid)), std::invalid_argument);
  valid = QuantizeTensor(Tensor::FromFloat("", {1}, {1}),
                         MakeQuantizationPlan(QuantizationFormat::kInt4, 1))
              .Encoded();
  valid.mutable_logical_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(0)
      ->set_dim_value(2);
  EXPECT_THROW(DequantizeTensor(RuntimeValue(valid)), std::invalid_argument);
}

TEST(Quantization, RepeatedBlockLayoutsShareOneDescriptor) {
  auto small_plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 128, 128);
  auto large_plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 128 * 100, 128);
  ASSERT_EQ(large_plan.runs.size(), 1u);
  EXPECT_EQ(large_plan.runs[0].layout.count, 128u);
  ASSERT_EQ(large_plan.runs[0].blocks.size(), 100u);
  for (size_t i = 0; i < large_plan.runs[0].blocks.size(); ++i)
    large_plan.runs[0].blocks[i].scale = double(i + 1);
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

TEST(Quantization, EnumFormatsHaveStableWireNames) {
  EXPECT_EQ(QuantizationFormatName(QuantizationFormat::kInt8), "int8");
  EXPECT_EQ(QuantizationFormatName(QuantizationFormat::kInt8PerChannel), "int8_per_channel");
  EXPECT_EQ(QuantizationFormatName(QuantizationFormat::kColumnMajor), "column_major");
  for (const auto format : QuantizationFormats()) {
    const auto name = QuantizationFormatName(format);
    EXPECT_EQ(ParseQuantizationFormat(name), format);
    if (format < QuantizationFormat::kOrtMatmulnbitsInt2)
      EXPECT_EQ(MakeQuantizationPlan(format, 0).format, format);
    else
      EXPECT_EQ(MakeMatMulNBitsPlan(format, 1, 1, 16).format, format);
  }
  EXPECT_THROW(QuantizationFormatName(static_cast<QuantizationFormat>(-1)), std::invalid_argument);
  EXPECT_THROW(QuantizationFormatName(static_cast<QuantizationFormat>(43)), std::invalid_argument);
  EXPECT_THROW(ParseQuantizationFormat("INT4"), std::invalid_argument);
}

TEST(Quantization, FactoryRunsMirrorArrayDimensionsAndShortTail) {
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 10, 4);
  ASSERT_EQ(plan.runs.size(), 2u);
  EXPECT_EQ(plan.runs[0].layout.count, 4u);
  ASSERT_EQ(plan.runs[0].blocks.size(), 2u);
  EXPECT_EQ(plan.runs[1].layout.count, 2u);
  ASSERT_EQ(plan.runs[1].blocks.size(), 1u);
  plan.runs[0].blocks[0].scale = 0.5;
  plan.runs[0].blocks[1].scale = 2;
  plan.runs[1].blocks[0].scale = 4;
  const std::vector<float> values{0.5, 1, 1.5, 2, 2, 4, 6, 8, 4, 8};
  const auto encoded = QuantizeTensor(Tensor::FromFloat("", {10}, values), plan);
  const auto &fields =
      encoded.Encoded().struct_type().structure().field(6).type().struct_type().structure().field();
  ASSERT_EQ(fields.size(), 3u);
  EXPECT_EQ(fields[0].constant().int64_data(0), 3);
  EXPECT_EQ(fields[1].name(), "run_0");
  EXPECT_EQ(fields[2].name(), "run_2");
  for (size_t i = 0; i < plan.runs.size(); ++i) {
    const auto &array = fields[i + 1].type().struct_type().array();
    EXPECT_EQ(array.dimension(), plan.runs[i].blocks.size());
    const auto &parameters = array.element_type().struct_type().structure().field(0).constant();
    EXPECT_EQ(parameters.int64_data(0), plan.runs[i].layout.count);
    EXPECT_EQ(parameters.int64_data(2), plan.runs[i].layout.bits);
  }
  ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), values);
  EXPECT_TRUE(MakeQuantizationPlan(QuantizationFormat::kInt4, 0).runs.empty());
  const auto short_plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 1, 4);
  ASSERT_EQ(short_plan.runs.size(), 1u);
  EXPECT_EQ(short_plan.runs[0].layout.count, 1u);
}

TEST(Quantization, AdjacentCompatibleRunsKeepCanonicalWireLayout) {
  auto plan = MakeQuantizationPlan(QuantizationFormat::kBinary, 6, 2);
  plan.runs[0].blocks[1].scale = 2;
  plan.runs[0].blocks[2].codebook = {-3, 3};
  const auto source = Tensor::FromFloat("", {6}, {-1, 1, -2, 2, -3, 3});
  const auto original = QuantizeTensor(source, plan);
  auto second = plan.runs[0];
  second.blocks.erase(second.blocks.begin());
  plan.runs[0].blocks.resize(1);
  plan.runs.push_back(std::move(second));
  const auto split = QuantizeTensor(source, plan);
  EXPECT_EQ(split.Encoded().SerializeAsString(), original.Encoded().SerializeAsString());
  ExpectValues(DequantizeTensor(WireRoundTrip(split)), {-1, 1, -2, 2, -3, 3});
}

TEST(Quantization, RejectsEmptyRunsInvalidLayoutsAndIncompleteCoverage) {
  const auto source = Tensor::FromFloat("", {2}, {1, 2});
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 2);
  plan.runs[0].blocks.clear();
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.runs[0].blocks.resize(1);
  plan.runs[0].layout.count = 0;
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.runs[0].layout.count = 1;
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.runs[0].layout.count = 3;
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.runs[0].layout.count = 2;
  plan.runs[0].layout.method = static_cast<QuantizationMethod>(99);
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
  plan.runs.clear();
  EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
}

TEST(Quantization, OrtMatMulNBitsGoldenInputsAndColumnPadding) {
  for (const auto format :
       {QuantizationFormat::kOrtMatmulnbitsInt2, QuantizationFormat::kOrtMatmulnbitsInt4,
        QuantizationFormat::kOrtMatmulnbitsInt8}) {
    SCOPED_TRACE(QuantizationFormatName(format));
    constexpr size_t k = 35, n = 2, block_size = 16, groups = 3;
    auto plan = MakeMatMulNBitsPlan(format, k, n, block_size);
    ASSERT_EQ(plan.runs.size(), 1u);
    ASSERT_EQ(plan.runs[0].blocks.size(), n * groups);
    const uint32_t bits = plan.runs[0].layout.bits;
    const size_t per_byte = 8 / bits;
    const uint32_t mask = (1u << bits) - 1;
    std::vector<float> values(k * n);
    for (size_t column = 0; column < n; ++column) {
      for (size_t group = 0; group < groups; ++group) {
        auto &block = plan.runs[0].blocks[column * groups + group];
        block.scale = 0.5 * (group + 1);
        block.zero_point = (column + group + 1) & mask;
      }
      for (size_t row = 0; row < k; ++row) {
        const auto &block = plan.runs[0].blocks[column * groups + row / block_size];
        values[row * n + column] =
            float(((row + column) & mask) - block.zero_point) * float(block.scale);
      }
    }
    auto encoded = WireRoundTrip(QuantizeTensor(Tensor::FromFloat("", {k, n}, values), plan));
    auto inputs = ExportMatMulNBitsInputs(encoded.Encoded());
    EXPECT_EQ(inputs.k, k);
    EXPECT_EQ(inputs.n, n);
    EXPECT_EQ(inputs.bits, bits);
    EXPECT_EQ(inputs.block_size, block_size);
    EXPECT_EQ(inputs.weights.data_type(), TensorProto::UINT8);
    ASSERT_EQ(inputs.weights.dims().size(), 3u);
    EXPECT_EQ(inputs.weights.dims(0), n);
    EXPECT_EQ(inputs.weights.dims(1), groups);
    EXPECT_EQ(inputs.weights.dims(2), block_size / per_byte);
    ASSERT_TRUE(inputs.zero_points.has_value());
    EXPECT_EQ(inputs.zero_points->data_type(), TensorProto::UINT8);
    EXPECT_EQ(inputs.zero_points->dims(1), (groups + per_byte - 1) / per_byte);
    std::vector<uint8_t> expected(n * groups * block_size / per_byte, 0);
    for (size_t column = 0; column < n; ++column)
      for (size_t row = 0; row < k; ++row) {
        const size_t index = column * groups * block_size + row;
        expected[index / per_byte] |= uint8_t(((row + column) & mask) << ((row % per_byte) * bits));
      }
    EXPECT_EQ(inputs.weights.raw_data().size(), expected.size());
    EXPECT_EQ(std::memcmp(inputs.weights.raw_data().data(), expected.data(), expected.size()), 0);
    EXPECT_EQ(expected[0], bits == 2 ? 0xe4 : bits == 4 ? 0x10 : 0);
    const auto &zeros = inputs.zero_points->raw_data();
    if (bits == 2) {
      ASSERT_EQ(zeros.size(), 2u);
      EXPECT_EQ(zeros.data()[0], 0x39);
      EXPECT_EQ(zeros.data()[1], 0x0e);
    } else if (bits == 4) {
      ASSERT_EQ(zeros.size(), 4u);
      EXPECT_EQ(zeros.data()[0], 0x21);
      EXPECT_EQ(zeros.data()[1], 0x03);
      EXPECT_EQ(zeros.data()[2], 0x32);
      EXPECT_EQ(zeros.data()[3], 0x04);
    }
    const auto scales = TensorFromProto(inputs.scales);
    ExpectValues(scales, {0.5, 1, 1.5, 0.5, 1, 1.5});
    ExpectValues(DequantizeTensor(encoded), values);
    encoded = RuntimeValue{};
    EXPECT_EQ(std::memcmp(inputs.weights.raw_data().data(), expected.data(), expected.size()), 0);
  }
}

TEST(Quantization, OrtMatMulNBitsImplicitAndFloatingZeroPoints) {
  const auto format = QuantizationFormat::kOrtMatmulnbitsInt4;
  auto plan = MakeMatMulNBitsPlan(format, 3, 2, 16);
  const auto source = Tensor::FromFloat("", {3, 2}, {-1, 1, 0, 2, 1, 3});
  auto encoded = QuantizeTensor(source, plan);
  EXPECT_FALSE(ExportMatMulNBitsInputs(encoded.Encoded()).zero_points.has_value());
  ExpectValues(DequantizeTensor(encoded), {-1, 1, 0, 2, 1, 3});
  plan.runs[0].blocks[0].scale = -0.5;
  plan.runs[0].blocks[0].zero_point = 3.25;
  plan.runs[0].blocks[1].scale = 0.5;
  plan.runs[0].blocks[1].zero_point = 4.5;
  const std::vector<float> values{1.625, -2.25, 0.125, 0.25, -1.875, 2.75};
  encoded = QuantizeTensor(Tensor::FromFloat("", {3, 2}, values), plan);
  auto inputs = ExportMatMulNBitsInputs(encoded.Encoded());
  ASSERT_TRUE(inputs.zero_points.has_value());
  EXPECT_EQ(inputs.zero_points->data_type(), TensorProto::FLOAT);
  ExpectValues(TensorFromProto(*inputs.zero_points), {3.25, 4.5});
  ExpectValues(DequantizeTensor(WireRoundTrip(encoded)), values);
  ModelProto model;
  *model.add_struct_types() = encoded.Encoded().struct_type();
  model.mutable_struct_types(0)->set_type_id(29);
  auto referenced = encoded.Encoded();
  referenced.mutable_struct_type()->Clear();
  referenced.mutable_struct_type()->set_type_ref(29);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  const auto exported = ExportMatMulNBitsInputs(referenced, catalogue);
  EXPECT_EQ(exported.weights.SerializeAsString(), inputs.weights.SerializeAsString());
  SimpleRawBufferAllocator allocator(4);
  ExpectValues(DequantizeTensor(referenced, catalogue, &allocator), values);
}

TEST(Quantization, OrtMatMulNBitsRoundsParametersToSourceDtype) {
  for (int32_t type : {TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    auto plan = MakeMatMulNBitsPlan(QuantizationFormat::kOrtMatmulnbitsInt8, 3, 1, 16);
    plan.runs[0].blocks[0].scale = 0.10001;
    plan.runs[0].blocks[0].zero_point = 128.3;
    const auto source = type == TensorProto::FLOAT16 ? MakeFloat16Tensor("", {3, 1}, {-1, 0, 1})
                                                     : MakeBfloat16Tensor("", {3, 1}, {-1, 0, 1});
    const auto encoded = QuantizeTensor(source, plan);
    const auto inputs = ExportMatMulNBitsInputs(encoded.Encoded());
    EXPECT_EQ(inputs.scales.data_type(), type);
    EXPECT_EQ(DequantizeTensor(encoded).data_type, type);
    const auto expected = type == TensorProto::FLOAT16 ? MakeFloat16Tensor("", {1}, {0.10001f})
                                                       : MakeBfloat16Tensor("", {1}, {0.10001f});
    const auto scales = TensorFromProto(inputs.scales);
    EXPECT_EQ(std::memcmp(scales.bytes(), expected.bytes(), 2), 0);
  }
}

TEST(Quantization, OrtMatMulNBitsRejectsUnsupportedPlansAndMalformedValues) {
  const auto format = QuantizationFormat::kOrtMatmulnbitsInt4;
  EXPECT_THROW(MakeQuantizationPlan(format, 32), std::invalid_argument);
  EXPECT_THROW(MakeMatMulNBitsPlan(QuantizationFormat::kInt4, 2, 3), std::invalid_argument);
  EXPECT_THROW(MakeMatMulNBitsPlan(format, 0, 3), std::invalid_argument);
  for (uint64_t size : {0, 8, 24})
    EXPECT_THROW(MakeMatMulNBitsPlan(format, 2, 3, size), std::invalid_argument);
  auto plan = MakeMatMulNBitsPlan(format, 2, 1, 16);
  const auto source = Tensor::FromFloat("", {2, 1}, {1, 2});
  EXPECT_THROW(QuantizeTensor(Tensor::FromFloat("", {2}, {1, 2}), plan), std::invalid_argument);
  EXPECT_THROW(QuantizeTensor(Tensor::FromDouble("", {2, 1}, {1, 2}), plan), std::invalid_argument);
  auto changed = plan;
  changed.runs[0].layout.bits = 2;
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  changed = plan;
  changed.runs[0].blocks.clear();
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  changed = plan;
  changed.runs[0].blocks[0].offset = 1;
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  changed = plan;
  changed.outliers = {0};
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  changed = plan;
  changed.matrix_shape = {1, 1};
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  changed = plan;
  changed.runs[0].blocks[0].scale = 0;
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  const auto zero = QuantizeTensor(Tensor::FromFloat("", {2, 1}, {0, 0}), changed);
  ExpectValues(DequantizeTensor(zero), {0, 0});
  changed.runs[0].blocks[0].scale = 1e-100;
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  changed.runs[0].blocks[0].scale = std::numeric_limits<double>::infinity();
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  changed = plan;
  changed.runs[0].blocks[0].zero_point = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(QuantizeTensor(source, changed), std::invalid_argument);
  EXPECT_THROW(
      QuantizeTensor(Tensor::FromFloat("", {2, 1}, {1, std::numeric_limits<float>::infinity()}),
                     plan),
      std::invalid_argument);
  const auto valid = QuantizeTensor(source, plan).Encoded();
  auto corrupt = valid;
  corrupt.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(0)
      ->mutable_constant()
      ->ref_int64_data()[0] = 2;
  EXPECT_THROW(DequantizeTensor(corrupt), std::invalid_argument);
  EXPECT_THROW(ExportMatMulNBitsInputs(corrupt), std::invalid_argument);
  corrupt = valid;
  auto raw = std::string(corrupt.raw_data());
  raw.pop_back();
  corrupt.set_raw_data(raw);
  EXPECT_THROW(DequantizeTensor(corrupt), std::invalid_argument);
  corrupt = valid;
  raw = std::string(corrupt.raw_data());
  raw.replace(raw.size() - 4, 4, "\0\0\xc0\x7f", 4);
  corrupt.set_raw_data(raw);
  EXPECT_THROW(DequantizeTensor(corrupt), std::invalid_argument);
  EXPECT_THROW(ExportMatMulNBitsInputs(corrupt), std::invalid_argument);
  corrupt = valid;
  corrupt.mutable_struct_type()->mutable_structure()->mutable_field(1)->set_name("weights");
  EXPECT_THROW(ExportMatMulNBitsInputs(corrupt), std::invalid_argument);
  const auto portable =
      QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kMatmulnbits, 2));
  EXPECT_THROW(ExportMatMulNBitsInputs(portable.Encoded()), std::invalid_argument);
}
