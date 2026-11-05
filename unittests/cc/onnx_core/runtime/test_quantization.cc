// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include "onnx_core/runtime/runtime_session.h"
#include <bit>
#include <cmath>
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

constexpr auto kQuantizationFormats = QuantizationFormats();
static_assert(kQuantizationFormats.size() == 44);
static_assert(kQuantizationFormats.front() == QuantizationFormat::kInt8);
static_assert(kQuantizationFormats.back() == QuantizationFormat::kMxfp8);
static_assert(static_cast<int>(QuantizationFormat::kOrtMatmulnbitsInt8) == 42);
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

ModelProto SharedModel(const QuantizationPlan &plan, const EncodedValueProto &encoded) {
  ModelProto model;
  auto *graph = model.mutable_graph();
  auto *annotation = graph->add_quantization_annotation();
  annotation->set_tensor_name("onnx_light.quantization.parameters:weights");
  auto add = [&](const std::string &role, int32_t type, const std::vector<int64_t> &shape,
                 const void *data, size_t bytes) {
    auto *tensor = graph->add_initializer();
    tensor->set_name(role);
    tensor->set_data_type(type);
    for (auto dim : shape)
      tensor->add_dims(dim);
    tensor->set_raw_data(data, bytes);
    auto *mapping = annotation->add_quant_parameter_tensor_names();
    mapping->set_key(role);
    mapping->set_value(role);
  };
  auto schema = encoded.struct_type().SerializeAsString();
  auto logical = encoded.logical_type().SerializeAsString();
  add("storage_type", TensorProto::UINT8, {static_cast<int64_t>(schema.size())}, schema.data(),
      schema.size());
  add("logical_type", TensorProto::UINT8, {static_cast<int64_t>(logical.size())}, logical.data(),
      logical.size());
  std::vector<double> scales, zeros, offsets, tables;
  for (const auto &run : plan.runs)
    for (const auto &block : run.blocks) {
      scales.push_back(block.scale);
      zeros.push_back(block.zero_point);
      offsets.push_back(block.offset);
      tables.insert(tables.end(), block.codebook.begin(), block.codebook.end());
    }
  auto floating = [&](const std::string &role, const std::vector<double> &values,
                      const std::vector<int64_t> &shape) {
    if (!values.empty())
      add(role, TensorProto::DOUBLE, shape, values.data(), values.size() * sizeof(double));
  };
  if (scales.empty())
    floating("scales", {1}, {});
  else
    floating("scales", scales, {static_cast<int64_t>(scales.size())});
  floating("zero_points", zeros, {static_cast<int64_t>(zeros.size())});
  if (plan.format < QuantizationFormat::kOrtMatmulnbitsInt2)
    floating("offsets", offsets, {static_cast<int64_t>(offsets.size())});
  floating("codebooks", tables, {static_cast<int64_t>(tables.size())});
  floating("forward", plan.forward, {plan.transform_size, plan.transform_size});
  floating("inverse", plan.inverse, {plan.transform_size, plan.transform_size});
  if (!plan.permutation.empty())
    add("permutation", TensorProto::INT64, {static_cast<int64_t>(plan.permutation.size())},
        plan.permutation.data(), plan.permutation.size() * sizeof(int64_t));
  if (!plan.outliers.empty())
    add("outliers", TensorProto::INT64, {static_cast<int64_t>(plan.outliers.size())},
        plan.outliers.data(), plan.outliers.size() * sizeof(int64_t));
  return model;
}

} // namespace

TEST(Quantization, Mxfp4WeightsPinnedPackingAndLosslessImport) {
  // Source-derived from compressed-tensors at
  // eb058d630f7344099459b6795b1e3eeffdda4b40.
  const std::vector<float> values{0,    .5f,  1,    1.5f,  2,    3,    4,    6,
                                  -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  auto row = values;
  row.insert(row.end(), values.begin(), values.end());
  const auto encoded = QuantizeMxfp4Weights(Tensor::FromFloat("weight", {1, 32}, row));
  EXPECT_EQ(encoded.struct_type().name().value(), "onnx_light.mxfp4_weights.v1");
  const auto weights = ExportMxfp4Weights(encoded);
  const std::vector<uint8_t> expected{0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe,
                                      0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe};
  EXPECT_EQ(std::vector<uint8_t>(weights.weight_packed.bytes(),
                                 weights.weight_packed.bytes() + expected.size()),
            expected);
  ASSERT_EQ(weights.weight_scale.size_bytes(), 1u);
  EXPECT_EQ(weights.weight_scale.bytes()[0], 127);
  ExpectValues(DequantizeTensor(WireRoundTrip(RuntimeValue(encoded))), row);
  EXPECT_EQ(ImportMxfp4Weights(weights).raw_data(), encoded.raw_data());

  const auto unit = ExportMxfp4Weights(
      QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, std::vector<float>(32, 1.f))));
  EXPECT_EQ(unit.weight_scale.bytes()[0], 125);
  EXPECT_EQ(std::vector<uint8_t>(unit.weight_packed.bytes(),
                                 unit.weight_packed.bytes() + unit.weight_packed.size_bytes()),
            std::vector<uint8_t>(16, 0x66));
}

TEST(Quantization, Mxfp4WeightsPinnedScaleRoundingBoundary) {
  // Source-derived from quantization/utils/mxfp_utils.py and compressors/mxfp4 at
  // compressed-tensors revision eb058d630f7344099459b6795b1e3eeffdda4b40.
  struct Fixture {
    float maximum;
    uint8_t scale;
    std::vector<uint8_t> packed;
    float reconstructed_maximum;
  };
  const Fixture fixtures[]{
      {std::nextafter(7.f, 0.f), 127, {0xf7, 0xa2, 0xc4, 0xd5}, 6.f},
      {7.f, 128, {0xe6, 0x91, 0xa2, 0xb3}, 8.f},
      {std::nextafter(7.f, 8.f), 128, {0xe6, 0x91, 0xa2, 0xb3}, 8.f},
  };
  for (const auto &fixture : fixtures) {
    SCOPED_TRACE(fixture.maximum);
    std::vector<float> values{fixture.maximum, -fixture.maximum, 1, -1, 2, -2, 3, -3};
    values.resize(32, 0.f);
    const auto encoded = QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, values));
    const auto weights = ExportMxfp4Weights(encoded);
    ASSERT_EQ(weights.weight_scale.size_bytes(), 1u);
    EXPECT_EQ(weights.weight_scale.bytes()[0], fixture.scale);
    auto expected_packed = fixture.packed;
    expected_packed.resize(16, 0);
    ASSERT_EQ(weights.weight_packed.size_bytes(), expected_packed.size());
    EXPECT_EQ(std::vector<uint8_t>(weights.weight_packed.bytes(),
                                   weights.weight_packed.bytes() + expected_packed.size()),
              expected_packed);
    values[0] = fixture.reconstructed_maximum;
    values[1] = -fixture.reconstructed_maximum;
    ExpectValues(DequantizeTensor(WireRoundTrip(RuntimeValue(encoded))), values);
  }
}

TEST(Quantization, Mxfp4WeightsTiesSaturationAndZeroGroups) {
  std::vector<float> values{.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f, -.25f, -0.f, 6.f};
  values.resize(32, 0.f);
  auto encoded = QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, values));
  const auto weights = ExportMxfp4Weights(encoded);
  const std::vector<uint8_t> expected{0x20, 0x42, 0x64, 0x86, 0x70};
  EXPECT_EQ(std::vector<uint8_t>(weights.weight_packed.bytes(),
                                 weights.weight_packed.bytes() + expected.size()),
            expected);
  EXPECT_EQ(weights.weight_scale.bytes()[0], 127);
  const auto decoded = DequantizeTensor(encoded);
  EXPECT_FALSE(std::signbit(decoded.AsFloat()[8]));
  EXPECT_TRUE(std::signbit(decoded.AsFloat()[7]));

  values.assign(32, 0.f);
  encoded = QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, values));
  const auto zero_weights = ExportMxfp4Weights(encoded);
  EXPECT_EQ(zero_weights.weight_scale.bytes()[0], 127);
  EXPECT_EQ(std::vector<uint8_t>(zero_weights.weight_packed.bytes(),
                                 zero_weights.weight_packed.bytes() +
                                     zero_weights.weight_packed.size_bytes()),
            std::vector<uint8_t>(16, 0));
  ExpectValues(DequantizeTensor(encoded), values);

  values[0] = 100.f;
  values[1] = -100.f;
  values[2] = -std::numeric_limits<float>::denorm_min();
  encoded = QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, values));
  const auto saturated_weights = ExportMxfp4Weights(encoded);
  EXPECT_EQ(saturated_weights.weight_packed.bytes()[0], 0xf7);
  EXPECT_EQ(saturated_weights.weight_packed.bytes()[1], 0x00);

  values.assign(32, 0.f);
  values[0] = 6.f * std::ldexp(1.f, -127);
  encoded = QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, values));
  EXPECT_EQ(ExportMxfp4Weights(encoded).weight_scale.bytes()[0], 0);
  EXPECT_EQ(ExportMxfp4Weights(encoded).weight_packed.bytes()[0], 7);
  ExpectValues(DequantizeTensor(encoded), values);

  values.assign(32, 0.f);
  values[0] = std::numeric_limits<float>::max();
  encoded = QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, values));
  EXPECT_EQ(ExportMxfp4Weights(encoded).weight_scale.bytes()[0], 252);
  EXPECT_TRUE(std::isfinite(DequantizeTensor(encoded).AsFloat()[0]));
}

TEST(Quantization, Mxfp4WeightsPinnedRowMajorGroups) {
  // Source-derived from compressed-tensors revision eb058d630f7344099459b6795b1e3eeffdda4b40.
  std::vector<float> values, reconstructed;
  for (float value : {1.f, -7.f, 2.f, 14.f})
    values.insert(values.end(), 32, value);
  for (float value : {1.f, -8.f, 2.f, 16.f})
    reconstructed.insert(reconstructed.end(), 32, value);
  const auto encoded = QuantizeMxfp4Weights(Tensor::FromFloat("", {2, 64}, values));
  const auto weights = ExportMxfp4Weights(encoded);
  EXPECT_EQ(weights.weight_packed.shape, (Shape{2, 32}));
  EXPECT_EQ(weights.weight_scale.shape, (Shape{2, 2}));
  const std::vector<uint8_t> scales{125, 128, 126, 129};
  EXPECT_EQ(std::vector<uint8_t>(weights.weight_scale.bytes(),
                                 weights.weight_scale.bytes() + scales.size()),
            scales);
  std::vector<uint8_t> packed;
  for (uint8_t code : {0x66, 0xee, 0x66, 0x66})
    packed.insert(packed.end(), 16, code);
  EXPECT_EQ(std::vector<uint8_t>(weights.weight_packed.bytes(),
                                 weights.weight_packed.bytes() + packed.size()),
            packed);
  const auto imported = WireRoundTrip(RuntimeValue(ImportMxfp4Weights(weights))).Encoded();
  EXPECT_EQ(imported.raw_data(), encoded.raw_data());
  ExpectValues(DequantizeTensor(imported), reconstructed);
  ExpectValues(TensorFromProto(DequantizeTensorProto(imported)), reconstructed);
}

TEST(Quantization, Mxfp4WeightsRejectsInvalidLayoutsAndScaleBytes) {
  EXPECT_THROW(QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 31}, std::vector<float>(31))),
               std::invalid_argument);
  EXPECT_THROW(QuantizeMxfp4Weights(Tensor::FromFloat("", {0, 32}, {})), std::invalid_argument);
  auto nonfinite = Tensor::FromFloat("", {1, 32}, std::vector<float>(32));
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(nonfinite.mutable_bytes(), &nan, sizeof(nan));
  EXPECT_THROW(QuantizeMxfp4Weights(nonfinite), std::invalid_argument);

  auto weights = ExportMxfp4Weights(
      QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, std::vector<float>(32))));
  weights.weight_scale.mutable_bytes()[0] = 0xff;
  EXPECT_THROW(ImportMxfp4Weights(weights), std::invalid_argument);
}

TEST(Quantization, Mxfp4WeightsRejectsMalformedStorageAndOverflow) {
  const auto encoded = QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, std::vector<float>(32)));
  auto rejects = [](const EncodedValueProto &value) {
    EXPECT_THROW(ExportMxfp4Weights(value), std::invalid_argument);
    EXPECT_THROW(DequantizeTensor(value), std::invalid_argument);
    EXPECT_THROW(DequantizeTensorProto(value), std::invalid_argument);
  };
  auto malformed = encoded;
  malformed.mutable_raw_data()->resize(16);
  rejects(malformed);
  malformed = encoded;
  malformed.mutable_struct_type()->set_name("onnx_light.mxfp4_weights.v2");
  rejects(malformed);
  malformed = encoded;
  malformed.mutable_logical_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(1)
      ->set_dim_value(64);
  rejects(malformed);
  malformed = encoded;
  malformed.mutable_raw_data()->data()[16] = 0xff;
  rejects(malformed);
  malformed.mutable_raw_data()->data()[16] = 254;
  malformed.mutable_raw_data()->data()[0] = 0x77;
  rejects(malformed);

  auto weights = ExportMxfp4Weights(encoded);
  weights.weight_scale = Tensor("weight_scale", TensorProto::UINT8, {1, 2}, {127, 127});
  EXPECT_THROW(ImportMxfp4Weights(weights), std::invalid_argument);
  weights.weight_scale = Tensor("weight_scale", TensorProto::INT8, {1, 1}, {127});
  EXPECT_THROW(ImportMxfp4Weights(weights), std::invalid_argument);
  weights.weight_packed.shape[1] = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(ImportMxfp4Weights(weights), std::invalid_argument);
  EXPECT_THROW(QuantizeMxfp4Weights(
                   Tensor("", TensorProto::FLOAT, {std::numeric_limits<int64_t>::max(), 32}, {})),
               std::invalid_argument);
  for (float value :
       {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
    EXPECT_THROW(
        QuantizeMxfp4Weights(Tensor::FromFloat("", {1, 32}, std::vector<float>(32, value))),
        std::invalid_argument);
}

TEST(Quantization, Mxfp8WeightsPinnedRoundingBoundaryAndRows) {
  // Source-derived from compressed-tensors eb058d630f7344099459b6795b1e3eeffdda4b40.
  // generate_mx_scales(num_bits=8) retains the FP4 mantissa-one rounding threshold.
  std::vector<float> values, expected;
  for (float value : {std::nextafter(7.f, 0.f), 7.f, -1.f, 2.f})
    values.insert(values.end(), 32, value);
  for (float value : {7.f, 7.f, -1.f, 2.f})
    expected.insert(expected.end(), 32, value);
  const auto source = Tensor::FromFloat("weight", {2, 64}, values);
  const auto encoded = QuantizeMxfp8Weights(source);
  const auto weights = ExportMxfp8Weights(encoded);
  EXPECT_EQ(weights.weight.data_type, TensorProto::FLOAT8E4M3FN);
  EXPECT_EQ(weights.weight.shape, (Shape{2, 64}));
  EXPECT_EQ(weights.weight_scale.data_type, TensorProto::UINT8);
  EXPECT_EQ(weights.weight_scale.shape, (Shape{2, 2}));
  const std::vector<uint8_t> scales{121, 122, 119, 120};
  EXPECT_EQ(std::vector<uint8_t>(weights.weight_scale.bytes(), weights.weight_scale.bytes() + 4),
            scales);
  std::vector<uint8_t> codes;
  for (uint8_t code : {0x7e, 0x76, 0xf8, 0x78})
    codes.insert(codes.end(), 32, code);
  EXPECT_EQ(std::vector<uint8_t>(weights.weight.bytes(), weights.weight.bytes() + 128), codes);
  ExpectValues(DequantizeTensor(WireRoundTrip(RuntimeValue(encoded))), expected);
  ExpectValues(TensorFromProto(DequantizeTensorProto(encoded)), expected);
  const auto imported = ImportMxfp8Weights(weights);
  EXPECT_EQ(imported.raw_data(), encoded.raw_data());
  EXPECT_EQ(imported.struct_type().SerializeAsString(), encoded.struct_type().SerializeAsString());
  EXPECT_EQ(imported.logical_type().SerializeAsString(),
            encoded.logical_type().SerializeAsString());
  const auto plan = MakeQuantizationPlan(QuantizationFormat::kMxfp8, values.size());
  EXPECT_EQ(QuantizeTensor(source, plan).Encoded().raw_data(), encoded.raw_data());
  TensorProto proto;
  proto.set_name("weight");
  proto.set_doc_string("native MXFP8");
  proto.set_data_type(TensorProto::FLOAT);
  proto.add_dims(2);
  proto.add_dims(64);
  for (float value : values)
    proto.add_float_data(value);
  const auto encoded_proto = QuantizeTensorProto(proto, plan);
  EXPECT_EQ(encoded_proto.raw_data(), encoded.raw_data());
  EXPECT_EQ(encoded_proto.doc_string().value(), "native MXFP8");
  EXPECT_EQ(DequantizeTensorProto(encoded_proto).doc_string().value(), "native MXFP8");
}

TEST(Quantization, Mxfp8WeightsFullBlockSourceDerivedFixture) {
  // Source-derived from compressed-tensors eb058d630f7344099459b6795b1e3eeffdda4b40.
  const std::vector<float> block{
      432,     -432,    1.0625f,  1.1875f,   1.3125f,   1.4375f,   1.5625f,   1.6875f,
      1.8125f, 1.9375f, 0x1p-10f, 0x1.8p-9f, 0x1.4p-8f, 0x1.cp-8f, 0x1.ep-7f, 0x1p-6f,
      0,       -0.f,    -1.0625f, -1.1875f,  .5f,       -.5f,      2,         -2,
      3,       -3,      6,        -6,        16,        -16,       240,       -240};
  const std::vector<uint8_t> codes{0x7e, 0xfe, 0x38, 0x3a, 0x3a, 0x3c, 0x3c, 0x3e, 0x3e, 0x40, 0,
                                   2,    2,    4,    8,    8,    0,    0x80, 0xb8, 0xba, 0x30, 0xb0,
                                   0x40, 0xc0, 0x44, 0xc4, 0x4c, 0xcc, 0x58, 0xd8, 0x77, 0xf7};
  const std::vector<float> reconstructed{
      448,     -448,    1,       1.25f,   1.25f,   1.5f, 1.5f, 1.75f, 1.75f,  2,   0,
      0x1p-8f, 0x1p-8f, 0x1p-7f, 0x1p-6f, 0x1p-6f, 0,    -0.f, -1,    -1.25f, .5f, -.5f,
      2,       -2,      3,       -3,      6,       -6,   16,   -16,   240,    -240};
  std::vector<float> values = block, expected = reconstructed;
  for (float value : block)
    values.push_back(value * 0x1p-8f);
  for (float value : reconstructed)
    expected.push_back(value * 0x1p-8f);
  const auto encoded = QuantizeMxfp8Weights(Tensor::FromFloat("", {2, 32}, values));
  const auto weights = ExportMxfp8Weights(encoded);
  EXPECT_EQ(weights.weight_scale.bytes()[0], 127);
  EXPECT_EQ(weights.weight_scale.bytes()[1], 119);
  for (size_t group = 0; group < 2; ++group)
    EXPECT_EQ(std::vector<uint8_t>(weights.weight.bytes() + group * 32,
                                   weights.weight.bytes() + (group + 1) * 32),
              codes);
  ExpectValues(DequantizeTensor(WireRoundTrip(RuntimeValue(encoded))), expected);
}

TEST(Quantization, Mxfp8WeightsNativeScaleExtremesAndCatalogueReferences) {
  std::vector<uint8_t> codes(64, 0);
  codes[0] = 1;
  codes[32] = 0x38;
  const Mxfp8Weights native{Tensor("weight", TensorProto::FLOAT8E4M3FN, {2, 32}, codes),
                            Tensor("weight_scale", TensorProto::UINT8, {2, 1}, {0, 254})};
  auto encoded = ImportMxfp8Weights(native);
  std::vector<float> expected(64, 0);
  expected[0] = 0x1p-136f;
  expected[32] = 0x1p127f;
  ExpectValues(DequantizeTensor(encoded), expected);
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(83);
  encoded.mutable_struct_type()->Clear();
  encoded.mutable_struct_type()->set_type_ref(83);
  const auto parsed = WireRoundTrip(RuntimeValue(encoded)).Encoded();
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  ExpectValues(DequantizeTensor(parsed, catalogue), expected);
  ExpectValues(TensorFromProto(DequantizeTensorProto(parsed, catalogue)), expected);
  const auto exported = ExportMxfp8Weights(parsed, catalogue);
  EXPECT_EQ(std::memcmp(exported.weight.bytes(), codes.data(), codes.size()), 0);
  EXPECT_EQ(exported.weight_scale.bytes()[0], 0);
  EXPECT_EQ(exported.weight_scale.bytes()[1], 254);
  EXPECT_THROW(ExportMxfp8Weights(parsed), std::invalid_argument);
}

TEST(Quantization, Mxfp8WeightsNearestEvenAndSignedZero) {
  std::vector<float> values{256,      1.0625f,   1.1875f,   -1.0625f, -1.1875f,
                            0x1p-10f, 0x1.8p-9f, -0x1p-10f, -0.f,     0x1p-9f};
  values.resize(32, 0);
  const auto encoded = QuantizeMxfp8Weights(Tensor::FromFloat("", {1, 32}, values));
  const auto weights = ExportMxfp8Weights(encoded);
  EXPECT_EQ(weights.weight_scale.bytes()[0], 127);
  const std::vector<uint8_t> codes{0x78, 0x38, 0x3a, 0xb8, 0xba, 0, 2, 0x80, 0x80, 1};
  EXPECT_EQ(std::vector<uint8_t>(weights.weight.bytes(), weights.weight.bytes() + codes.size()),
            codes);
  std::vector<float> expected{256, 1, 1.25f, -1, -1.25f, 0, 0x1p-8f, -0.f, -0.f, 0x1p-9f};
  expected.resize(32, 0);
  const auto decoded = DequantizeTensor(encoded);
  ExpectValues(decoded, expected);
  EXPECT_TRUE(std::signbit(decoded.AsFloat()[7]));
  EXPECT_TRUE(std::signbit(decoded.AsFloat()[8]));
}

TEST(Quantization, Mxfp8WeightsScaleExtensionsAndFiniteMaximum) {
  struct Fixture {
    float source;
    uint8_t scale, code;
    float reconstructed;
  };
  const Fixture fixtures[]{
      {0, 127, 0, 0},
      {std::numeric_limits<float>::denorm_min(), 0, 0, 0},
      {std::bit_cast<float>(uint32_t{0x007fffff}), 0, 0x40, 0x1p-126f},
      {std::numeric_limits<float>::min(), 0, 0x40, 0x1p-126f},
      {0x1p-119f, 0, 0x78, 0x1p-119f},
      {0x1p-118f, 1, 0x78, 0x1p-118f},
      {0x1p126f, 245, 0x78, 0x1p126f},
      {std::numeric_limits<float>::max(), 246, 0x7e, 448 * 0x1p119f},
  };
  for (const auto &fixture : fixtures) {
    SCOPED_TRACE(fixture.source);
    const auto encoded = QuantizeMxfp8Weights(
        Tensor::FromFloat("", {1, 32}, std::vector<float>(32, fixture.source)));
    const auto weights = ExportMxfp8Weights(encoded);
    EXPECT_EQ(weights.weight_scale.bytes()[0], fixture.scale);
    for (size_t j = 0; j < 32; ++j)
      EXPECT_EQ(weights.weight.bytes()[j], fixture.code);
    const auto decoded = DequantizeTensor(encoded);
    for (size_t j = 0; j < 32; ++j) {
      EXPECT_EQ(decoded.AsFloat()[j], fixture.reconstructed);
      EXPECT_TRUE(std::isfinite(decoded.AsFloat()[j]));
    }
  }
}

TEST(Quantization, Mxfp8WeightsIndependentNativeImportPreservesAllFiniteCodes) {
  std::vector<uint8_t> codes;
  for (unsigned code = 0; code < 256; ++code)
    if (code != 127 && code != 255)
      codes.push_back(static_cast<uint8_t>(code));
  codes.resize(256, 0);
  Mxfp8Weights native{
      Tensor("weight", TensorProto::FLOAT8E4M3FN, {2, 128}, codes),
      Tensor("weight_scale", TensorProto::UINT8, {2, 4}, std::vector<uint8_t>(8, 127))};
  const auto encoded = ImportMxfp8Weights(native);
  const auto parsed = WireRoundTrip(RuntimeValue(encoded)).Encoded();
  const auto exported = ExportMxfp8Weights(parsed);
  EXPECT_EQ(std::memcmp(exported.weight.bytes(), codes.data(), codes.size()), 0);
  EXPECT_EQ(ImportMxfp8Weights(exported).raw_data(), encoded.raw_data());
  const auto decoded = DequantizeTensor(parsed);
  EXPECT_EQ(decoded.AsFloat()[0], 0);
  EXPECT_EQ(decoded.AsFloat()[1], 0x1p-9f);
  EXPECT_EQ(decoded.AsFloat()[126], 448);
  EXPECT_TRUE(std::signbit(decoded.AsFloat()[127]));
  EXPECT_EQ(decoded.AsFloat()[253], -448);
  native.weight.mutable_bytes()[0] = 0x7e;
  EXPECT_EQ(exported.weight.bytes()[0], 0); // Exported tensors own their storage.
}

TEST(Quantization, Mxfp8WeightsRejectsInvalidSourcesAndPlans) {
  const auto source = Tensor::FromFloat("", {1, 32}, std::vector<float>(32, 1));
  const auto plan = MakeQuantizationPlan(QuantizationFormat::kMxfp8, 32);
  EXPECT_EQ(ParseQuantizationFormat("mxfp8"), QuantizationFormat::kMxfp8);
  for (uint64_t count : {0, 1, 31, 33})
    EXPECT_THROW(MakeQuantizationPlan(QuantizationFormat::kMxfp8, count), std::invalid_argument);
  EXPECT_NO_THROW(MakeQuantizationPlan(QuantizationFormat::kMxfp8, 32, 32));
  EXPECT_THROW(MakeQuantizationPlan(QuantizationFormat::kMxfp8, 32, 64), std::invalid_argument);
  EXPECT_THROW(MakeQuantizationType(plan), std::invalid_argument);
  for (const Shape &shape : {Shape{32}, Shape{1, 1, 32}, Shape{0, 32}, Shape{1, 31}, Shape{-1, 32},
                             Shape{std::numeric_limits<int64_t>::max(), 32}})
    EXPECT_THROW(QuantizeMxfp8Weights(Tensor("", TensorProto::FLOAT, shape, {})),
                 std::invalid_argument);
  EXPECT_THROW(QuantizeMxfp8Weights(Tensor::FromDouble("", {1, 32}, std::vector<double>(32))),
               std::invalid_argument);
  EXPECT_THROW(QuantizeMxfp8Weights(Tensor("", TensorProto::FLOAT, {1, 32}, {})),
               std::invalid_argument);
  for (float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()})
    EXPECT_THROW(QuantizeTensor(Tensor::FromFloat("", {1, 32}, std::vector<float>(32, bad)), plan),
                 std::invalid_argument);
  TensorProto proto;
  proto.set_data_type(TensorProto::FLOAT);
  proto.add_dims(1);
  proto.add_dims(32);
  for (int j = 0; j < 32; ++j)
    proto.add_float_data(1);
  auto rejects = [&](const QuantizationPlan &bad) {
    EXPECT_THROW(QuantizeTensor(source, bad), std::invalid_argument);
    EXPECT_THROW(QuantizeTensorProto(proto, bad), std::invalid_argument);
  };
  auto bad = plan;
  bad.runs.clear();
  rejects(bad);
  bad = plan;
  bad.runs.push_back(plan.runs[0]);
  rejects(bad);
  bad = plan;
  bad.runs[0].blocks.clear();
  rejects(bad);
  bad = plan;
  bad.runs[0].blocks[0].scale = 2;
  rejects(bad);
  bad = plan;
  bad.runs[0].blocks[0].zero_point = 1;
  rejects(bad);
  bad = plan;
  bad.runs[0].blocks[0].offset = 1;
  rejects(bad);
  bad = plan;
  bad.runs[0].blocks[0].codebook = {0};
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.count = 16;
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.cast_type = TensorProto::FLOAT;
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.bits = 4;
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.method = QuantizationMethod::kAffine;
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.signed_codes = false;
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.books = 2;
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.entries = 1;
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.vector_size = 2;
  rejects(bad);
  bad = plan;
  bad.runs[0].layout.base3 = true;
  rejects(bad);
  bad = plan;
  bad.reconstruction_dtype = TensorProto::DOUBLE;
  rejects(bad);
  bad = plan;
  bad.matrix_shape = {1, 32};
  rejects(bad);
  bad = plan;
  bad.permutation = {0};
  rejects(bad);
  bad = plan;
  bad.transform_size = 1;
  rejects(bad);
  bad = plan;
  bad.forward = {1};
  rejects(bad);
  bad = plan;
  bad.inverse = {1};
  rejects(bad);
  bad = plan;
  bad.outliers = {0};
  rejects(bad);
  EXPECT_THROW(QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kMxfp8, 64)),
               std::invalid_argument);
}

TEST(Quantization, Mxfp8WeightsRejectsMalformedSchemaScalesAndStorage) {
  const auto encoded =
      QuantizeMxfp8Weights(Tensor::FromFloat("", {1, 32}, std::vector<float>(32, 1)));
  auto rejects = [](const EncodedValueProto &bad) {
    EXPECT_THROW(ExportMxfp8Weights(bad), std::invalid_argument);
    EXPECT_THROW(DequantizeTensor(bad), std::invalid_argument);
    EXPECT_THROW(DequantizeTensorProto(bad), std::invalid_argument);
  };
  auto bad = encoded;
  bad.mutable_raw_data()->resize(32);
  rejects(bad);
  bad = encoded;
  bad.mutable_raw_data()->resize(34);
  rejects(bad);
  bad = encoded;
  bad.mutable_struct_type()->set_name("onnx_light.mxfp8_weights.v2");
  rejects(bad);
  for (int field = 0; field < 7; ++field) {
    bad = encoded;
    bad.mutable_struct_type()
        ->mutable_structure()
        ->mutable_field(field)
        ->mutable_constant()
        ->ref_int64_data()[0] = -1;
    rejects(bad);
  }
  bad = encoded;
  bad.mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
  rejects(bad);
  bad = encoded;
  bad.mutable_logical_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_value(
      64);
  rejects(bad);
  bad = encoded;
  bad.set_parameter_ref("unsupported");
  rejects(bad);
  for (uint8_t code : {127, 255}) {
    bad = encoded;
    bad.mutable_raw_data()->data()[0] = code;
    rejects(bad);
  }
  bad = encoded;
  bad.mutable_raw_data()->data()[32] = 255;
  rejects(bad);
  bad = encoded;
  bad.mutable_raw_data()->data()[32] = 254;
  bad.mutable_raw_data()->data()[0] = 0x7e;
  rejects(bad);
  auto native = ExportMxfp8Weights(encoded);
  native.weight_scale.mutable_bytes()[0] = 255;
  EXPECT_THROW(ImportMxfp8Weights(native), std::invalid_argument);
  native = ExportMxfp8Weights(encoded);
  native.weight.mutable_bytes()[0] = 127;
  EXPECT_THROW(ImportMxfp8Weights(native), std::invalid_argument);
  native = ExportMxfp8Weights(encoded);
  native.weight_scale.data_type = TensorProto::INT8;
  EXPECT_THROW(ImportMxfp8Weights(native), std::invalid_argument);
  native = ExportMxfp8Weights(encoded);
  native.weight.data_type = TensorProto::UINT8;
  EXPECT_THROW(ImportMxfp8Weights(native), std::invalid_argument);
  native = ExportMxfp8Weights(encoded);
  native.weight_scale.shape = {1, 2};
  EXPECT_THROW(ImportMxfp8Weights(native), std::invalid_argument);
  native = ExportMxfp8Weights(encoded);
  native.weight.shape = {32};
  EXPECT_THROW(ImportMxfp8Weights(native), std::invalid_argument);
}

TEST(Quantization, Nvfp4WeightsGoldenPackingAndLosslessImport) {
  const std::vector<float> values{0,    .5f,  1,  1.5f,  2,  3,  4,  6,
                                  -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
  const auto source = Tensor::FromFloat("weight", {1, 16}, values);
  const auto encoded = QuantizeNvfp4Weights(source, Tensor::FromFloat("", {1, 1}, {2}), 2);
  EXPECT_EQ(encoded.struct_type().name().value(), "onnx_light.nvfp4_weights.v1");
  EXPECT_EQ(encoded.logical_type().tensor_type().elem_type(), TensorProto::FLOAT);
  const auto weights = ExportNvfp4Weights(encoded);
  const std::vector<uint8_t> expected{0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe};
  ASSERT_EQ(weights.weight_packed.size_bytes(), expected.size());
  EXPECT_EQ(std::vector<uint8_t>(weights.weight_packed.bytes(),
                                 weights.weight_packed.bytes() + expected.size()),
            expected);
  ASSERT_EQ(weights.weight_scale.size_bytes(), 1u);
  EXPECT_EQ(weights.weight_scale.bytes()[0], 0x40);
  EXPECT_EQ(weights.weight_global_scale, 2);
  ExpectValues(DequantizeTensor(encoded), values);
  ExpectValues(TensorFromProto(DequantizeTensorProto(encoded)), values);
  const auto imported = ImportNvfp4Weights(weights);
  EXPECT_EQ(std::string(imported.raw_data()), std::string(encoded.raw_data()));
  EXPECT_EQ(imported.struct_type().SerializeAsString(), encoded.struct_type().SerializeAsString());
  EXPECT_EQ(imported.logical_type().SerializeAsString(),
            encoded.logical_type().SerializeAsString());
  EncodedValueProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(imported.SerializeAsString()));
  EXPECT_EQ(std::string(ImportNvfp4Weights(ExportNvfp4Weights(parsed)).raw_data()),
            std::string(imported.raw_data()));
}

TEST(Quantization, Nvfp4WeightsRoundingAndSignedZero) {
  const auto source = Tensor::FromFloat("", {1, 16},
                                        {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5, -.25f, -.75f,
                                         -1.25f, -1.75f, -2.5f, -3.5f, -5, -0.f, -.001f});
  const auto encoded = QuantizeNvfp4Weights(source, Tensor::FromFloat("", {1, 1}, {1}), 1);
  const auto weights = ExportNvfp4Weights(encoded);
  const std::vector<uint8_t> expected{0x20, 0x42, 0x64, 0x86, 0xaa, 0xcc, 0xee, 0x80};
  EXPECT_EQ(std::vector<uint8_t>(weights.weight_packed.bytes(),
                                 weights.weight_packed.bytes() + expected.size()),
            expected);
  const auto decoded = DequantizeTensor(encoded);
  EXPECT_FALSE(std::signbit(decoded.AsFloat()[0]));
  EXPECT_TRUE(std::signbit(decoded.AsFloat()[7]));
  EXPECT_FALSE(std::signbit(decoded.AsFloat()[14]));
  EXPECT_TRUE(std::signbit(decoded.AsFloat()[15]));

  const auto underflow =
      QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 16},
                                             {-std::numeric_limits<float>::denorm_min(), -0.f, -1,
                                              1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                           Tensor::FromFloat("", {1, 1}, {1}), std::numeric_limits<float>::min());
  const auto underflow_weights = ExportNvfp4Weights(underflow);
  EXPECT_EQ(underflow_weights.weight_packed.bytes()[0], 0x00);
  EXPECT_EQ(underflow_weights.weight_packed.bytes()[1], 0x08);
  const auto underflow_decoded = DequantizeTensor(underflow);
  EXPECT_FALSE(std::signbit(underflow_decoded.AsFloat()[0]));
  EXPECT_FALSE(std::signbit(underflow_decoded.AsFloat()[1]));
  EXPECT_TRUE(std::signbit(underflow_decoded.AsFloat()[2]));
  EXPECT_FALSE(std::signbit(underflow_decoded.AsFloat()[3]));

  const auto zeros = Tensor::FromFloat("", {2, 32}, std::vector<float>(64, 0));
  const auto rounded = ExportNvfp4Weights(QuantizeNvfp4Weights(
      zeros, Tensor::FromFloat("", {2, 2}, {1.0625f, 1.1875f, 449, 1.f / 512}), 3));
  const std::vector<uint8_t> scale_bytes{0x38, 0x3a, 0x7e, 0x01};
  EXPECT_EQ(std::vector<uint8_t>(rounded.weight_scale.bytes(), rounded.weight_scale.bytes() + 4),
            scale_bytes);
}

TEST(Quantization, Nvfp4WeightsSourceDerivedCalibrationGolden) {
  // Source-derived from compressed-tensors helpers.py and forward_helpers.py at
  // eb058d630f7344099459b6795b1e3eeffdda4b40; not generated by executing upstream.
  const std::vector<float> q{0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
  std::vector<float> values;
  for (float scale : {2.f, .5f})
    for (float value : q)
      values.push_back(value * scale);
  const auto encoded = QuantizeNvfp4Weights(Tensor::FromFloat("", {2, 16}, values),
                                            Tensor::FromFloat("", {2, 1}, {448, 112}), 224);
  const auto weights = ExportNvfp4Weights(encoded);
  const std::vector<uint8_t> expected{0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe,
                                      0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe};
  EXPECT_EQ(std::vector<uint8_t>(weights.weight_packed.bytes(),
                                 weights.weight_packed.bytes() + expected.size()),
            expected);
  EXPECT_EQ(weights.weight_scale.bytes()[0], 0x7e);
  EXPECT_EQ(weights.weight_scale.bytes()[1], 0x6e);
  EXPECT_EQ(weights.weight_global_scale, 224);
  const auto decoded = DequantizeTensor(encoded);
  ExpectValues(decoded, values);
  EXPECT_FALSE(std::signbit(decoded.AsFloat()[8]));
  EXPECT_FALSE(std::signbit(decoded.AsFloat()[24]));
  EXPECT_EQ(std::string(ImportNvfp4Weights(weights).raw_data()), std::string(encoded.raw_data()));
}

TEST(Quantization, Nvfp4WeightsUsesRoundedLocalDividedByGlobal) {
  const auto source = Tensor::FromFloat("", {2, 32}, std::vector<float>(64, 1));
  const auto encoded =
      QuantizeNvfp4Weights(source, Tensor::FromFloat("", {2, 2}, {1.0625f, 2, 4, 8}), 2);
  const auto weights = ExportNvfp4Weights(encoded);
  EXPECT_EQ(weights.weight_packed.bytes()[0], 0x44);
  EXPECT_EQ(weights.weight_packed.bytes()[8], 0x22);
  EXPECT_EQ(weights.weight_packed.bytes()[16], 0x11);
  EXPECT_EQ(weights.weight_packed.bytes()[24], 0x00);
  const auto reconstructed = DequantizeTensor(encoded);
  for (size_t i = 0; i < 48; ++i)
    EXPECT_EQ(reconstructed.AsFloat()[i], 1);
  for (size_t i = 48; i < 64; ++i)
    EXPECT_EQ(reconstructed.AsFloat()[i], 0);

  Tensor fp8("", TensorProto::FLOAT8E4M3FN, {2, 2}, {0x38, 0x40, 0x48, 0x50});
  const auto exact = ExportNvfp4Weights(QuantizeNvfp4Weights(source, fp8, 2));
  EXPECT_EQ(std::vector<uint8_t>(exact.weight_scale.bytes(), exact.weight_scale.bytes() + 4),
            std::vector<uint8_t>({0x38, 0x40, 0x48, 0x50}));

  const auto rounded =
      QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 16}, std::vector<float>(16, .43f)),
                           Tensor::FromFloat("", {1, 1}, {1}), 7);
  volatile float effective = 1.f / 7.f;
  const float expected = 3.f * effective;
  EXPECT_NE(expected, 3.f / 7.f);
  ExpectValues(DequantizeTensor(rounded), std::vector<float>(16, expected));
  EXPECT_EQ(ExportNvfp4Weights(rounded).weight_global_scale, 7);

  const auto order_sensitive =
      QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 16}, std::vector<float>(16, 1.f / 2048)),
                           Tensor("", TensorProto::FLOAT8E4M3FN, {1, 1}, {0x01}), 7);
  const auto order_weights = ExportNvfp4Weights(order_sensitive);
  for (size_t i = 0; i < order_weights.weight_packed.size_bytes(); ++i)
    EXPECT_EQ(order_weights.weight_packed.bytes()[i], 0x33);
  volatile float ordered_effective = (1.f / 512) / 7.f;
  const float ordered_expected = 1.5f * ordered_effective;
  ExpectValues(DequantizeTensor(order_sensitive), std::vector<float>(16, ordered_expected));
}

TEST(Quantization, Nvfp4WeightsResolvesCatalogueReferences) {
  auto encoded = QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 16}, std::vector<float>(16, 1)),
                                      Tensor::FromFloat("", {1, 1}, {1}), 1);
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(71);
  encoded.mutable_struct_type()->Clear();
  encoded.mutable_struct_type()->set_type_ref(71);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  ExpectValues(DequantizeTensor(encoded, catalogue), std::vector<float>(16, 1));
  ExpectValues(TensorFromProto(DequantizeTensorProto(encoded, catalogue)),
               std::vector<float>(16, 1));
  EXPECT_EQ(ExportNvfp4Weights(encoded, catalogue).weight_scale.bytes()[0], 0x38);
  EXPECT_THROW(ExportNvfp4Weights(encoded), std::invalid_argument);
}

TEST(Quantization, Nvfp4WeightsRejectsInvalidInputsAndScales) {
  const auto source = Tensor::FromFloat("", {1, 16}, std::vector<float>(16, 0));
  const auto scales = Tensor::FromFloat("", {1, 1}, {1});
  for (float scale : {0.f, -0.f, -1.f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN(), 1.f / 1024}) {
    EXPECT_THROW(QuantizeNvfp4Weights(source, Tensor::FromFloat("", {1, 1}, {scale}), 1),
                 std::invalid_argument);
  }
  for (uint8_t scale : {uint8_t{0}, uint8_t{0x80}, uint8_t{0xb8}, uint8_t{0x7f}, uint8_t{0xff}})
    EXPECT_THROW(
        QuantizeNvfp4Weights(source, Tensor("", TensorProto::FLOAT8E4M3FN, {1, 1}, {scale}), 1),
        std::invalid_argument);
  for (float global :
       {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    EXPECT_THROW(QuantizeNvfp4Weights(source, scales, global), std::invalid_argument);
  EXPECT_THROW(QuantizeNvfp4Weights(source, scales, std::numeric_limits<float>::denorm_min()),
               std::invalid_argument);
  EXPECT_NO_THROW(QuantizeNvfp4Weights(source, Tensor::FromFloat("", {1, 1}, {1.f / 512}),
                                       std::numeric_limits<float>::max()));
  for (const Shape &shape : {Shape{16}, Shape{0, 16}, Shape{1, 15}, Shape{1, -16}})
    EXPECT_THROW(QuantizeNvfp4Weights(Tensor("", TensorProto::FLOAT, shape, {}), scales, 1),
                 std::invalid_argument);
  EXPECT_THROW(QuantizeNvfp4Weights(Tensor("", TensorProto::DOUBLE, {1, 16}, {}), scales, 1),
               std::invalid_argument);
  EXPECT_THROW(QuantizeNvfp4Weights(source, Tensor::FromFloat("", {}, {1}), 1),
               std::invalid_argument);
  EXPECT_THROW(QuantizeNvfp4Weights(source, Tensor("", TensorProto::FLOAT, {1, 1}, {}), 1),
               std::invalid_argument);
  auto nonfinite = source;
  const auto infinity = std::numeric_limits<float>::infinity();
  std::memcpy(nonfinite.mutable_bytes(), &infinity, sizeof(float));
  EXPECT_THROW(QuantizeNvfp4Weights(nonfinite, scales, 1), std::invalid_argument);
}

TEST(Quantization, Nvfp4WeightsReportsScaleFailureStageAndGroup) {
  const auto source = Tensor::FromFloat("", {1, 32}, std::vector<float>(32, 0));
  auto rejects = [&](const Tensor &scales, float global, const char *message) {
    try {
      QuantizeNvfp4Weights(source, scales, global);
      FAIL() << "Expected invalid NVFP4 scales.";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
    }
  };
  rejects(
      Tensor::FromFloat("", {1, 2}, {1, 1.f / 1024}), 1,
      "NVFP4 WEIGHT local scale rounds to zero or a nonfinite value after FP8 rounding in group 1");
  rejects(Tensor::FromFloat("", {1, 2}, {1, 448}), std::numeric_limits<float>::min(),
          "NVFP4 WEIGHT effective scale overflow or nonfinite value in group 1");
}

TEST(Quantization, Nvfp4WeightsValidatesEmittedCodesAndReportsWirePosition) {
  const auto scales = Tensor::FromFloat("", {1, 2}, {1, 1});
  std::vector<float> values(32, 0);
  values[19] = std::numeric_limits<float>::max();
  auto rejects = [](auto operation) {
    try {
      operation();
      FAIL() << "Expected nonfinite NVFP4 reconstruction.";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(
          std::string(error.what())
              .find(
                  "NVFP4 WEIGHT reconstruction overflow or nonfinite value in group 1, element 3"),
          std::string::npos)
          << error.what();
    }
  };
  rejects([&] {
    QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 32}, values), scales,
                         std::numeric_limits<float>::min());
  });
  values[19] = std::numeric_limits<float>::max() / 2;
  EXPECT_NO_THROW(QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 32}, values), scales,
                                       std::numeric_limits<float>::min()));
  auto encoded = QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 32}, std::vector<float>(32, 0)),
                                      scales, std::numeric_limits<float>::min());
  auto weights = ExportNvfp4Weights(encoded);
  weights.weight_packed.mutable_bytes()[9] = 0x70;
  rejects([&] { ImportNvfp4Weights(weights); });
  (*encoded.mutable_raw_data())[9] = 0x70;
  rejects([&] { ExportNvfp4Weights(encoded); });
  rejects([&] { DequantizeTensor(encoded); });
  rejects([&] { DequantizeTensorProto(encoded); });
}

TEST(Quantization, Nvfp4WeightsExtremeScalingAndReconstruction) {
  const auto source =
      Tensor::FromFloat("", {1, 16},
                        {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  const auto encoded = QuantizeNvfp4Weights(source, Tensor::FromFloat("", {1, 1}, {1}),
                                            std::numeric_limits<float>::max());
  EXPECT_EQ(ExportNvfp4Weights(encoded).weight_packed.bytes()[0], 0xf7);
  const auto result = DequantizeTensor(encoded);
  volatile float effective = 1.f / std::numeric_limits<float>::max();
  EXPECT_EQ(result.AsFloat()[0], 6.f * effective);
  EXPECT_EQ(result.AsFloat()[1], -6.f * effective);

  auto weights = ExportNvfp4Weights(
      QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 16}, std::vector<float>(16, 0)),
                           Tensor::FromFloat("", {1, 1}, {1}), 1));
  const auto original = weights;
  Nvfp4Weights missing_global;
  missing_global.weight_packed = original.weight_packed;
  missing_global.weight_scale = original.weight_scale;
  EXPECT_EQ(missing_global.weight_global_scale, 0);
  EXPECT_THROW(ImportNvfp4Weights(missing_global), std::invalid_argument);
  weights.weight_scale.data_type = TensorProto::FLOAT;
  EXPECT_THROW(ImportNvfp4Weights(weights), std::invalid_argument);
  weights = original;
  weights.weight_scale.shape = {1};
  EXPECT_THROW(ImportNvfp4Weights(weights), std::invalid_argument);
  weights = original;
  weights.weight_packed.data_type = TensorProto::INT8;
  EXPECT_THROW(ImportNvfp4Weights(weights), std::invalid_argument);
  weights = original;
  weights.weight_global_scale = std::numeric_limits<float>::min();
  EXPECT_NO_THROW(ImportNvfp4Weights(weights));
  weights.weight_packed.mutable_bytes()[0] = 7;
  EXPECT_THROW(ImportNvfp4Weights(weights), std::invalid_argument);
  weights.weight_packed.shape = {1, std::numeric_limits<int64_t>::max()};
  EXPECT_THROW(ImportNvfp4Weights(weights), std::invalid_argument);
  weights.weight_packed.shape = {std::numeric_limits<int64_t>::max(), 8};
  EXPECT_THROW(ImportNvfp4Weights(weights), std::invalid_argument);
}

TEST(Quantization, Nvfp4WeightsRejectsMalformedWire) {
  const auto valid = QuantizeNvfp4Weights(Tensor::FromFloat("", {1, 16}, std::vector<float>(16, 1)),
                                          Tensor::FromFloat("", {1, 1}, {1}), 1);
  auto rejects = [](const EncodedValueProto &value) {
    EXPECT_THROW(ExportNvfp4Weights(value), std::invalid_argument);
    EXPECT_THROW(DequantizeTensor(value), std::invalid_argument);
    EXPECT_THROW(DequantizeTensorProto(value), std::invalid_argument);
  };
  for (size_t size = 0; size < valid.raw_data().size(); ++size) {
    auto corrupt = valid;
    corrupt.set_raw_data(std::string(valid.raw_data()).substr(0, size));
    rejects(corrupt);
  }
  auto corrupt = valid;
  corrupt.set_raw_data(std::string(valid.raw_data()) + "x");
  rejects(corrupt);
  corrupt = valid;
  corrupt.set_parameter_ref("unsupported");
  rejects(corrupt);
  corrupt = valid;
  corrupt.clear_logical_type();
  rejects(corrupt);
  corrupt = valid;
  corrupt.mutable_struct_type()->set_name("onnx_light.nvfp4_weights.v2");
  rejects(corrupt);
  corrupt = valid;
  corrupt.mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
  rejects(corrupt);
  corrupt = valid;
  corrupt.mutable_logical_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(1)
      ->set_dim_value(32);
  rejects(corrupt);
  for (size_t field = 0; field < 7; ++field) {
    corrupt = valid;
    corrupt.mutable_struct_type()->mutable_structure()->mutable_field(field)->set_name("changed");
    rejects(corrupt);
  }
  corrupt = valid;
  auto *constant =
      corrupt.mutable_struct_type()->mutable_structure()->mutable_field(0)->mutable_constant();
  constant->clear_int64_data();
  constant->add_int64_data(32);
  rejects(corrupt);
  for (uint8_t scale : {uint8_t{0}, uint8_t{0x80}, uint8_t{0xb8}, uint8_t{0x7f}}) {
    corrupt = valid;
    auto bytes = std::string(valid.raw_data());
    bytes[8] = static_cast<char>(scale);
    corrupt.set_raw_data(bytes);
    rejects(corrupt);
    for (uint32_t global :
         {uint32_t{0x7f800000}, uint32_t{0x7fc00000}, uint32_t{0xbf800000}, uint32_t{0x00800000}}) {
      corrupt = valid;
      bytes = std::string(valid.raw_data());
      bytes[0] = 7;
      for (size_t j = 0; j < 4; ++j)
        bytes[9 + j] = static_cast<char>(global >> (j * 8));
      corrupt.set_raw_data(bytes);
      rejects(corrupt);
    }
  }
  corrupt = valid;
  auto bytes = std::string(valid.raw_data());
  bytes[9] = bytes[10] = bytes[11] = bytes[12] = 0;
  corrupt.set_raw_data(bytes);
  rejects(corrupt);
}

TEST(Quantization, FloatReconstructionSurvivesSharedParametersAndOutputOverride) {
  const auto source = Tensor::FromFloat("X", {16}, std::vector<float>(16, 2.f));
  for (auto format : QuantizationFormats()) {
    if (format >= QuantizationFormat::kOrtMatmulnbitsInt2)
      continue;
    auto plan = WithTables(format, 16);
    plan.reconstruction_dtype = TensorProto::FLOAT;
    // Exercises inverse transforms, scattering and outlier restoration together.
    plan.permutation = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    plan.outliers = {0, 15};
    const auto full = QuantizeTensor(source, plan);
    EXPECT_TRUE(
        full.Encoded().struct_type().name().value().starts_with("onnx_light.quantization.v2/"));
    const auto decoded = DequantizeTensor(full);
    const auto widened = DequantizeTensor(full.Encoded(), {}, nullptr, TensorProto::DOUBLE);
    for (size_t i = 0; i < 16; ++i)
      EXPECT_EQ(widened.AsDouble()[i], static_cast<double>(decoded.AsFloat()[i]));
    const auto model = SharedModel(plan, full.Encoded());
    const auto parameters = QuantizationParameterCatalogue::Build(model);
    const auto shared =
        QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
    const auto materialized = MaterializeQuantizedValue(shared);
    EXPECT_EQ(materialized.struct_type().name().value(),
              full.Encoded().struct_type().name().value());
    const auto restored = DequantizeTensor(shared);
    for (size_t i = 0; i < 16; ++i)
      EXPECT_EQ(restored.AsFloat()[i], decoded.AsFloat()[i]);
  }
}

TEST(Quantization, FloatAffineWireGoldenIgnoresRequestedOutputPrecision) {
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 1, 1);
  plan.reconstruction_dtype = TensorProto::FLOAT;
  EncodedValueProto encoded;
  *encoded.mutable_struct_type() = MakeQuantizationType(plan);
  auto *logical = encoded.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(1);
  // Independent little-endian wire: reserved, scale=1+2^-23, zero=0, offset=-3, code=3.
  const std::array<uint8_t, 26> bytes = {0, 0, 0, 0, 0x20, 0, 0, 0xf0, 0x3f, 0, 0, 0,    0,
                                         0, 0, 0, 0, 0,    0, 0, 0,    0,    0, 8, 0xc0, 3};
  encoded.set_raw_data(bytes.data(), bytes.size());
  const auto result = DequantizeTensor(encoded, {}, nullptr, TensorProto::DOUBLE);
  EXPECT_EQ(result.AsDouble()[0], 0x1p-21);
  EXPECT_NE(result.AsDouble()[0], 3 * 0x1p-23);
}

TEST(Quantization, PagedCacheRetainsSharedParametersAcrossSerialization) {
  RuntimeValue retained;
  Tensor expected;
  {
    const auto source = Tensor::FromFloat("K", {1, 1, 4, 4}, std::vector<float>(16, 2.f));
    const auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 16, 4);
    const auto full = QuantizeTensor(source, plan);
    auto model = SharedModel(plan, full.Encoded());
    const auto parameters = QuantizationParameterCatalogue::Build(model);
    auto shared = QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
    expected = DequantizeTensor(shared);
    PagedCacheProto cache;
    auto *block = cache.add_blocks();
    block->set_start(0);
    block->set_length(1);
    *block->mutable_encoded_key() = shared.Encoded();
    *block->mutable_encoded_value() = shared.Encoded();
    EXPECT_THROW(RuntimeValue::FromPagedCache(cache), std::invalid_argument);
    retained = RuntimeValue::FromPagedCache(cache, {}, parameters);
    block->mutable_encoded_key()->set_parameter_ref("missing");
    EXPECT_THROW(RuntimeValue::FromPagedCache(cache, {}, parameters), std::invalid_argument);
  }
  const auto &key = retained.fields.at("blocks").elements[0].fields.at("key");
  PagedCacheProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(retained.ToPagedCache().SerializeAsString()));
  EXPECT_EQ(parsed.blocks(0).encoded_key().parameter_ref(), "weights");
  const auto restored = RuntimeValue::FromPagedCache(parsed, {}, key.quantization_parameters);
  const auto actual = DequantizeTensor(restored.fields.at("blocks").elements[0].fields.at("key"));
  EXPECT_EQ(actual.shape, expected.shape);
  EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), expected.size_bytes()), 0);
}

TEST(Quantization, SharedParametersEveryFormatAndLifetime) {
  for (auto format : QuantizationFormats()) {
    if (format == QuantizationFormat::kMxfp8)
      continue; // MXFP8 intentionally does not support shared numerical parameters.
    SCOPED_TRACE(std::string(QuantizationFormatName(format)));
    auto plan = WithTables(format, 16);
    auto source = Tensor::FromFloat(
        "X", format >= QuantizationFormat::kOrtMatmulnbitsInt2 ? Shape{8, 2} : Shape{16},
        std::vector<float>(16, 1));
    const auto full = QuantizeTensor(source, plan);
    RuntimeValue shared;
    {
      auto model = SharedModel(plan, full.Encoded());
      const auto memory = QuantizationParameterCatalogue::EstimateMemory(model);
      auto parameters = QuantizationParameterCatalogue::Build(model);
      const auto &entry = parameters->Get("weights");
      size_t retained = entry.common.capacity() +
                        entry.local_ranges.capacity() * sizeof(std::pair<size_t, size_t>) +
                        entry.plan.runs.capacity() * sizeof(QuantizationRun);
      for (const auto &run : entry.plan.runs) {
        retained += run.blocks.capacity() * sizeof(QuantizationBlockParameters);
        for (const auto &block : run.blocks)
          retained += block.codebook.capacity() * sizeof(double);
      }
      retained += (entry.plan.permutation.capacity() + entry.plan.outliers.capacity() +
                   entry.plan.matrix_shape.size()) *
                  sizeof(int64_t);
      retained += (entry.plan.forward.capacity() + entry.plan.inverse.capacity()) * sizeof(double);
      EXPECT_GE(memory.retained_bytes, retained);
      EXPECT_GT(memory.temporary_bytes, 0u);
      shared = QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
      EXPECT_EQ(shared.quantization_parameters, parameters);
      const auto second =
          QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
      EXPECT_EQ(shared.quantization_parameters, second.quantization_parameters);
      EXPECT_LT(shared.Encoded().raw_data().size(), full.Encoded().raw_data().size());
      EXPECT_EQ(StructTypeCatalogue{}.ValidateEncodedValue(shared.Encoded()).record_count, 1);
      auto wire = WireRoundTrip(shared);
      EXPECT_EQ(wire.Encoded().parameter_ref().value(), "weights");
      EXPECT_THROW(MaterializeQuantizedValue(wire), std::invalid_argument);
      wire.quantization_parameters = parameters;
      EXPECT_EQ(MaterializeQuantizedValue(wire).raw_data(), full.Encoded().raw_data());
      auto bad = source.ToOwned();
      bad.shape = Shape{16, 1};
      EXPECT_THROW(QuantizeTensorShared(bad, full.Encoded().struct_type(), "weights", parameters),
                   std::invalid_argument);
      EXPECT_THROW(
          QuantizeTensorShared(source, full.Encoded().struct_type(), "missing", parameters),
          std::invalid_argument);
    }
    const auto expected = DequantizeTensor(full);
    for (auto value : {shared.BorrowView(), shared.DeepCopy()}) {
      const auto actual = DequantizeTensor(value);
      for (int64_t i = 0; i < actual.element_count(); ++i)
        EXPECT_EQ(actual.AsFloat()[i], expected.AsFloat()[i]);
    }
  }
}

TEST(Quantization, SharedCatalogueDoesNotMaterializeLogicalTensorOrCodes) {
  const uint64_t count = uint64_t{1} << (sizeof(size_t) > 4 ? 30 : 27);
  for (auto format : {QuantizationFormat::kInt4, QuantizationFormat::kOrtMatmulnbitsInt4}) {
    SCOPED_TRACE(QuantizationFormatName(format));
    const bool ort = format == QuantizationFormat::kOrtMatmulnbitsInt4;
    auto plan = ort ? MakeMatMulNBitsPlan(format, count, 1, count)
                    : MakeQuantizationPlan(format, count, count);
    EncodedValueProto descriptor;
    auto *logical = descriptor.mutable_logical_type()->mutable_tensor_type();
    logical->set_elem_type(TensorProto::FLOAT);
    logical->mutable_shape()->add_dim()->set_dim_value(count);
    if (ort) {
      logical->mutable_shape()->add_dim()->set_dim_value(1);
      const auto small = QuantizeTensor(Tensor::FromFloat("", {16, 1}, std::vector<float>(16, 0)),
                                        MakeMatMulNBitsPlan(format, 16, 1, 16));
      *descriptor.mutable_struct_type() = small.Encoded().struct_type();
      auto *fields = descriptor.mutable_struct_type()->mutable_structure();
      fields->mutable_field(0)->mutable_constant()->ref_int64_data()[1] = count;
      fields->mutable_field(1)
          ->mutable_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(2)
          ->set_dim_value(count / 2);
    } else {
      *descriptor.mutable_struct_type() = MakeQuantizationType(plan);
    }
    const auto model = SharedModel(plan, descriptor);
    const auto parameters = QuantizationParameterCatalogue::Build(model);
    const auto &entry = parameters->Get("weights");
    EXPECT_EQ(entry.common.size(), ort ? 4u : 25u);
    EXPECT_EQ(entry.full_size, count / 2 + entry.common.size());
    ASSERT_FALSE(entry.local_ranges.empty());
    EXPECT_EQ(entry.local_ranges.back().second, count / 2);
    ASSERT_EQ(entry.plan.runs.size(), 1u);
    EXPECT_EQ(entry.plan.runs[0].blocks.size(), 1u);
    EXPECT_EQ(entry.plan.runs[0].layout.count, count);
  }
}

TEST(Quantization, SharedInitializerRetainsParametersAndRespectsExistingValues) {
  RuntimeValue retained;
  {
    auto plan = WithTables(QuantizationFormat::kInt4, 16);
    const auto source = Tensor::FromFloat("X", {16}, std::vector<float>(16, 1));
    const auto full = QuantizeTensor(source, plan);
    auto model = SharedModel(plan, full.Encoded());
    auto parameters = QuantizationParameterCatalogue::Build(model);
    auto shared = QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
    *model.mutable_graph()->add_extended_initializer()->mutable_encoded_value() = shared.Encoded();
    RuntimeSession session(model);
    RuntimeContext context;
    session.Run(context);
    ASSERT_EQ(context.values().count("X"), 1u);
    EXPECT_EQ(context.values().at("X").quantization_parameters, context.quantization_parameters());
    retained = context.values().at("X").DeepCopy();
    session.Run(context);
    EXPECT_EQ(context.values().at("X").quantization_parameters, retained.quantization_parameters);
    context.Remove("X");
    session.Run(context);
    EXPECT_EQ(context.values().at("X").Encoded().raw_data(), shared.Encoded().raw_data());
    EXPECT_TRUE(context.values().at("X").Encoded().has_parameter_ref());
    EXPECT_EQ(context.values().at("X").quantization_parameters, retained.quantization_parameters);

    context.Put("X", Tensor::FromFloat("X", {1}, {42}));
    session.Run(context);
    EXPECT_TRUE(context.Has("X"));
    EXPECT_EQ(context.values().count("X"), 0u);
    EXPECT_FLOAT_EQ(context.Get("X").AsFloat()[0], 42);
  }
  ExpectValues(DequantizeTensor(retained), std::vector<float>(16, 1));
}

TEST(Quantization, SharedInitializerSeedingRejectsInvalidCompactValues) {
  const auto plan = WithTables(QuantizationFormat::kInt4, 16);
  const auto source = Tensor::FromFloat("X", {16}, std::vector<float>(16, 1));
  const auto full = QuantizeTensor(source, plan);
  for (const auto &failure : {"reference", "logical_type", "payload", "reserved"}) {
    SCOPED_TRACE(failure);
    auto model = SharedModel(plan, full.Encoded());
    const auto parameters = QuantizationParameterCatalogue::Build(model);
    const auto shared =
        QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
    auto *encoded = model.mutable_graph()->add_extended_initializer()->mutable_encoded_value();
    *encoded = shared.Encoded();
    const std::string kind = failure;
    if (kind == "reference")
      encoded->set_parameter_ref("missing");
    else if (kind == "logical_type")
      encoded->mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
    else if (kind == "payload")
      encoded->mutable_raw_data()->resize(1);
    else
      (*encoded->mutable_raw_data())[0] = 1;
    RuntimeSession session(model);
    RuntimeContext context;
    EXPECT_THROW(session.Run(context), std::invalid_argument);
    EXPECT_FALSE(context.HasValue("X"));
  }
}

TEST(Quantization, SharedTransformsPermutationAndLocalOutliers) {
  auto plan = WithTables(QuantizationFormat::kInt4, 16);
  plan.permutation = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  plan.transform_size = 2;
  plan.forward = {1, 0, 0, 1};
  plan.inverse = plan.forward;
  plan.outliers = {3, 8};
  auto source = Tensor::FromFloat("X", {16}, std::vector<float>(16, 2));
  source.AsFloat()[3] = 100;
  auto full = QuantizeTensor(source, plan);
  auto model = SharedModel(plan, full.Encoded());
  auto parameters = QuantizationParameterCatalogue::Build(model);
  auto shared = QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
  EXPECT_EQ(shared.Encoded().raw_data().size(), 1u + 16u + 8u);
  EXPECT_EQ(MaterializeQuantizedValue(shared).raw_data(), full.Encoded().raw_data());
  auto unnamed = shared.Encoded();
  unnamed.clear_name();
  unnamed.clear_doc_string();
  const auto materialized = MaterializeQuantizedValue(unnamed, parameters.get());
  EXPECT_FALSE(materialized.has_name());
  EXPECT_FALSE(materialized.has_doc_string());
  source.AsFloat()[3] = 200;
  auto second = QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
  EXPECT_EQ(DequantizeTensor(shared).AsFloat()[3], 100);
  EXPECT_EQ(DequantizeTensor(second).AsFloat()[3], 200);
}

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
  EXPECT_EQ(QuantizationFormats().size(), 44u);
  for (const auto &format : QuantizationFormats()) {
    if (format == QuantizationFormat::kMxfp8)
      continue; // Covered with a complete rank-two group in the MXFP8-specific tests.
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
    const auto full = QuantizeTensor(source, plan);
    auto model = SharedModel(plan, full.Encoded());
    auto parameters = QuantizationParameterCatalogue::Build(model);
    auto shared = QuantizeTensorShared(source, full.Encoded().struct_type(), "weights", parameters);
    EXPECT_EQ(DequantizeTensor(shared).shape, shape);
    EXPECT_EQ(MaterializeQuantizedValue(shared).raw_data(), full.Encoded().raw_data());
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
    auto model = SharedModel(plan, encoded.Encoded());
    auto parameters = QuantizationParameterCatalogue::Build(model);
    auto shared =
        QuantizeTensorShared(source, encoded.Encoded().struct_type(), "weights", parameters);
    const auto shared_result = DequantizeTensor(shared);
    EXPECT_EQ(shared_result.data_type, dtype);
    EXPECT_EQ(std::memcmp(shared_result.bytes(), source.bytes(), source.size_bytes()), 0);
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
  plan.format = static_cast<QuantizationFormat>(44);
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
  ASSERT_FALSE(encoded.has_name());
  ASSERT_FALSE(encoded.has_doc_string());
  EncodedValueProto loaded;
  loaded.ParseFromString(encoded.SerializeAsString());
  ASSERT_FALSE(loaded.has_name());
  ASSERT_FALSE(loaded.has_doc_string());
  auto tensor = DequantizeTensor(RuntimeValue(loaded));
  EXPECT_TRUE(tensor.name.empty());
  ExpectValues(tensor, {1, 2});
  const auto decoded = DequantizeTensorProto(loaded);
  EXPECT_FALSE(decoded.has_name());
  EXPECT_FALSE(decoded.has_doc_string());
  EXPECT_TRUE(decoded.name().empty());
  EXPECT_TRUE(decoded.doc_string().empty());
  ExpectValues(TensorFromProto(decoded), {1, 2});
}

TEST(Quantization, PreservesIndependentMetadataPresenceAcrossProtoRoundTrips) {
  for (const auto format : {QuantizationFormat::kInt4, QuantizationFormat::kOrtMatmulnbitsInt4}) {
    const auto plan = format == QuantizationFormat::kInt4 ? MakeQuantizationPlan(format, 2)
                                                          : MakeMatMulNBitsPlan(format, 2, 1, 16);
    for (int name = 0; name < 3; ++name)
      for (int doc = 0; doc < 3; ++doc) {
        SCOPED_TRACE(QuantizationFormatName(format));
        SCOPED_TRACE(name);
        SCOPED_TRACE(doc);
        TensorProto source;
        source.set_data_type(TensorProto::FLOAT);
        source.add_dims(2);
        source.add_dims(1);
        source.add_float_data(1);
        source.add_float_data(2);
        if (name != 0)
          source.set_name(name == 1 ? "" : "weight");
        if (doc != 0)
          source.set_doc_string(doc == 1 ? "" : "weight documentation");
        const auto original = source.SerializeAsString();
        const auto encoded = QuantizeTensorProto(source, plan);
        EXPECT_EQ(source.SerializeAsString(), original);
        EXPECT_EQ(encoded.has_name(), source.has_name());
        EXPECT_EQ(encoded.has_doc_string(), source.has_doc_string());
        EXPECT_EQ(encoded.name(), source.name());
        EXPECT_EQ(encoded.doc_string(), source.doc_string());
        EncodedValueProto loaded;
        loaded.ParseFromString(encoded.SerializeAsString());
        const auto decoded = DequantizeTensorProto(loaded);
        EXPECT_EQ(decoded.has_name(), source.has_name());
        EXPECT_EQ(decoded.has_doc_string(), source.has_doc_string());
        EXPECT_EQ(decoded.name(), source.name());
        EXPECT_EQ(decoded.doc_string(), source.doc_string());
        ExpectValues(TensorFromProto(decoded), {1, 2});
      }
  }
}

TEST(Quantization, AcceptsPresentEmptyExternalRawData) {
  TensorProto source;
  source.set_data_type(TensorProto::FLOAT);
  source.add_dims(0);
  source.set_data_location(TensorProto::EXTERNAL);
  const auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 0);
  EXPECT_THROW(QuantizeTensorProto(source, plan), std::invalid_argument);
  source.set_raw_data(std::string{});
  ASSERT_TRUE(source.is_raw_data());
  const auto original = source.SerializeAsString();
  const auto encoded = QuantizeTensorProto(source, plan);
  const auto decoded = DequantizeTensorProto(encoded);
  EXPECT_EQ(decoded.dims(0), 0);
  EXPECT_EQ(decoded.raw_data().size(), 0u);
  EXPECT_EQ(source.SerializeAsString(), original);
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

TEST(Quantization, ValidatesCastTypeForEveryPortableMethod) {
  const auto source = Tensor::FromFloat("", {3}, {-1, 0, 1});
  for (auto format :
       {QuantizationFormat::kInt4, QuantizationFormat::kNf4, QuantizationFormat::kTiledFloat}) {
    SCOPED_TRACE(QuantizationFormatName(format));
    auto plan = MakeQuantizationPlan(format, 3);
    const auto valid = QuantizeTensor(source, plan).Encoded();
    for (int32_t type : {-1, 0, int32_t(TensorProto::INT8), int32_t(TensorProto::STRING),
                         std::numeric_limits<int32_t>::max()}) {
      SCOPED_TRACE(type);
      plan.runs[0].layout.cast_type = type;
      EXPECT_THROW(QuantizeTensor(source, plan), std::invalid_argument);
      auto corrupt = valid;
      auto *parameters = corrupt.mutable_struct_type()
                             ->mutable_structure()
                             ->mutable_field(6)
                             ->mutable_type()
                             ->mutable_struct_type()
                             ->mutable_structure()
                             ->mutable_field(1)
                             ->mutable_type()
                             ->mutable_struct_type()
                             ->mutable_array()
                             ->mutable_element_type()
                             ->mutable_struct_type()
                             ->mutable_structure()
                             ->mutable_field(0)
                             ->mutable_constant();
      parameters->ref_int64_data()[8] = type;
      EXPECT_THROW(DequantizeTensor(corrupt), std::invalid_argument);
      EXPECT_THROW(DequantizeTensorProto(corrupt), std::invalid_argument);
    }
    for (int32_t type :
         {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
      SCOPED_TRACE(type);
      plan.runs[0].layout.cast_type = type;
      ExpectValues(DequantizeTensor(WireRoundTrip(QuantizeTensor(source, plan))), {-1, 0, 1});
    }
  }
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

TEST(Quantization, RejectsAffineReconstructionOverflowDuringEncoding) {
  for (int32_t type :
       {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT, TensorProto::DOUBLE}) {
    SCOPED_TRACE(type);
    const double scale = std::ldexp(1., type == TensorProto::FLOAT16  ? 15
                                        : type == TensorProto::DOUBLE ? 1023
                                                                      : 127);
    const auto source = [type](double value) {
      if (type == TensorProto::DOUBLE)
        return Tensor::FromDouble("", {1, 1}, {value});
      if (type == TensorProto::FLOAT)
        return Tensor::FromFloat("", {1, 1}, {static_cast<float>(value)});
      if (type == TensorProto::FLOAT16)
        return MakeFloat16Tensor("", {1, 1}, {static_cast<float>(value)});
      return MakeBfloat16Tensor("", {1, 1}, {static_cast<float>(value)});
    };
    for (double sign : {-1., 1.}) {
      auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 1);
      plan.runs[0].blocks[0].scale = scale;
      const auto input = source(sign * 1.5 * scale);
      EXPECT_THROW(QuantizeTensor(input, plan), std::invalid_argument);
      plan.runs[0].blocks[0].scale = scale / 2;
      EXPECT_NO_THROW(DequantizeTensor(QuantizeTensor(input, plan)));
    }
    if (type == TensorProto::DOUBLE)
      continue;
    for (auto format :
         {QuantizationFormat::kOrtMatmulnbitsInt2, QuantizationFormat::kOrtMatmulnbitsInt4,
          QuantizationFormat::kOrtMatmulnbitsInt8}) {
      SCOPED_TRACE(QuantizationFormatName(format));
      auto plan = MakeMatMulNBitsPlan(format, 1, 1, 16);
      plan.runs[0].blocks[0].scale = scale;
      plan.runs[0].blocks[0].zero_point = 0;
      const auto input = source(1.5 * scale);
      EXPECT_THROW(QuantizeTensor(input, plan), std::invalid_argument);
      plan.runs[0].blocks[0].scale = scale / 2;
      EXPECT_NO_THROW(DequantizeTensor(QuantizeTensor(input, plan)));
    }
  }
}

TEST(Quantization, ChecksReconstructionAfterInverseTransformsAndOutlierRestoration) {
  const auto input = MakeFloat16Tensor("", {1}, {49152});
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 1);
  plan.runs[0].blocks[0].scale = 32768;
  plan.transform_size = 1;
  plan.forward = {0.5};
  plan.inverse = {2};
  EXPECT_THROW(QuantizeTensor(input, plan), std::invalid_argument);
  plan.forward = {2};
  plan.inverse = {0.5};
  const auto decoded = DequantizeTensor(QuantizeTensor(input, plan));
  EXPECT_EQ(std::memcmp(decoded.bytes(), input.bytes(), input.size_bytes()), 0);

  const auto outlier_input = MakeFloat16Tensor("", {2}, {49152, 1});
  plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 2, 1);
  plan.permutation = {1, 0};
  plan.outliers = {0};
  plan.runs[0].blocks[1].scale = 1e308;
  plan.runs[0].blocks[1].offset = 5e307;
  const auto restored = DequantizeTensor(QuantizeTensor(outlier_input, plan));
  EXPECT_EQ(std::memcmp(restored.bytes(), outlier_input.bytes(), outlier_input.size_bytes()), 0);
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
    if (format == QuantizationFormat::kMxfp8)
      EXPECT_EQ(MakeQuantizationPlan(format, 32).format, format);
    else if (format < QuantizationFormat::kOrtMatmulnbitsInt2)
      EXPECT_EQ(MakeQuantizationPlan(format, 0).format, format);
    else
      EXPECT_EQ(MakeMatMulNBitsPlan(format, 1, 1, 16).format, format);
  }
  EXPECT_THROW(QuantizationFormatName(static_cast<QuantizationFormat>(-1)), std::invalid_argument);
  EXPECT_THROW(QuantizationFormatName(static_cast<QuantizationFormat>(44)), std::invalid_argument);
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
    const auto model = SharedModel(plan, encoded.Encoded());
    const auto parameters = QuantizationParameterCatalogue::Build(model);
    const auto shared =
        QuantizeTensorShared(Tensor::FromFloat("", {k, n}, values), encoded.Encoded().struct_type(),
                             "weights", parameters);
    EXPECT_EQ(MaterializeQuantizedValue(shared).raw_data(), encoded.Encoded().raw_data());
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

TEST(Quantization, OrtMatMulNBitsRejectsEveryNonzeroPaddingBit) {
  for (const auto format :
       {QuantizationFormat::kOrtMatmulnbitsInt2, QuantizationFormat::kOrtMatmulnbitsInt4,
        QuantizationFormat::kOrtMatmulnbitsInt8}) {
    SCOPED_TRACE(QuantizationFormatName(format));
    for (size_t k : {35, 36, 48, 64}) {
      SCOPED_TRACE(k);
      constexpr size_t n = 2, block_size = 16;
      auto plan = MakeMatMulNBitsPlan(format, k, n, block_size);
      for (auto &block : plan.runs[0].blocks)
        block.zero_point = 0;
      const size_t bits = plan.runs[0].layout.bits;
      const size_t groups = (k + block_size - 1) / block_size;
      const size_t weight_column_bytes = groups * block_size * bits / 8;
      const size_t zero_column_bytes = (groups * bits + 7) / 8;
      const std::vector<float> values(k * n, 1);
      const auto encoded = QuantizeTensor(Tensor::FromFloat("", {int64_t(k), n}, values), plan);
      const auto valid = encoded.Encoded();
      ExpectValues(DequantizeTensor(encoded), values);
      ExpectValues(TensorFromProto(DequantizeTensorProto(valid)), values);
      const auto inputs = ExportMatMulNBitsInputs(valid);
      ASSERT_TRUE(inputs.zero_points.has_value());
      ASSERT_EQ(inputs.zero_points->data_type(), TensorProto::UINT8);
      const size_t zero_offset = inputs.weights.raw_data().size() + inputs.scales.raw_data().size();
      const auto reject_bit = [&](size_t bit) {
        SCOPED_TRACE(bit);
        auto corrupt = valid;
        auto raw = std::string(valid.raw_data());
        ASSERT_EQ(static_cast<uint8_t>(raw[bit / 8]) & (1u << (bit % 8)), 0u);
        raw[bit / 8] = static_cast<char>(static_cast<uint8_t>(raw[bit / 8]) | (1u << (bit % 8)));
        corrupt.set_raw_data(raw);
        EXPECT_THROW(DequantizeTensor(corrupt), std::invalid_argument);
        EXPECT_THROW(DequantizeTensorProto(corrupt), std::invalid_argument);
        EXPECT_THROW(ExportMatMulNBitsInputs(corrupt), std::invalid_argument);
      };
      for (size_t column = 0; column < n; ++column) {
        SCOPED_TRACE(column);
        for (size_t bit = k * bits; bit < weight_column_bytes * 8; ++bit)
          reject_bit(column * weight_column_bytes * 8 + bit);
        for (size_t bit = groups * bits; bit < zero_column_bytes * 8; ++bit)
          reject_bit((zero_offset + column * zero_column_bytes) * 8 + bit);
      }
    }
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
    auto model = SharedModel(plan, encoded.Encoded());
    auto parameters = QuantizationParameterCatalogue::Build(model);
    auto shared =
        QuantizeTensorShared(source, encoded.Encoded().struct_type(), "weights", parameters);
    EXPECT_EQ(MaterializeQuantizedValue(shared).raw_data(), encoded.Encoded().raw_data());
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

TEST(Quantization, SchemaMismatchReportsFieldPath) {
  const auto source = Tensor::FromFloat("", {2, 1}, {1, 2});
  for (auto format : {QuantizationFormat::kInt4, QuantizationFormat::kOrtMatmulnbitsInt4}) {
    const auto plan = format == QuantizationFormat::kInt4 ? MakeQuantizationPlan(format, 2)
                                                          : MakeMatMulNBitsPlan(format, 2, 1, 16);
    auto encoded = QuantizeTensor(source, plan).Encoded();
    encoded.mutable_struct_type()->mutable_structure()->mutable_field(1)->set_doc_string("extra");
    try {
      DequantizeTensor(encoded);
      FAIL() << "Expected a schema mismatch";
    } catch (const std::invalid_argument &error) {
      const std::string message = error.what();
      EXPECT_NE(message.find("descriptor does not match its versioned schema"), std::string::npos);
      EXPECT_NE(message.find("structure.field[1].doc_string: presence differs"), std::string::npos);
    }
  }
  auto encoded =
      QuantizeTensor(source, MakeMatMulNBitsPlan(QuantizationFormat::kOrtMatmulnbitsInt4, 2, 1, 16))
          .Encoded();
  encoded.mutable_struct_type()->mutable_structure()->mutable_field(1)->set_name("weights");
  try {
    ExportMatMulNBitsInputs(encoded);
    FAIL() << "Expected an ORT schema mismatch";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what()).find("structure.field[1].name: values differ"),
              std::string::npos);
  }
}
