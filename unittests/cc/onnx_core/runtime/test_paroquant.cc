// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

ParoQuantWeights Fixture() {
  ParoQuantWeights result;
  result.in_features = 8;
  result.out_features = 8;
  result.group_size = 4;
  result.krot = 2;
  // GPTQModel c1976918 AWQ pack_awq_columns gathers [0,2,4,6,1,3,5,7].
  // Unpacked q[k,n]=(2*k+n)%16, zeros[g,n]=n; no GPTQ zero-point-minus-one.
  result.tensors.emplace("qweight",
                         Tensor::FromUint32("qweight", {8, 1},
                                            {0x75316420u, 0x97538642u, 0xb975a864u, 0xdb97ca86u,
                                             0xfdb9eca8u, 0x1fdb0ecau, 0x31fd20ecu, 0x531f420eu}));
  result.tensors.at("qweight").data_type = TensorProto::INT32;
  result.tensors.emplace("qzeros", Tensor::FromInt32("qzeros", {2, 1}, {0x75316420, 0x75316420}));
  result.tensors.emplace("scales",
                         MakeFloat16Tensor("scales", {2, 8},
                                           {.25f, .25f, .25f, .25f, .25f, .25f, .25f, .25f, .5f,
                                            .5f, .5f, .5f, .5f, .5f, .5f, .5f}));
  // ParoLinear's group-local pairs and theta are consumed in ascending krot order.
  result.tensors.emplace(
      "pairs",
      Tensor::FromInt16("pairs", {2, 8}, {0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 1, 3, 3, 1, 2, 0}));
  result.tensors.emplace(
      "theta", MakeFloat16Tensor("theta", {2, 4}, {.25f, -.5f, .75f, -1, -.25f, .5f, -.75f, 1}));
  result.tensors.emplace("channel_scales", MakeFloat16Tensor("channel_scales", {1, 8},
                                                             {.5f, 2, 1, 1.5f, 2, .25f, 1, 3}));
  result.tensors.emplace(
      "bias", MakeFloat16Tensor("bias", {8}, {0, .125f, .25f, .375f, .5f, .625f, .75f, .875f}));
  return result;
}

float FloatAt(const Tensor &tensor, size_t index) {
  float result;
  std::memcpy(&result, tensor.bytes() + index * sizeof(float), sizeof(float));
  return result;
}

Tensor Input() { return Tensor::FromFloat("input", {1, 8}, {1, -2, 3, -4, 5, -6, 7, -8}); }

void SetPair(ParoQuantWeights &weights, size_t index, int16_t value) {
  std::memcpy(weights.tensors.at("pairs").mutable_bytes() + index * 2, &value, 2);
}

void RejectAll(const EncodedValueProto &value) {
  EXPECT_THROW(ExportParoQuantWeights(value), std::invalid_argument);
  EXPECT_THROW(UnpackParoQuantCodes(value), std::invalid_argument);
  EXPECT_THROW(ReconstructParoQuantWeights(value), std::invalid_argument);
  EXPECT_THROW(ApplyParoQuantLinear(value, Input()), std::invalid_argument);
}

} // namespace

TEST(ParoQuantWeights, SourcePackingRoundTripAndTransformedBasisReconstruction) {
  const auto checkpoint = Fixture();
  const auto original = ImportParoQuantWeights(checkpoint);
  EncodedValueProto encoded;
  ASSERT_TRUE(encoded.ParseFromString(original.SerializeAsString()));
  EXPECT_EQ(encoded.struct_type().name().value(), "onnx_light.paroquant_weights.v1");
  const auto exported = ExportParoQuantWeights(encoded);
  EXPECT_EQ(exported.in_features, 8);
  EXPECT_EQ(exported.out_features, 8);
  EXPECT_EQ(exported.group_size, 4);
  EXPECT_EQ(exported.krot, 2);
  EXPECT_EQ(exported.bits, 4u);
  EXPECT_EQ(exported.group_axis, 1);
  EXPECT_EQ(exported.upstream_revision, checkpoint.upstream_revision);
  EXPECT_EQ(ImportParoQuantWeights(exported).SerializeAsString(), original.SerializeAsString());
  for (const auto &[name, tensor] : checkpoint.tensors) {
    const auto &actual = exported.tensors.at(name);
    EXPECT_EQ(actual.data_type, tensor.data_type);
    EXPECT_EQ(actual.shape, tensor.shape);
    ASSERT_EQ(actual.size_bytes(), tensor.size_bytes());
    EXPECT_EQ(std::memcmp(actual.bytes(), tensor.bytes(), tensor.size_bytes()), 0);
  }
  const auto codes = UnpackParoQuantCodes(encoded);
  const auto weight = ReconstructParoQuantWeights(encoded);
  EXPECT_EQ(codes.shape, (Shape{8, 8}));
  EXPECT_EQ(codes.data_type, TensorProto::UINT8);
  EXPECT_EQ(weight.shape, (Shape{8, 8}));
  EXPECT_EQ(weight.data_type, TensorProto::FLOAT);
  for (size_t n = 0; n < 8; ++n)
    for (size_t k = 0; k < 8; ++k) {
      EXPECT_EQ(codes.bytes()[n * 8 + k], (2 * k + n) % 16);
      EXPECT_EQ(FloatAt(weight, n * 8 + k),
                (float((2 * k + n) % 16) - float(n)) * (k < 4 ? .25f : .5f));
    }
  AwqWeights awq;
  awq.variant = "gemm";
  awq.in_features = awq.out_features = 8;
  awq.group_size = 4;
  for (const char *name : {"qweight", "qzeros", "scales"})
    awq.tensors.emplace(name, checkpoint.tensors.at(name).BorrowView());
  const auto awq_weight = ReconstructAwqWeights(ImportAwqWeights(awq));
  EXPECT_EQ(std::memcmp(awq_weight.bytes(), weight.bytes(), weight.size_bytes()), 0);
  EXPECT_THROW(ExportAwqWeights(encoded), std::invalid_argument);
}

TEST(ParoQuantWeights, AppliesNoncommutingRotationsScalingAndBiasAcrossGroups) {
  auto checkpoint = Fixture();
  const auto encoded = ImportParoQuantWeights(checkpoint);
  const auto output = ApplyParoQuantLinear(encoded, Input());
  // Independent double-precision evaluation of the pinned ParoQuant reference equations.
  const std::vector<float> expected{-8.77120199f,   -8.64620199f,   -10.25163007f, -10.12663007f,
                                    -156.01471701f, -155.88971701f, -70.84906089f, -70.72406089f};
  EXPECT_EQ(output.shape, (Shape{1, 8}));
  EXPECT_EQ(output.data_type, TensorProto::FLOAT);
  for (size_t n = 0; n < 8; ++n)
    EXPECT_NEAR(FloatAt(output, n), expected[n], 5e-5f);
  auto input = Tensor::FromFloat("input", {2, 1, 8},
                                 {1, -2, 3, -4, 5, -6, 7, -8, 1, -2, 3, -4, 5, -6, 7, -8});
  const auto batched = ApplyParoQuantLinear(encoded, input);
  EXPECT_EQ(batched.shape, (Shape{2, 1, 8}));
  for (size_t n = 0; n < 16; ++n)
    EXPECT_NEAR(FloatAt(batched, n), expected[n % 8], 5e-5f);
  for (const char *name : {"pairs", "theta"}) {
    auto &tensor = checkpoint.tensors.at(name);
    const size_t row = tensor.size_bytes() / 2;
    for (size_t j = 0; j < row; ++j)
      std::swap(tensor.mutable_bytes()[j], tensor.mutable_bytes()[row + j]);
  }
  const auto reversed = ApplyParoQuantLinear(ImportParoQuantWeights(checkpoint), Input());
  EXPECT_GT(std::abs(FloatAt(reversed, 0) - expected[0]), 1);
  checkpoint = Fixture();
  checkpoint.tensors.erase("bias");
  const auto no_bias = ApplyParoQuantLinear(ImportParoQuantWeights(checkpoint), Input());
  for (size_t n = 0; n < 8; ++n)
    EXPECT_NEAR(FloatAt(no_bias, n), expected[n] - float(n) * .125f, 5e-5f);
}

TEST(ParoQuantWeights, MaterializesNonPowerScalesInPinnedWeightPrecision) {
  for (int32_t dtype : {TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    auto checkpoint = Fixture();
    checkpoint.tensors.at("qweight") =
        Tensor::FromInt32("qweight", {8, 1}, std::vector<int32_t>(8, 0x77777777));
    checkpoint.tensors.at("qzeros") = Tensor::FromInt32("qzeros", {2, 1}, {0, 0});
    checkpoint.tensors.at("theta") = MakeFloat16Tensor("theta", {2, 4}, std::vector<float>(8, 0));
    checkpoint.tensors.at("channel_scales") =
        MakeFloat16Tensor("channel_scales", {1, 8}, std::vector<float>(8, 1));
    checkpoint.tensors.erase("bias");
    auto &scales = checkpoint.tensors.at("scales");
    scales = MakeFloat16Tensor("scales", {2, 8}, std::vector<float>(16, .3f));
    if (dtype == TensorProto::BFLOAT16) {
      scales.data_type = dtype;
      const uint16_t bits = FloatToBfloat16Bits(.3f);
      for (size_t i = 0; i < scales.size_bytes(); i += 2)
        std::memcpy(scales.mutable_bytes() + i, &bits, 2);
    }
    const auto value = ImportParoQuantWeights(checkpoint);
    const auto weights = ReconstructParoQuantWeights(value);
    // dequantize_gemm materializes 7*scale in the original scales dtype, then casts to x.dtype.
    const float expected = dtype == TensorProto::FLOAT16 ? 2.099609375f : 2.109375f;
    for (size_t i = 0; i < 64; ++i)
      EXPECT_EQ(FloatAt(weights, i), expected);
    const auto output =
        ApplyParoQuantLinear(value, Tensor::FromFloat("input", {1, 8}, {1, 2, 3, 4, 5, 6, 7, 8}));
    for (size_t i = 0; i < 8; ++i)
      EXPECT_EQ(FloatAt(output, i), dtype == TensorProto::FLOAT16 ? 75.5859375f : 75.9375f);
    if (dtype == TensorProto::FLOAT16) {
      scales = MakeFloat16Tensor("scales", {2, 8}, std::vector<float>(16, 65504));
      EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument);
      auto changed = value;
      constexpr size_t scale_offset = 8 * 4 + 2 * 4;
      (*changed.mutable_raw_data())[scale_offset] = '\xff';
      (*changed.mutable_raw_data())[scale_offset + 1] = '\x7b';
      RejectAll(changed);
    }
  }
}

TEST(ParoQuantWeights, PreservesSupportedParameterDtypesAndGroupSentinel) {
  for (int32_t dtype : {TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    auto checkpoint = Fixture();
    for (const char *name : {"theta", "channel_scales", "bias"}) {
      auto &tensor = checkpoint.tensors.at(name);
      std::vector<float> values(tensor.size_bytes() / 2);
      for (size_t j = 0; j < values.size(); ++j) {
        uint16_t half;
        std::memcpy(&half, tensor.bytes() + j * 2, 2);
        values[j] = Float16BitsToFloat(half);
      }
      if (dtype == TensorProto::BFLOAT16) {
        tensor.data_type = dtype;
        for (size_t j = 0; j < values.size(); ++j) {
          const uint16_t bits = FloatToBfloat16Bits(values[j]);
          std::memcpy(tensor.mutable_bytes() + j * 2, &bits, 2);
        }
      }
    }
    auto &scales = checkpoint.tensors.at("scales");
    scales.data_type = TensorProto::BFLOAT16;
    for (size_t j = 0; j < 16; ++j) {
      const auto bits = FloatToBfloat16Bits(j < 8 ? .25f : .5f);
      std::memcpy(scales.mutable_bytes() + j * 2, &bits, 2);
    }
    const auto value = ImportParoQuantWeights(checkpoint);
    const auto exported = ExportParoQuantWeights(value);
    for (const auto &[name, tensor] : checkpoint.tensors) {
      const auto &actual = exported.tensors.at(name);
      EXPECT_EQ(actual.data_type, tensor.data_type);
      EXPECT_EQ(std::memcmp(actual.bytes(), tensor.bytes(), tensor.size_bytes()), 0);
    }
    EXPECT_NEAR(FloatAt(ApplyParoQuantLinear(value, Input()), 0), -8.77120199f, 5e-5f);
  }
  auto checkpoint = Fixture();
  checkpoint.group_size = -1;
  checkpoint.tensors.at("scales") =
      MakeFloat16Tensor("scales", {1, 8}, std::vector<float>(8, .25f));
  checkpoint.tensors.at("qzeros") = Tensor::FromInt32("qzeros", {1, 1}, {0x75316420});
  checkpoint.tensors.at("pairs") =
      Tensor::FromInt16("pairs", {2, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 7, 5, 6, 4});
  const auto value = ImportParoQuantWeights(checkpoint);
  EXPECT_EQ(ExportParoQuantWeights(value).group_size, -1);
  EXPECT_NO_THROW(ApplyParoQuantLinear(value, Input()));
}

TEST(ParoQuantWeights, ResolvesExactCatalogueSchemaAndRejectsGenericConsumers) {
  const auto original = ImportParoQuantWeights(Fixture());
  ModelProto model;
  *model.add_struct_types() = original.struct_type();
  model.mutable_struct_types(0)->set_type_id(23);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  auto referenced = original;
  referenced.mutable_struct_type()->clear_structure();
  referenced.mutable_struct_type()->clear_name();
  referenced.mutable_struct_type()->set_type_ref(23);
  EXPECT_EQ(ExportParoQuantWeights(referenced, 4096, catalogue).krot, 2);
  EXPECT_NO_THROW(UnpackParoQuantCodes(referenced, 4096, catalogue));
  EXPECT_NO_THROW(ReconstructParoQuantWeights(referenced, 4096, catalogue));
  EXPECT_NO_THROW(ApplyParoQuantLinear(referenced, Input(), 4096, catalogue));
  EXPECT_THROW(ExportParoQuantWeights(referenced), std::invalid_argument);
  for (const auto *value : std::array<const EncodedValueProto *, 2>{&original, &referenced}) {
    EXPECT_THROW(DequantizeTensor(*value, catalogue), std::invalid_argument);
    EXPECT_THROW(DequantizeTensorProto(*value, catalogue), std::invalid_argument);
    EXPECT_THROW(DequantizeTensor(RuntimeValue::FromEncodedView(*value), catalogue),
                 std::invalid_argument);
    try {
      DequantizeTensor(*value, catalogue);
      FAIL() << "Generic dequantization dropped ParoQuant side state.";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find("ApplyParoQuantLinear"), std::string::npos);
    }
  }
  auto typed = original;
  typed.mutable_struct_type()->set_type_id(42);
  EXPECT_THROW(ExportParoQuantWeights(typed), std::invalid_argument);
}

TEST(ParoQuantWeights, RejectsInvalidMetadataAndOverflowBeforeAllocation) {
  const auto good = Fixture();
  for (int64_t invalid :
       {int64_t{0}, int64_t{-2}, int64_t{3}, int64_t{32770}, std::numeric_limits<int64_t>::max()}) {
    auto checkpoint = good;
    checkpoint.group_size = invalid;
    EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument);
  }
  for (int field = 0; field < 6; ++field)
    for (int64_t invalid : {int64_t{0}, int64_t{-1}, std::numeric_limits<int64_t>::max()}) {
      auto checkpoint = good;
      if (field == 0)
        checkpoint.in_features = invalid;
      if (field == 1)
        checkpoint.out_features = invalid;
      if (field == 2)
        checkpoint.krot = invalid;
      if (field == 3)
        checkpoint.bits = static_cast<uint32_t>(invalid);
      if (field == 4)
        checkpoint.group_axis = invalid;
      if (field == 5)
        checkpoint.upstream_revision = "future";
      EXPECT_THROW(ImportParoQuantWeights(checkpoint, std::numeric_limits<size_t>::max()),
                   std::invalid_argument);
    }
  auto checkpoint = good;
  checkpoint.krot = std::numeric_limits<int64_t>::max() / 2;
  EXPECT_THROW(ImportParoQuantWeights(checkpoint, std::numeric_limits<size_t>::max()),
               std::invalid_argument);
  checkpoint = good;
  checkpoint.in_features = int64_t{1} << 62;
  EXPECT_THROW(ImportParoQuantWeights(checkpoint, std::numeric_limits<size_t>::max()),
               std::invalid_argument);
  const auto encoded = ImportParoQuantWeights(good);
  EXPECT_THROW(ImportParoQuantWeights(good, 0), std::invalid_argument);
  EXPECT_THROW(ImportParoQuantWeights(good, 255), std::invalid_argument);
  EXPECT_THROW(ExportParoQuantWeights(encoded, 255), std::invalid_argument);
  EXPECT_THROW(UnpackParoQuantCodes(encoded, 255), std::invalid_argument);
  EXPECT_THROW(ReconstructParoQuantWeights(encoded, 255), std::invalid_argument);
  EXPECT_THROW(ApplyParoQuantLinear(encoded, Input(), 255), std::invalid_argument);
  const size_t checkpoint_budget = encoded.raw_data().size() * 2 + 8 * 8 * sizeof(float);
  EXPECT_NO_THROW(ImportParoQuantWeights(good, checkpoint_budget));
  EXPECT_THROW(ImportParoQuantWeights(good, checkpoint_budget - 1), std::invalid_argument);
  EXPECT_NO_THROW(ExportParoQuantWeights(encoded, checkpoint_budget));
  const size_t apply_budget = checkpoint_budget + 2 * 8 * sizeof(float);
  EXPECT_NO_THROW(ApplyParoQuantLinear(encoded, Input(), apply_budget));
  EXPECT_THROW(ApplyParoQuantLinear(encoded, Input(), apply_budget - 1), std::invalid_argument);
}

TEST(ParoQuantWeights, SupportsLargestInt16PermutationGroup) {
  auto checkpoint = Fixture();
  checkpoint.in_features = checkpoint.group_size = 32768;
  checkpoint.krot = 1;
  checkpoint.tensors.at("qweight") =
      Tensor::FromInt32("qweight", {32768, 1}, std::vector<int32_t>(32768, 0x75316420));
  checkpoint.tensors.at("qzeros") = Tensor::FromInt32("qzeros", {1, 1}, {0x75316420});
  checkpoint.tensors.at("scales") = MakeFloat16Tensor("scales", {1, 8}, std::vector<float>(8, 1));
  std::vector<int16_t> pairs(32768);
  for (size_t i = 0; i < pairs.size(); ++i)
    pairs[i] = static_cast<int16_t>(i);
  checkpoint.tensors.at("pairs") = Tensor::FromInt16("pairs", {1, 32768}, pairs);
  checkpoint.tensors.at("theta") =
      MakeFloat16Tensor("theta", {1, 16384}, std::vector<float>(16384, 0));
  checkpoint.tensors.at("channel_scales") =
      MakeFloat16Tensor("channel_scales", {1, 32768}, std::vector<float>(32768, 1));
  const auto encoded = ImportParoQuantWeights(checkpoint);
  EXPECT_EQ(ExportParoQuantWeights(encoded).group_size, 32768);
  EXPECT_NO_THROW(UnpackParoQuantCodes(encoded));
  SetPair(checkpoint, 32767, 0);
  EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument);
}

TEST(ParoQuantWeights, ValidatesAllRequiredSideStateEvenForIdentity) {
  auto good = Fixture();
  good.tensors.at("theta") = MakeFloat16Tensor("theta", {2, 4}, std::vector<float>(8, 0));
  good.tensors.at("channel_scales") =
      MakeFloat16Tensor("channel_scales", {1, 8}, std::vector<float>(8, 1));
  for (const char *name : {"qweight", "qzeros", "scales", "pairs", "theta", "channel_scales"}) {
    auto checkpoint = good;
    checkpoint.tensors.erase(name);
    EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument) << name;
  }
  for (int16_t invalid : {int16_t{-1}, int16_t{1}, int16_t{4}, int16_t{32767}}) {
    auto checkpoint = good;
    SetPair(checkpoint, 0, invalid);
    EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument);
  }
  for (const auto &[name, tensor] : good.tensors) {
    auto checkpoint = good;
    checkpoint.tensors.at(name).shape = {1};
    EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument) << name;
    checkpoint = good;
    checkpoint.tensors.at(name).data_type = TensorProto::DOUBLE;
    EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument) << name;
    checkpoint = good;
    checkpoint.tensors.at(name) =
        Tensor(name, tensor.data_type, tensor.shape, std::vector<uint8_t>{0});
    EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument) << name;
  }
  auto checkpoint = good;
  checkpoint.tensors.emplace("rotation_matrix", Tensor::FromFloat("extra", {1}, {0}));
  EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument);
  for (const char *name : {"scales", "theta", "channel_scales", "bias"}) {
    checkpoint = good;
    const uint16_t infinity = 0x7c00;
    std::memcpy(checkpoint.tensors.at(name).mutable_bytes(), &infinity, 2);
    EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument);
    checkpoint = good;
    const auto shape = checkpoint.tensors.at(name).shape;
    const size_t count = checkpoint.tensors.at(name).size_bytes() / 2;
    checkpoint.tensors.at(name) = Tensor::FromFloat(name, shape, std::vector<float>(count, 1));
    EXPECT_THROW(ImportParoQuantWeights(checkpoint), std::invalid_argument);
  }
}

TEST(ParoQuantWeights, RejectsMalformedEncodedMetadataPayloadAndSideState) {
  const auto original = ImportParoQuantWeights(Fixture());
  auto changed = original;
  changed.mutable_raw_data()->resize(changed.raw_data().size() - 1);
  RejectAll(changed);
  changed = original;
  changed.mutable_raw_data()->push_back(0);
  RejectAll(changed);
  changed = original;
  changed.mutable_struct_type()->set_name("onnx_light.paroquant_weights.v2");
  RejectAll(changed);
  changed = original;
  changed.mutable_struct_type()->mutable_structure()->mutable_field(7)->set_name("order");
  RejectAll(changed);
  changed = original;
  changed.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(0)
      ->mutable_constant()
      ->ref_int64_data()
      .clear();
  RejectAll(changed);
  changed = original;
  changed.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(3)
      ->mutable_constant()
      ->ref_int64_data()[0] = std::numeric_limits<int64_t>::max();
  RejectAll(changed);
  changed = original;
  changed.mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT16);
  RejectAll(changed);
  // Payload offsets are flat AWQ fields, then required pairs/theta/channel_scales.
  constexpr size_t pairs_offset = 8 * 4 + 2 * 4 + 16 * 2;
  changed = original;
  (*changed.mutable_raw_data())[pairs_offset] = 1;
  RejectAll(changed);
  for (size_t offset : {pairs_offset + 16 * 2, pairs_offset + 16 * 2 + 8 * 2,
                        pairs_offset + 16 * 2 + 8 * 2 + 8 * 2}) {
    changed = original;
    (*changed.mutable_raw_data())[offset] = 0;
    (*changed.mutable_raw_data())[offset + 1] = '\x7c';
    RejectAll(changed);
  }
}

TEST(ParoQuantWeights, ValidatesInputsWorkspaceAndFloatOverflow) {
  const auto value = ImportParoQuantWeights(Fixture());
  auto input = Input();
  input.shape = {8};
  EXPECT_THROW(ApplyParoQuantLinear(value, input), std::invalid_argument);
  input = Input();
  input.shape = {1, 7};
  EXPECT_THROW(ApplyParoQuantLinear(value, input), std::invalid_argument);
  input = Input();
  input.data_type = TensorProto::INT32;
  EXPECT_THROW(ApplyParoQuantLinear(value, input), std::invalid_argument);
  input = Input();
  input.shape = {-1, 8};
  EXPECT_THROW(ApplyParoQuantLinear(value, input), std::invalid_argument);
  input.shape = {std::numeric_limits<int64_t>::max(), 2, 8};
  EXPECT_THROW(ApplyParoQuantLinear(value, input, std::numeric_limits<size_t>::max()),
               std::invalid_argument);
  input =
      Tensor::FromFloat("bad", {1, 8}, std::vector<float>(8, std::numeric_limits<float>::max()));
  EXPECT_THROW(ApplyParoQuantLinear(value, input), std::invalid_argument);
  input = Tensor::FromFloat("bad", {1, 8},
                            std::vector<float>(8, std::numeric_limits<float>::quiet_NaN()));
  EXPECT_THROW(ApplyParoQuantLinear(value, input), std::invalid_argument);
  input = Tensor("empty", TensorProto::FLOAT, {2, 0, 8}, std::vector<uint8_t>{});
  const auto empty = ApplyParoQuantLinear(value, input);
  EXPECT_EQ(empty.shape, (Shape{2, 0, 8}));
  EXPECT_EQ(empty.size_bytes(), size_t{0});
  input = Tensor::FromFloat("large", {16, 8}, std::vector<float>(128, 1));
  EXPECT_THROW(ApplyParoQuantLinear(value, input, 1200), std::invalid_argument);
}
