// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {
Tensor Ints(const char *name, Shape shape, const std::vector<int32_t> &values) {
  std::vector<uint8_t> bytes(values.size() * 4);
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return Tensor(name, TensorProto::INT32, std::move(shape), std::move(bytes));
}
Tensor Floats(const char *name, Shape shape, const std::vector<float> &values) {
  std::vector<uint8_t> bytes(values.size() * 4);
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return Tensor(name, TensorProto::FLOAT, std::move(shape), std::move(bytes));
}
void SetInt(Tensor &t, size_t i, int32_t value) {
  std::memcpy(t.mutable_bytes() + i * 4, &value, 4);
}
VptqWeights Fixture(bool outliers = false) {
  VptqWeights w;
  w.in_features = 17;
  w.out_features = 5;
  w.vector_lens = {3, 2};
  w.num_centroids = {7, 3};
  w.num_res_centroids = {-1, 5};
  w.enable_norm = true;
  w.enable_perm = true;
  w.bits = 6;
  w.kmeans_mode = "fixture";
  w.kmeans_iters = 37;
  w.kmeans_tol = 3.5e-6;
  w.kmeans_seed = 42;
  w.block_size = 64;
  w.percdamp = .125;
  w.npercent = outliers ? 20 : 0;
  w.group_size = outliers ? -1 : 9;
  w.group_num = outliers ? 3 : -1;
  const size_t groups = outliers ? 3 : 2, columns = outliers ? 4 : 9;
  const size_t words = (columns * 5 + 31) / 32;
  std::vector<int32_t> packed(groups * 3 * words, 0);
  // Independent bit-at-a-time producer includes row-crossing and signed INT32 words.
  for (size_t group = 0; group < groups; ++group)
    for (size_t row = 0; row < 3; ++row)
      for (size_t col = 0; col < columns; ++col) {
        const uint32_t main = (group + row + col) % 3;
        const uint32_t residual = (2 * group + row + col) % 5;
        const uint32_t merged = main | (residual << 2);
        for (size_t bit = 0; bit < 5; ++bit) {
          const size_t position = col * 5 + bit;
          auto &word = packed[(group * 3 + row) * words + position / 32];
          uint32_t raw;
          std::memcpy(&raw, &word, 4);
          raw |= ((merged >> bit) & 1) << (position % 32);
          std::memcpy(&word, &raw, 4);
        }
      }
  w.tensors.emplace("indices", Ints("indices", {int64_t(groups), 3, int64_t(words)}, packed));
  std::vector<float> main(groups * 3 * 2), residual(groups * 5 * 2);
  for (size_t group = 0; group < groups; ++group) {
    for (size_t entry = 0; entry < 3; ++entry)
      for (size_t component = 0; component < 2; ++component)
        main[(group * 3 + entry) * 2 + component] = 100 * group + 10 * entry + component;
    for (size_t entry = 0; entry < 5; ++entry)
      for (size_t component = 0; component < 2; ++component)
        residual[(group * 5 + entry) * 2 + component] = entry * .25f + component * .125f;
  }
  w.tensors.emplace("centroids.weight", Floats("centroids.weight", {int64_t(groups), 6}, main));
  w.tensors.emplace("res_centroids.weight",
                    Floats("res_centroids.weight", {int64_t(groups), 10}, residual));
  if (outliers) {
    std::vector<float> book(21);
    for (size_t i = 0; i < book.size(); ++i)
      book[i] = -100.f - static_cast<float>(i);
    w.tensors.emplace("outlier_centroids.weight",
                      Floats("outlier_centroids.weight", {1, 21}, book));
    std::vector<int32_t> indices(10);
    for (size_t i = 0; i < indices.size(); ++i)
      indices[i] = i % 7;
    w.tensors.emplace("outlier_indices", Ints("outlier_indices", {1, 2, 5}, indices));
  }
  std::vector<int32_t> perm(17);
  std::vector<float> scale(17), shift(17);
  for (size_t i = 0; i < 17; ++i) {
    perm[i] = (i + 3) % 17;
    scale[i] = i + 1;
    shift[i] = -.5f * i;
  }
  w.tensors.emplace("perm", Ints("perm", {17}, perm));
  w.tensors.emplace("weight_scale", Floats("weight_scale", {17}, scale));
  w.tensors.emplace("weight_bias", Floats("weight_bias", {17}, shift));
  w.tensors.emplace("bias", Floats("bias", {5}, {99, 98, 97, 96, 95}));
  return w;
}
void Equal(const VptqWeights &a, const VptqWeights &b) {
  EXPECT_EQ(a.upstream_revision, b.upstream_revision);
  EXPECT_EQ(a.schema_version, b.schema_version);
  EXPECT_EQ(a.in_features, b.in_features);
  EXPECT_EQ(a.out_features, b.out_features);
  EXPECT_EQ(a.bits, b.bits);
  EXPECT_EQ(a.vector_lens, b.vector_lens);
  EXPECT_EQ(a.num_centroids, b.num_centroids);
  EXPECT_EQ(a.num_res_centroids, b.num_res_centroids);
  EXPECT_EQ(a.npercent, b.npercent);
  EXPECT_EQ(a.group_size, b.group_size);
  EXPECT_EQ(a.group_num, b.group_num);
  EXPECT_EQ(a.enable_norm, b.enable_norm);
  EXPECT_EQ(a.norm_dim, b.norm_dim);
  EXPECT_EQ(a.enable_perm, b.enable_perm);
  EXPECT_EQ(a.kmeans_mode, b.kmeans_mode);
  EXPECT_EQ(a.kmeans_iters, b.kmeans_iters);
  EXPECT_EQ(a.kmeans_tol, b.kmeans_tol);
  EXPECT_EQ(a.kmeans_seed, b.kmeans_seed);
  EXPECT_EQ(a.block_size, b.block_size);
  EXPECT_EQ(a.percdamp, b.percdamp);
  ASSERT_EQ(a.tensors.size(), b.tensors.size());
  for (const auto &[name, tensor] : a.tensors) {
    const auto &actual = b.tensors.at(name);
    EXPECT_EQ(tensor.data_type, actual.data_type);
    EXPECT_EQ(tensor.shape, actual.shape);
    ASSERT_EQ(tensor.size_bytes(), actual.size_bytes());
    EXPECT_EQ(std::memcmp(tensor.bytes(), actual.bytes(), tensor.size_bytes()), 0);
  }
}
} // namespace

TEST(VptqWeights, NativeGroupedResidualOutlierPartialVectorPermutationNormRoundTrip) {
  for (bool outlier : {false, true}) {
    const auto w = Fixture(outlier);
    const auto encoded = ImportVptqWeights(w);
    EncodedValueProto restored;
    ASSERT_TRUE(restored.ParseFromString(encoded.SerializeAsString()));
    Equal(w, ExportVptqWeights(restored));
    const auto tensor = ReconstructVptqWeights(restored);
    ASSERT_EQ(tensor.shape, (Shape{5, 17}));
    ASSERT_EQ(tensor.data_type, TensorProto::FLOAT);
    const auto *actual = reinterpret_cast<const float *>(tensor.bytes());
    const size_t outliers = outlier ? 5 : 0, columns = outlier ? 4 : 9;
    for (size_t row = 0; row < 5; ++row)
      for (size_t col = 0; col < 17; ++col) {
        float expected;
        if (col < outliers) {
          expected = -100 - float(((row / 3 * outliers + col) % 7) * 3 + row % 3);
        } else {
          const size_t group = (col - outliers) / columns, column = (col - outliers) % columns;
          const size_t main = (group + row / 2 + column) % 3;
          const size_t residual = (2 * group + row / 2 + column) % 5;
          expected = 100 * group + 10 * main + row % 2 + residual * .25f + (row % 2) * .125f;
        }
        const size_t dest = (col + 3) % 17;
        expected = expected * (dest + 1) - .5f * dest;
        EXPECT_EQ(actual[row * 17 + dest], expected) << row << "," << col;
      }
    const auto generic = DequantizeTensor(restored);
    EXPECT_EQ(std::memcmp(tensor.bytes(), generic.bytes(), tensor.size_bytes()), 0);
  }
}

TEST(VptqWeights, ExportsValidatedConfigWithoutMaterializingTensors) {
  const auto encoded = ImportVptqWeights(Fixture());
  const auto config = ExportVptqConfig(encoded);
  EXPECT_TRUE(config.tensors.empty());
  EXPECT_EQ(config.in_features, 17);
  EXPECT_EQ(config.out_features, 5);
  EXPECT_EQ(config.group_size, 9);
  EXPECT_EQ(config.enable_norm, true);

  auto corrupt = encoded;
  corrupt.mutable_raw_data()->data()[0] = static_cast<char>(0xff);
  EXPECT_ANY_THROW(ExportVptqConfig(corrupt));
}

TEST(VptqWeights, RoundsEverySourceDtypeOperationAndSupportsOutputAxisNorm) {
  for (int32_t dtype : {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT}) {
    VptqWeights w;
    w.in_features = 1;
    w.out_features = 1;
    w.vector_lens = {-1, 1};
    w.num_centroids = {-1, 2};
    w.num_res_centroids = {-1, 2};
    w.enable_norm = true;
    w.norm_dim = 1;
    const float base = dtype == TensorProto::FLOAT16    ? 2048.f
                       : dtype == TensorProto::BFLOAT16 ? 256.f
                                                        : 16777216.f;
    const float factor = dtype == TensorProto::FLOAT16    ? 1.0009765625f
                         : dtype == TensorProto::BFLOAT16 ? 1.0078125f
                                                          : 1.00000011920928955078125f;
    auto make = [&](const char *name, Shape shape, std::vector<float> values) {
      return dtype == TensorProto::FLOAT16    ? MakeFloat16Tensor(name, shape, values)
             : dtype == TensorProto::BFLOAT16 ? MakeBfloat16Tensor(name, shape, values)
                                              : Floats(name, shape, values);
    };
    w.tensors.emplace("indices", Ints("indices", {1, 1, 1}, {0}));
    w.tensors.emplace("centroids.weight", make("centroids.weight", {1, 2}, {base, 0}));
    w.tensors.emplace("res_centroids.weight", make("res_centroids.weight", {1, 2}, {1, 0}));
    w.tensors.emplace("weight_scale", make("weight_scale", {1}, {factor}));
    w.tensors.emplace("weight_bias", make("weight_bias", {1}, {1}));
    const auto encoded = ImportVptqWeights(w);
    Equal(w, ExportVptqWeights(encoded));
    const auto result = ReconstructVptqWeights(encoded);
    const float expected = dtype == TensorProto::FLOAT16    ? 2052.f
                           : dtype == TensorProto::BFLOAT16 ? 260.f
                                                            : 16777220.f;
    EXPECT_EQ(reinterpret_cast<const float *>(result.bytes())[0], expected);
  }
}

TEST(VptqWeights, BaseOnlyAndNonpersistentInversePermutation) {
  auto w = Fixture();
  w.num_res_centroids[1] = -1;
  w.tensors.erase("res_centroids.weight");
  w.tensors.at("indices") = Ints("indices", {2, 3, 1}, {0, 0, 0, 0, 0, 0});
  const auto encoded = ImportVptqWeights(w);
  Equal(w, ExportVptqWeights(encoded));
  EXPECT_EQ(ReconstructVptqWeights(encoded).shape, (Shape{5, 17}));
}

TEST(VptqWeights, SupportsFullWidthSignedPackedWordsAndRejectsWiderIndices) {
  VptqWeights w;
  w.in_features = 2;
  w.out_features = 1;
  w.vector_lens = {-1, 1};
  w.num_centroids = {-1, 65536};
  w.num_res_centroids = {-1, 65536};
  std::vector<float> main(65536), residual(65536);
  for (size_t i = 0; i < main.size(); ++i) {
    main[i] = static_cast<float>(i);
    residual[i] = -.5f * static_cast<float>(i);
  }
  const uint32_t raw[] = {0x8000ffffu, 0xffff8000u};
  std::vector<int32_t> words(2);
  std::memcpy(words.data(), raw, sizeof(raw));
  w.tensors.emplace("indices", Ints("indices", {1, 1, 2}, words));
  w.tensors.emplace("centroids.weight", Floats("centroids.weight", {1, 65536}, main));
  w.tensors.emplace("res_centroids.weight", Floats("res_centroids.weight", {1, 65536}, residual));
  const auto encoded = ImportVptqWeights(w);
  Equal(w, ExportVptqWeights(encoded));
  const auto tensor = ReconstructVptqWeights(encoded);
  const auto *actual = reinterpret_cast<const float *>(tensor.bytes());
  EXPECT_EQ(actual[0], 49151.f);
  EXPECT_EQ(actual[1], .5f);
  w.num_centroids[1] = 65537;
  EXPECT_ANY_THROW(ImportVptqWeights(w));
}

TEST(VptqWeights, RejectsInvalidNativeStateAndResourceLimits) {
  VptqWeights tiny;
  tiny.in_features = 1;
  tiny.out_features = 1;
  tiny.vector_lens = {-1, 1};
  tiny.num_centroids = {-1, 2};
  tiny.tensors.emplace("indices", Ints("indices", {1, 1, 1}, {0}));
  tiny.tensors.emplace("centroids.weight", MakeFloat16Tensor("centroids.weight", {1, 2}, {1, 2}));
  const auto small = ImportVptqWeights(tiny, 8);
  Equal(tiny, ExportVptqWeights(small, 8));
  EXPECT_EQ(ReconstructVptqWeights(small, 8).size_bytes(), size_t{4});
  tiny.tensors.at("centroids.weight") = Floats("centroids.weight", {1, 2}, {1, 2});
  const auto small_float = ImportVptqWeights(tiny, 12);
  Equal(tiny, ExportVptqWeights(small_float, 12));
  EXPECT_EQ(ReconstructVptqWeights(small_float, 12).size_bytes(), size_t{4});
  auto w = Fixture();
  EXPECT_ANY_THROW(ImportVptqWeights(w, 0));
  EXPECT_ANY_THROW(ImportVptqWeights(w, 339));
  EXPECT_ANY_THROW(ImportVptqWeights(w, 399));
  EXPECT_EQ(ImportVptqWeights(w, 400).raw_data().size(), size_t{400});
  w.upstream_revision = "future";
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  w.schema_version = 2;
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  w.num_res_centroids[0] = 2;
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  w.group_num = 2;
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  SetInt(w.tensors.at("indices"), 0, 3);
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  SetInt(w.tensors.at("indices"), 0, 5 << 2);
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture(true);
  SetInt(w.tensors.at("outlier_indices"), 0, -1);
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  SetInt(w.tensors.at("perm"), 0, 4);
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  SetInt(w.tensors.at("perm"), 0, -1);
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  w.tensors.emplace("inverse_perm", Ints("inverse_perm", {17}, std::vector<int32_t>(17)));
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  w.tensors.emplace("unpacked_indices", Ints("unpacked_indices", {1}, {0}));
  EXPECT_ANY_THROW(ImportVptqWeights(w));
  w = Fixture();
  w.in_features = int64_t(std::numeric_limits<int32_t>::max()) + 2;
  EXPECT_ANY_THROW(ImportVptqWeights(w, std::numeric_limits<size_t>::max()));
  w = Fixture();
  w.in_features = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ImportVptqWeights(w, std::numeric_limits<size_t>::max()));
}

TEST(VptqWeights, RejectsHostileProtoAndResolvesSharedCatalogue) {
  const auto valid = ImportVptqWeights(Fixture());
  EXPECT_ANY_THROW(ExportVptqWeights(valid, 339));
  EXPECT_ANY_THROW(ReconstructVptqWeights(valid, 339));
  auto corrupt = valid;
  corrupt.mutable_raw_data()->resize(corrupt.raw_data().size() - 1);
  EXPECT_ANY_THROW(ExportVptqWeights(corrupt));
  corrupt = valid;
  corrupt.mutable_raw_data()->push_back(0);
  EXPECT_ANY_THROW(ReconstructVptqWeights(corrupt));
  corrupt = valid;
  corrupt.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(1)
      ->mutable_constant()
      ->ref_int64_data()[0] = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ExportVptqWeights(corrupt, std::numeric_limits<size_t>::max()));
  corrupt = valid;
  corrupt.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(1)
      ->mutable_constant()
      ->ref_int64_data()[0] = int64_t(std::numeric_limits<int32_t>::max()) + 2;
  EXPECT_ANY_THROW(ExportVptqWeights(corrupt, std::numeric_limits<size_t>::max()));
  corrupt = valid;
  corrupt.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(24)
      ->mutable_constant()
      ->ref_int64_data()[0] = 0;
  EXPECT_ANY_THROW(ExportVptqWeights(corrupt));
  corrupt = valid;
  corrupt.mutable_struct_type()->mutable_structure()->mutable_field(0)->set_name("revision_alias");
  EXPECT_ANY_THROW(ExportVptqWeights(corrupt));
  corrupt = valid;
  corrupt.mutable_logical_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(0)
      ->set_dim_value(999);
  EXPECT_ANY_THROW(ReconstructVptqWeights(corrupt));
  corrupt = valid;
  corrupt.mutable_raw_data()->data()[0] = 3;
  EXPECT_ANY_THROW(ReconstructVptqWeights(corrupt));
  ModelProto model;
  *model.add_struct_types() = valid.struct_type();
  model.mutable_struct_types(0)->set_type_id(97);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  auto referenced = valid;
  referenced.mutable_struct_type()->clear_structure();
  referenced.mutable_struct_type()->clear_name();
  referenced.mutable_struct_type()->set_type_ref(97);
  Equal(Fixture(), ExportVptqWeights(referenced, size_t{1} << 30, catalogue));
  EXPECT_EQ(ReconstructVptqWeights(referenced, size_t{1} << 30, catalogue).shape, (Shape{5, 17}));
}
