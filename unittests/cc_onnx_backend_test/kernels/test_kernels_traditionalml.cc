// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::OpsetId;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::ArrayFeatureExtractor;
using onnx_backend_test::kernel::Binarizer;
using onnx_backend_test::kernel::Imputer;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::LabelEncoder;
using onnx_backend_test::kernel::LinearClassifier;
using onnx_backend_test::kernel::LinearRegressor;
using onnx_backend_test::kernel::OneHotEncoder;
using onnx_backend_test::kernel::Scaler;
using onnx_backend_test::kernel::SVMClassifier;
using onnx_backend_test::kernel::SVMRegressor;
using onnx_backend_test::kernel::ZipMap;

namespace Test {

TEST(BackendKernelClass, LabelEncoderInt64ToFloatMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<int64_t> keys{0, 1, 2};
  const std::vector<float> values{0.5f, 1.5f, 2.5f};
  Tensor x = Tensor::FromInt64("", {4}, {0, 1, 2, 7});
  Tensor y = label_encoder.operator()<int64_t, float>(x, keys, values, /*default=*/-1.0f);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.5f);
  EXPECT_FLOAT_EQ(py[1], 1.5f);
  EXPECT_FLOAT_EQ(py[2], 2.5f);
  EXPECT_FLOAT_EQ(py[3], -1.0f);
}

TEST(BackendKernelClass, LabelEncoderFloatToInt64MatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<float> keys{1.0f, 2.0f, 3.0f};
  const std::vector<int64_t> values{10, 20, 30};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 9.0f});
  Tensor y = label_encoder.operator()<float, int64_t>(x, keys, values, /*default=*/-1);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 10);
  EXPECT_EQ(py[1], 20);
  EXPECT_EQ(py[2], 30);
  EXPECT_EQ(py[3], -1);
}

TEST(BackendKernelClass, LabelEncoderInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<int64_t> keys{0, 1};
  const std::vector<float> values{7.0f, 8.0f};
  Tensor x = Tensor::FromInt64("", {3}, {1, 0, 9});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {3},
             std::vector<uint8_t>(3 * sizeof(float), 0u));
  label_encoder.operator()<int64_t, float>(x, keys, values, /*default=*/-2.0f, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], 8.0f);
  EXPECT_FLOAT_EQ(po[1], 7.0f);
  EXPECT_FLOAT_EQ(po[2], -2.0f);
}

TEST(BackendKernelClass, LabelEncoderRejectsMismatchedKeysValues) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<int64_t> keys{0, 1, 2};
  const std::vector<float> values{0.5f, 1.5f};
  Tensor x = Tensor::FromInt64("", {1}, {0});
  EXPECT_THROW(((void)label_encoder.operator()<int64_t, float>(x, keys, values, 0.0f)),
               std::invalid_argument);
}

TEST(BackendKernelClass, LabelEncoderRejectsWrongInputDtype) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<int64_t> keys{0, 1};
  const std::vector<float> values{0.5f, 1.5f};
  Tensor x = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  EXPECT_THROW(((void)label_encoder.operator()<int64_t, float>(x, keys, values, 0.0f)),
               std::invalid_argument);
}

TEST(BackendKernelClass, LabelEncoderStringToInt64WithDefault) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<std::string> keys{"a", "b", "c"};
  const std::vector<int64_t> values{0, 1, 2};
  Tensor x = Tensor::FromStrings("", {5}, {"a", "b", "d", "c", "g"});
  Tensor y = label_encoder.operator()<std::string, int64_t>(x, keys, values, /*default=*/42);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{5}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 42);
  EXPECT_EQ(py[3], 2);
  EXPECT_EQ(py[4], 42);
}

TEST(BackendKernelClass, LabelEncoderStringToInt16WithDefault) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<std::string> keys{"a", "b", "c"};
  const std::vector<int16_t> values{0, 1, 2};
  Tensor x = Tensor::FromStrings("", {5}, {"a", "b", "d", "c", "g"});
  Tensor y = label_encoder.operator()<std::string, int16_t>(x, keys, values, /*default=*/42);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT16));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{5}));
  const int16_t *py = reinterpret_cast<const int16_t *>(y.data.data());
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 42);
  EXPECT_EQ(py[3], 2);
  EXPECT_EQ(py[4], 42);
}

TEST(BackendKernelClass, LabelEncoderStringRejectsNonStringInput) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<std::string> keys{"a"};
  const std::vector<int64_t> values{0};
  Tensor x = Tensor::FromInt64("", {1}, {0});
  EXPECT_THROW(((void)label_encoder.operator()<std::string, int64_t>(x, keys, values, -1)),
               std::invalid_argument);
}

TEST(BackendKernelClass, BinarizerFloatThresholdElementwise) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Binarizer binarizer{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f});
  Tensor y = binarizer.operator()<float>(x, /*threshold=*/1.0f);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 0.0f); // x == threshold maps to 0 (strict greater than).
  EXPECT_FLOAT_EQ(py[4], 1.0f);
  EXPECT_FLOAT_EQ(py[5], 1.0f);
}

TEST(BackendKernelClass, BinarizerInt64ThresholdElementwise) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Binarizer binarizer{ctx};
  Tensor x = Tensor::FromInt64("", {5}, {0, 3, 4, -2, 10});
  Tensor y = binarizer.operator()<int64_t>(x, /*threshold=*/3);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{5}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 0); // equal to threshold → 0
  EXPECT_EQ(py[2], 1);
  EXPECT_EQ(py[3], 0);
  EXPECT_EQ(py[4], 1);
}

TEST(BackendKernelClass, BinarizerInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Binarizer binarizer{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-0.5f, 0.5f, 1.5f});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {3},
             std::vector<uint8_t>(3 * sizeof(float), 0u));
  binarizer.operator()<float>(x, /*threshold=*/0.0f, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], 0.0f);
  EXPECT_FLOAT_EQ(po[1], 1.0f);
  EXPECT_FLOAT_EQ(po[2], 1.0f);
}

TEST(BackendKernelClass, BinarizerRejectsWrongInputDtype) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Binarizer binarizer{ctx};
  Tensor x = Tensor::FromFloat("", {1}, {1.0f});
  EXPECT_THROW(((void)binarizer.operator()<int64_t>(x, /*threshold=*/0)), std::invalid_argument);
}

TEST(BackendKernelClass, BinarizerRejectsMismatchedPreallocatedOutputShape) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Binarizer binarizer{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-0.5f, 0.5f, 1.5f});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {2},
             std::vector<uint8_t>(2 * sizeof(float), 0u));
  EXPECT_THROW(binarizer.operator()<float>(x, /*threshold=*/0.0f, out), std::invalid_argument);
}

TEST(BackendKernelClass, ScalerPerFeatureFloat) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Scaler scaler{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  const std::vector<float> offset{0.5f, 1.0f, 1.5f};
  const std::vector<float> scale{2.0f, 0.5f, 1.0f};
  Tensor y = scaler.operator()<float>(x, offset, scale);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], (0.0f - 0.5f) * 2.0f);
  EXPECT_FLOAT_EQ(py[1], (1.0f - 1.0f) * 0.5f);
  EXPECT_FLOAT_EQ(py[2], (2.0f - 1.5f) * 1.0f);
  EXPECT_FLOAT_EQ(py[3], (3.0f - 0.5f) * 2.0f);
  EXPECT_FLOAT_EQ(py[4], (4.0f - 1.0f) * 0.5f);
  EXPECT_FLOAT_EQ(py[5], (5.0f - 1.5f) * 1.0f);
}

TEST(BackendKernelClass, ScalerBroadcastInt64ProducesFloat) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Scaler scaler{ctx};
  Tensor x = Tensor::FromInt64("", {5}, {0, 1, 2, 3, 4});
  const std::vector<float> offset{1.0f};
  const std::vector<float> scale{0.5f};
  Tensor y = scaler.operator()<int64_t>(x, offset, scale);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{5}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -0.5f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 0.5f);
  EXPECT_FLOAT_EQ(py[3], 1.0f);
  EXPECT_FLOAT_EQ(py[4], 1.5f);
}

TEST(BackendKernelClass, ScalerInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Scaler scaler{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {3},
             std::vector<uint8_t>(3 * sizeof(float), 0u));
  scaler.operator()<float>(x, /*offset=*/{0.5f}, /*scale=*/{2.0f}, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], 1.0f);
  EXPECT_FLOAT_EQ(po[1], 3.0f);
  EXPECT_FLOAT_EQ(po[2], 5.0f);
}

TEST(BackendKernelClass, ScalerRejectsMismatchedOffsetScaleSizes) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Scaler scaler{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  EXPECT_THROW(((void)scaler.operator()<float>(x, /*offset=*/{0.0f, 0.0f}, /*scale=*/{1.0f})),
               std::invalid_argument);
}

TEST(BackendKernelClass, ScalerRejectsOffsetSizeNotMatchingLastDim) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Scaler scaler{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  // length 2 does not match last dim 3 and is not 1.
  EXPECT_THROW(((void)scaler.operator()<float>(x, /*offset=*/{0.0f, 0.0f}, /*scale=*/{1.0f, 1.0f})),
               std::invalid_argument);
}

TEST(BackendKernelClass, ScalerRejectsWrongInputDtype) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Scaler scaler{ctx};
  Tensor x = Tensor::FromFloat("", {1}, {1.0f});
  EXPECT_THROW(((void)scaler.operator()<int64_t>(x, /*offset=*/{0.0f}, /*scale=*/{1.0f})),
               std::invalid_argument);
}

TEST(BackendKernelClass, ArrayFeatureExtractorGathersAlongLastAxis) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  ArrayFeatureExtractor afe{ctx};
  Tensor x = Tensor::FromFloat(
      "", {3, 4}, {0.0f, 1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f});
  Tensor y = Tensor::FromInt64("", {3}, {0, 2, 3});
  Tensor z = afe.operator()<float>(x, y);
  ASSERT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(z.shape, (std::vector<int64_t>{3, 3}));
  const float *pz = z.AsFloat();
  const std::vector<float> expected{0.0f, 2.0f, 3.0f, 10.0f, 12.0f, 13.0f, 20.0f, 22.0f, 23.0f};
  for (int64_t i = 0; i < static_cast<int64_t>(expected.size()); ++i) {
    EXPECT_FLOAT_EQ(pz[i], expected[static_cast<size_t>(i)]);
  }
}

TEST(BackendKernelClass, ArrayFeatureExtractorInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  ArrayFeatureExtractor afe{ctx};
  Tensor x = Tensor::FromInt64("", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor y = Tensor::FromInt64("", {2}, {3, 1});
  Tensor out("", onnx_backend_test::DataType::INT64, {2, 2},
             std::vector<uint8_t>(4 * sizeof(int64_t), 0u));
  afe.operator()<int64_t>(x, y, out);
  const int64_t *po = out.AsInt64();
  EXPECT_EQ(po[0], 4);
  EXPECT_EQ(po[1], 2);
  EXPECT_EQ(po[2], 8);
  EXPECT_EQ(po[3], 6);
}

TEST(BackendKernelClass, ArrayFeatureExtractorRejectsOutOfBoundsIndex) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  ArrayFeatureExtractor afe{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f});
  Tensor y = Tensor::FromInt64("", {1}, {4});
  EXPECT_THROW(((void)afe.operator()<float>(x, y)), std::invalid_argument);
}

TEST(BackendKernelClass, ZipMapInt64LabelsCopiesFloatScores) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  ZipMap zipmap{ctx};
  const std::vector<int64_t> labels{10, 20, 30};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.1f, 0.7f, 0.2f, 0.3f, 0.4f, 0.3f});
  Tensor y = zipmap(x, labels);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.1f);
  EXPECT_FLOAT_EQ(py[1], 0.7f);
  EXPECT_FLOAT_EQ(py[2], 0.2f);
  EXPECT_FLOAT_EQ(py[3], 0.3f);
  EXPECT_FLOAT_EQ(py[4], 0.4f);
  EXPECT_FLOAT_EQ(py[5], 0.3f);
}

TEST(BackendKernelClass, ZipMapRank1ExpandsToSingleRow) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  ZipMap zipmap{ctx};
  const std::vector<std::string> labels{"a", "b", "c"};
  Tensor x = Tensor::FromFloat("", {3}, {0.1f, 0.7f, 0.2f});
  Tensor y = zipmap(x, labels);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 3}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.1f);
  EXPECT_FLOAT_EQ(py[1], 0.7f);
  EXPECT_FLOAT_EQ(py[2], 0.2f);
}

TEST(BackendKernelClass, ZipMapInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  ZipMap zipmap{ctx};
  const std::vector<int64_t> labels{10, 20, 30};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.1f, 0.7f, 0.2f, 0.3f, 0.4f, 0.3f});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {2, 3},
             std::vector<uint8_t>(6 * sizeof(float), 0u));
  zipmap(x, labels, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], 0.1f);
  EXPECT_FLOAT_EQ(po[1], 0.7f);
  EXPECT_FLOAT_EQ(po[2], 0.2f);
  EXPECT_FLOAT_EQ(po[3], 0.3f);
  EXPECT_FLOAT_EQ(po[4], 0.4f);
  EXPECT_FLOAT_EQ(po[5], 0.3f);
}

TEST(BackendKernelClass, ZipMapRejectsMismatchedClassCount) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  ZipMap zipmap{ctx};
  const std::vector<int64_t> labels{10, 20};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.1f, 0.7f, 0.2f, 0.3f, 0.4f, 0.3f});
  EXPECT_THROW(((void)zipmap(x, labels)), std::invalid_argument);
}

TEST(BackendKernelClass, OneHotEncoderInt64MatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  OneHotEncoder one_hot{ctx};
  const std::vector<int64_t> cats{0, 1, 2, 3};
  Tensor x = Tensor::FromInt64("", {3}, {0, 2, 7});
  Tensor y = one_hot.operator()<int64_t>(x, cats, /*zeros=*/true);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 4}));
  const float *py = y.AsFloat();
  // x=0 -> [1,0,0,0]; x=2 -> [0,0,1,0]; x=7 -> [0,0,0,0] (zeros=true)
  const std::vector<float> expected{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, OneHotEncoderStringMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  OneHotEncoder one_hot{ctx};
  const std::vector<std::string> cats{"a", "b", "c"};
  Tensor x = Tensor::FromStrings("", {4}, {"a", "b", "d", "c"});
  Tensor y = one_hot(x, cats, /*zeros=*/true);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4, 3}));
  const float *py = y.AsFloat();
  const std::vector<float> expected{1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, OneHotEncoderFloatInputCastsToInt64) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  OneHotEncoder one_hot{ctx};
  const std::vector<int64_t> cats{0, 1, 2};
  Tensor x = Tensor::FromFloat("", {2}, {1.7f, 2.0f});
  Tensor y = one_hot.operator()<float>(x, cats, /*zeros=*/true);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const float *py = y.AsFloat();
  // 1.7 -> cast to 1 -> [0,1,0]; 2.0 -> 2 -> [0,0,1]
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[1], 1.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
  EXPECT_FLOAT_EQ(py[4], 0.0f);
  EXPECT_FLOAT_EQ(py[5], 1.0f);
}

TEST(BackendKernelClass, OneHotEncoderInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  OneHotEncoder one_hot{ctx};
  const std::vector<int64_t> cats{0, 1, 2};
  Tensor x = Tensor::FromInt64("", {2}, {1, 0});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {2, 3},
             std::vector<uint8_t>(6 * sizeof(float), 0u));
  one_hot.operator()<int64_t>(x, cats, /*zeros=*/true, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], 0.0f);
  EXPECT_FLOAT_EQ(po[1], 1.0f);
  EXPECT_FLOAT_EQ(po[2], 0.0f);
  EXPECT_FLOAT_EQ(po[3], 1.0f);
  EXPECT_FLOAT_EQ(po[4], 0.0f);
  EXPECT_FLOAT_EQ(po[5], 0.0f);
}

TEST(BackendKernelClass, OneHotEncoderThrowsWhenZerosFalseAndValueMissing) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  OneHotEncoder one_hot{ctx};
  const std::vector<int64_t> cats{0, 1};
  Tensor x = Tensor::FromInt64("", {2}, {0, 5});
  EXPECT_THROW(((void)one_hot.operator()<int64_t>(x, cats, /*zeros=*/false)),
               std::invalid_argument);
}

TEST(BackendKernelClass, OneHotEncoderRejectsWrongInputDtype) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  OneHotEncoder one_hot{ctx};
  const std::vector<int64_t> cats{0, 1};
  Tensor x = Tensor::FromFloat("", {1}, {0.0f});
  EXPECT_THROW(((void)one_hot.operator()<int64_t>(x, cats, /*zeros=*/true)), std::invalid_argument);
}

TEST(BackendKernelClass, OneHotEncoderRejectsMismatchedPreallocatedOutputShape) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  OneHotEncoder one_hot{ctx};
  const std::vector<int64_t> cats{0, 1, 2};
  Tensor x = Tensor::FromInt64("", {2}, {1, 0});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {2, 2},
             std::vector<uint8_t>(4 * sizeof(float), 0u));
  EXPECT_THROW(one_hot.operator()<int64_t>(x, cats, /*zeros=*/true, out), std::invalid_argument);
}

TEST(BackendKernelClass, SVMClassifierInt64LabelsBinaryLinear) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  SVMClassifier svm{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 1.0f, 0.0f, 3.0f});
  auto yz = svm.operator()<float>(x, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, -1.0f}, {0.0f}, {1, 1},
                                  {0, 1}, "LINEAR", 0.0f, 0.0f, 0.0f);
  ASSERT_EQ(yz.first.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_EQ(yz.first.shape, (std::vector<int64_t>{2}));
  ASSERT_EQ(yz.second.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(yz.second.shape, (std::vector<int64_t>{2, 1}));
  const int64_t *labels = yz.first.AsInt64();
  const float *scores = yz.second.AsFloat();
  EXPECT_EQ(labels[0], 1);
  EXPECT_EQ(labels[1], 0);
  EXPECT_FLOAT_EQ(scores[0], 1.0f);
  EXPECT_FLOAT_EQ(scores[1], -3.0f);
}

TEST(BackendKernelClass, SVMClassifierStringLabelsBinaryLinear) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  SVMClassifier svm{ctx};
  Tensor x = Tensor::FromFloat("", {1, 2}, {0.0f, 2.0f});
  auto yz =
      svm.operator()<float>(x, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, -1.0f}, {0.0f}, {1, 1},
                            std::vector<std::string>{"neg", "pos"}, "LINEAR", 0.0f, 0.0f, 0.0f);
  ASSERT_EQ(yz.first.data_type, static_cast<int32_t>(TensorProto::DataType::STRING));
  ASSERT_EQ(yz.first.shape, (std::vector<int64_t>{1}));
  const auto &labels = yz.first.AsStrings();
  ASSERT_EQ(labels.size(), 1u);
  EXPECT_EQ(labels[0], "neg");
}

TEST(BackendKernelClass, SVMRegressorLinearKernelMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  SVMRegressor svm{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {3.0f, 1.0f, 0.0f, 2.0f});
  Tensor y = svm.operator()<float>(x, {1.0f, 0.0f, 0.0f, 1.0f}, {2.0f, -1.0f}, {0.5f}, "LINEAR",
                                   0.0f, 0.0f, 0.0f);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 4.5f);
  EXPECT_FLOAT_EQ(py[1], -2.5f);
}

TEST(BackendKernelClass, LinearRegressorMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  LinearRegressor reg{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 1.0f, 0.0f, 3.0f});
  Tensor y = reg.operator()<float>(x, {0.5f, -1.0f}, {0.25f}, 1, "NONE");
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const float *py = y.AsFloat();
  // 2*0.5 + 1*(-1) + 0.25 = 0.25 ; 0*0.5 + 3*(-1) + 0.25 = -2.75
  EXPECT_FLOAT_EQ(py[0], 0.25f);
  EXPECT_FLOAT_EQ(py[1], -2.75f);
}

TEST(BackendKernelClass, LinearRegressorMultiTargetMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  LinearRegressor reg{ctx};
  Tensor x = Tensor::FromFloat("", {1, 2}, {1.0f, 2.0f});
  // 2 targets, 2 features → coefficients laid out [t0_c0, t0_c1, t1_c0, t1_c1]
  Tensor y = reg.operator()<float>(x, {1.0f, 0.0f, 0.0f, 1.0f}, {}, 2, "NONE");
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
}

TEST(BackendKernelClass, LinearClassifierInt64BinaryMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  LinearClassifier cls{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 1.0f, 0.0f, 3.0f});
  auto yz = cls.operator()<float>(x, {1.0f, -1.0f}, {0.0f}, std::vector<int64_t>{0, 1}, "NONE");
  ASSERT_EQ(yz.first.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_EQ(yz.first.shape, (std::vector<int64_t>{2}));
  ASSERT_EQ(yz.second.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  // Binary classifier with single intercept expands to [-z, z] (E_out=2).
  ASSERT_EQ(yz.second.shape, (std::vector<int64_t>{2, 2}));
  const int64_t *labels = yz.first.AsInt64();
  const float *scores = yz.second.AsFloat();
  // sample 0: z = 2*1 + 1*(-1) + 0 = 1   -> argmax of [-1, 1] = 1
  // sample 1: z = 0*1 + 3*(-1) + 0 = -3  -> argmax of [3, -3] = 0
  EXPECT_EQ(labels[0], 1);
  EXPECT_EQ(labels[1], 0);
  EXPECT_FLOAT_EQ(scores[0], -1.0f);
  EXPECT_FLOAT_EQ(scores[1], 1.0f);
  EXPECT_FLOAT_EQ(scores[2], 3.0f);
  EXPECT_FLOAT_EQ(scores[3], -3.0f);
}

TEST(BackendKernelClass, LinearClassifierStringMulticlassMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  LinearClassifier cls{ctx};
  // 3 classes, 2 features: rows of W are [1,0], [0,1], [-1,-1]; intercepts zero.
  Tensor x = Tensor::FromFloat("", {2, 2}, {3.0f, 1.0f, 0.0f, 2.0f});
  auto yz = cls.operator()<float>(x, {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 0.0f},
                                  std::vector<std::string>{"a", "b", "c"}, "NONE");
  ASSERT_EQ(yz.first.data_type, static_cast<int32_t>(TensorProto::DataType::STRING));
  ASSERT_EQ(yz.first.shape, (std::vector<int64_t>{2}));
  ASSERT_EQ(yz.second.shape, (std::vector<int64_t>{2, 3}));
  const auto &labels = yz.first.AsStrings();
  ASSERT_EQ(labels.size(), 2u);
  // sample 0 scores: [3, 1, -4] → argmax = a
  // sample 1 scores: [0, 2, -2] → argmax = b
  EXPECT_EQ(labels[0], "a");
  EXPECT_EQ(labels[1], "b");
}

TEST(BackendKernelClass, ImputerFloatReplacesMatchingElements) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Imputer imputer{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 1.0f, 0.0f, 5.0f, 0.0f, 6.0f});
  const std::vector<float> imputed{1.0f, 2.0f, 3.0f};
  Tensor y = imputer.operator()<float>(x, imputed, 0.0f);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f); // 0.0 replaced by imputed[0 % 3 = 0]
  EXPECT_FLOAT_EQ(py[1], 1.0f); // 1.0 unchanged
  EXPECT_FLOAT_EQ(py[2], 3.0f); // 0.0 replaced by imputed[2 % 3 = 2]
  EXPECT_FLOAT_EQ(py[3], 5.0f); // 5.0 unchanged
  EXPECT_FLOAT_EQ(py[4], 2.0f); // 0.0 replaced by imputed[4 % 3 = 1]
  EXPECT_FLOAT_EQ(py[5], 6.0f); // 6.0 unchanged
}

TEST(BackendKernelClass, ImputerFloatBroadcastReplacement) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Imputer imputer{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {-1.0f, 2.0f, -1.0f, 4.0f});
  const std::vector<float> imputed{0.0f};
  Tensor y = imputer.operator()<float>(x, imputed, -1.0f);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 4.0f);
}

TEST(BackendKernelClass, ImputerInt64ReplacesMatchingElements) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Imputer imputer{ctx};
  Tensor x = Tensor::FromInt64("", {3, 2}, {0, 0, 1, 2, 0, 3});
  const std::vector<int64_t> imputed{10, 20};
  Tensor y = imputer.operator()<int64_t>(x, imputed, static_cast<int64_t>(0));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 10); // 0 replaced by imputed[0]
  EXPECT_EQ(py[1], 20); // 0 replaced by imputed[1]
  EXPECT_EQ(py[2], 1);  // 1 unchanged
  EXPECT_EQ(py[3], 2);  // 2 unchanged
  EXPECT_EQ(py[4], 10); // 0 replaced by imputed[0]
  EXPECT_EQ(py[5], 3);  // 3 unchanged
}

TEST(BackendKernelClass, ImputerInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Imputer imputer{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {0.0f, 5.0f, 0.0f});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {3},
             std::vector<uint8_t>(3 * sizeof(float), 0u));
  imputer.operator()<float>(x, std::vector<float>{9.0f}, 0.0f, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], 9.0f);
  EXPECT_FLOAT_EQ(po[1], 5.0f);
  EXPECT_FLOAT_EQ(po[2], 9.0f);
}

TEST(BackendKernelClass, ImputerRejectsWrongInputDtype) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  Imputer imputer{ctx};
  Tensor x = Tensor::FromInt64("", {3}, {0, 1, 2});
  EXPECT_THROW((imputer.operator()<float>(x, std::vector<float>{1.0f}, 0.0f)),
               std::invalid_argument);
}

TEST(BackendKernelClass, TreeEnsembleRegressorSumSingleTargetMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  onnx_backend_test::kernel::TreeEnsembleRegressor reg{ctx};
  // Single-tree: node 0 splits feature[0] <= 1.0; true -> leaf weight 1.0, false -> 3.0.
  Tensor x = Tensor::FromFloat("", {2, 1}, {0.5f, 2.0f});
  const std::vector<int64_t> treeids{0, 0, 0};
  const std::vector<int64_t> nodeids{0, 1, 2};
  const std::vector<int64_t> featureids{0, 0, 0};
  const std::vector<float> values{1.0f, 0.0f, 0.0f};
  const std::vector<std::string> modes{"BRANCH_LEQ", "LEAF", "LEAF"};
  const std::vector<int64_t> truenodes{1, 0, 0};
  const std::vector<int64_t> falsenodes{2, 0, 0};
  const std::vector<int64_t> missing{};
  const std::vector<int64_t> t_treeids{0, 0};
  const std::vector<int64_t> t_nodeids{1, 2};
  const std::vector<int64_t> t_ids{0, 0};
  const std::vector<float> t_weights{1.0f, 3.0f};
  Tensor y = reg.operator()<float>(x, treeids, nodeids, featureids, values, modes, truenodes,
                                   falsenodes, missing, t_treeids, t_nodeids, t_ids, t_weights,
                                   /*n_targets=*/1, /*aggregate_function=*/"SUM",
                                   /*post_transform=*/"NONE", /*base_values=*/{});
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const float *py = y.AsFloat();
  // sample 0: feature[0]=0.5 <= 1.0 -> leaf at node 1 -> weight 1.0
  // sample 1: feature[0]=2.0  > 1.0 -> leaf at node 2 -> weight 3.0
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
}

TEST(BackendKernelClass, TreeEnsembleClassifierInt64BinaryMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  onnx_backend_test::kernel::TreeEnsembleClassifier cls{ctx};
  // Single-tree binary: node 0 splits feature[0] <= 0.5; true -> class 0, false -> class 1.
  Tensor x = Tensor::FromFloat("", {2, 1}, {0.0f, 1.0f});
  const std::vector<int64_t> treeids{0, 0, 0};
  const std::vector<int64_t> nodeids{0, 1, 2};
  const std::vector<int64_t> featureids{0, 0, 0};
  const std::vector<float> values{0.5f, 0.0f, 0.0f};
  const std::vector<std::string> modes{"BRANCH_LEQ", "LEAF", "LEAF"};
  const std::vector<int64_t> truenodes{1, 0, 0};
  const std::vector<int64_t> falsenodes{2, 0, 0};
  const std::vector<int64_t> missing{};
  const std::vector<int64_t> c_treeids{0, 0};
  const std::vector<int64_t> c_nodeids{1, 2};
  const std::vector<int64_t> c_ids{0, 1};
  const std::vector<float> c_weights{1.0f, 1.0f};
  auto yz = cls.operator()<float>(x, treeids, nodeids, featureids, values, modes, truenodes,
                                  falsenodes, missing, c_treeids, c_nodeids, c_ids, c_weights,
                                  std::vector<int64_t>{0, 1}, {}, "NONE");
  ASSERT_EQ(yz.first.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_EQ(yz.first.shape, (std::vector<int64_t>{2}));
  const int64_t *labels = yz.first.AsInt64();
  EXPECT_EQ(labels[0], 0); // x[0]=0.0 <= 0.5 -> class 0
  EXPECT_EQ(labels[1], 1); // x[1]=1.0  > 0.5 -> class 1
}

TEST(BackendKernelClass, TreeEnsembleV5SingleTreeMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 5)};
  onnx_backend_test::kernel::TreeEnsemble te{ctx};
  // Single-tree v5: root=0, node 0 splits feature[0] LEQ 0.5 (mode=0).
  //   true  -> leaf index 0 (target 0, weight 1.0)
  //   false -> leaf index 1 (target 0, weight 2.0)
  Tensor x = Tensor::FromFloat("", {2, 1}, {0.0f, 1.0f});
  Tensor y =
      te.operator()<float>(x, {0}, {0}, {0.5f}, {0}, {0}, {1}, {1}, {1}, {}, {0, 0}, {1.0f, 2.0f},
                           /*n_targets=*/1, /*aggregate_function=*/1, /*post_transform=*/0);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const float *py = y.AsFloat();
  // sample 0: feature[0]=0.0 <= 0.5 -> leaf 0 -> weight 1.0
  // sample 1: feature[0]=1.0  > 0.5 -> leaf 1 -> weight 2.0
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
}

} // namespace Test