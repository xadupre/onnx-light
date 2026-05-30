// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::OpsetId;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::ArrayFeatureExtractor;
using onnx_backend_test::kernel::Binarizer;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::LabelEncoder;
using onnx_backend_test::kernel::ZipMap;

namespace Test {

TEST(BackendKernelClass, LabelEncoderInt64ToFloatMatchesReference) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 4)};
  LabelEncoder label_encoder{ctx};
  const std::vector<int64_t> keys{0, 1, 2};
  const std::vector<float> values{0.5f, 1.5f, 2.5f};
  Tensor x = Tensor::FromInt64("", {4}, {0, 1, 2, 7});
  Tensor y = label_encoder.operator()<int64_t, float>(x, keys, values, /*default=*/-1.0f);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
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
  Tensor out("", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float), 0u));
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
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
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
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT16));
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
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
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
  Tensor out("", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float), 0u));
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
  Tensor out("", TensorProto::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float), 0u));
  EXPECT_THROW(binarizer.operator()<float>(x, /*threshold=*/0.0f, out), std::invalid_argument);
}

TEST(BackendKernelClass, ArrayFeatureExtractorGathersAlongLastAxis) {
  const KernelContext ctx{OpsetId("ai.onnx.ml", 1)};
  ArrayFeatureExtractor afe{ctx};
  Tensor x = Tensor::FromFloat(
      "", {3, 4}, {0.0f, 1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f});
  Tensor y = Tensor::FromInt64("", {3}, {0, 2, 3});
  Tensor z = afe.operator()<float>(x, y);
  ASSERT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
  Tensor out("", TensorProto::DataType::INT64, {2, 2},
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
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
  Tensor out("", TensorProto::DataType::FLOAT, {2, 3}, std::vector<uint8_t>(6 * sizeof(float), 0u));
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

} // namespace Test
