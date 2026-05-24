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
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::LabelEncoder;

namespace Test {

TEST(BackendKernelClass, LabelEncoderInt64ToFloatMatchesReference) {
  LabelEncoder label_encoder{KernelContext(OpsetId("ai.onnx.ml", 4))};
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
  LabelEncoder label_encoder{KernelContext(OpsetId("ai.onnx.ml", 4))};
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
  LabelEncoder label_encoder{KernelContext(OpsetId("ai.onnx.ml", 4))};
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
  LabelEncoder label_encoder{KernelContext(OpsetId("ai.onnx.ml", 4))};
  const std::vector<int64_t> keys{0, 1, 2};
  const std::vector<float> values{0.5f, 1.5f};
  Tensor x = Tensor::FromInt64("", {1}, {0});
  EXPECT_THROW(((void)label_encoder.operator()<int64_t, float>(x, keys, values, 0.0f)),
               std::invalid_argument);
}

TEST(BackendKernelClass, LabelEncoderRejectsWrongInputDtype) {
  LabelEncoder label_encoder{KernelContext(OpsetId("ai.onnx.ml", 4))};
  const std::vector<int64_t> keys{0, 1};
  const std::vector<float> values{0.5f, 1.5f};
  Tensor x = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  EXPECT_THROW(((void)label_encoder.operator()<int64_t, float>(x, keys, values, 0.0f)),
               std::invalid_argument);
}

} // namespace Test
