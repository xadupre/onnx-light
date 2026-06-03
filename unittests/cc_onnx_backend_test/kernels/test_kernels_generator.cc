// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"
#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::Bernoulli;
using onnx_backend_test::kernel::Constant;
using onnx_backend_test::kernel::ConstantOfShape;
using onnx_backend_test::kernel::EyeLike;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Range;

namespace Test {

TEST(BackendKernelClass, ConstantClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Constant constant_kernel{ctx};
  Tensor value = Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.5f, 0.0f});
  Tensor y = constant_kernel(value);
  ASSERT_EQ(y.data_type, value.data_type);
  EXPECT_EQ(y.shape, value.shape);
  ASSERT_EQ(y.element_count(), value.element_count());
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], -2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.5f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
}

TEST(BackendKernelClass, ConstantRejectsMismatchedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Constant constant_kernel{ctx};
  Tensor value = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor bad_shape("", onnx_backend_test::DataType::FLOAT, {3},
                   std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(constant_kernel(value, bad_shape), std::invalid_argument);
  Tensor bad_type("", onnx_backend_test::DataType::INT32, {2},
                  std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(constant_kernel(value, bad_type), std::invalid_argument);
}

TEST(BackendKernelClass, ConstantOfShapeFloatOnes) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {3}, {2, 3, 1});
  const Tensor value = Tensor::FromFloat("", {1}, {1.0f});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 3, 1}));
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(py[i], 1.0f);
  }
}

TEST(BackendKernelClass, ConstantOfShapeInt64Fill) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {2}, {2, 2});
  const Tensor value = Tensor::FromInt64("", {1}, {static_cast<int64_t>(-7)});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], -7);
  EXPECT_EQ(py[1], -7);
  EXPECT_EQ(py[2], -7);
  EXPECT_EQ(py[3], -7);
}

TEST(BackendKernelClass, ConstantOfShapeDefaultValueIsFloatZero) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {1}, {static_cast<int64_t>(4)});
  Tensor y = kernel(shape, Tensor());
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{4}));
  const float *py = y.AsFloat();
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(py[i], 0.0f);
  }
}

TEST(BackendKernelClass, ConstantOfShapeEmptyShapeProducesScalar) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  // An empty 1-D ``shape`` input produces a scalar output.
  const Tensor shape = Tensor::FromInt64("", {0}, {});
  const Tensor value = Tensor::FromFloat("", {1}, {2.5f});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.shape, (std::vector<int64_t>{}));
  ASSERT_EQ(y.element_count(), 1);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 2.5f);
}

TEST(BackendKernelClass, ConstantOfShapeRejectsNonInt64Shape) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor bad_shape = Tensor::FromInt32("", {2}, {2, 3});
  const Tensor value = Tensor::FromFloat("", {1}, {0.0f});
  EXPECT_THROW(kernel(bad_shape, value), std::invalid_argument);
}

TEST(BackendKernelClass, EyeLikeDefaultDtypeAndMainDiagonal) {
  const KernelContext ctx{DefaultOpset(22)};
  EyeLike kernel{ctx};
  const Tensor x = Tensor::FromInt32("", {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor y = kernel(x);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 4}));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 1);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[4], 0);
  EXPECT_EQ(py[5], 1);
  EXPECT_EQ(py[10], 1);
  EXPECT_EQ(py[11], 0);
}

TEST(BackendKernelClass, EyeLikeDtypeOverrideAndUpperDiagonal) {
  const KernelContext ctx{DefaultOpset(22)};
  EyeLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  Tensor y = kernel(x, /*k=*/1, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 4}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 0);
  EXPECT_EQ(py[3], 0);
  EXPECT_EQ(py[4], 0);
  EXPECT_EQ(py[5], 0);
  EXPECT_EQ(py[6], 1);
  EXPECT_EQ(py[7], 0);
}

TEST(BackendKernelClass, EyeLikeFloatOutputUsesOneValue) {
  const KernelContext ctx{DefaultOpset(22)};
  EyeLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
  Tensor y = kernel(x);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 1.0f);
}

TEST(BackendKernelClass, EyeLikeRejectsNonMatrixInput) {
  const KernelContext ctx{DefaultOpset(22)};
  EyeLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_THROW(kernel(x), std::invalid_argument);
}

TEST(BackendKernelClass, BernoulliPreservesShapeAndProducesZeroOrOne) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const std::vector<float> probs = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 0.5f};
  const Tensor x = Tensor::FromFloat("", {2, 3}, probs);
  Tensor y = kernel(x);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(py[i] == 0.0f || py[i] == 1.0f);
  }
  // Endpoints: probability 0 must produce 0 and probability 1 must produce 1.
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[4], 1.0f);
}

TEST(BackendKernelClass, BernoulliIsDeterministicAcrossInvocations) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {8}, {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.2f, 0.4f, 0.6f});
  Tensor a = kernel(x);
  Tensor b = kernel(x);
  ASSERT_EQ(a.data.size(), b.data.size());
  EXPECT_EQ(a.data, b.data);
}

TEST(BackendKernelClass, BernoulliRespectsSeedAttribute) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {8}, {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.2f, 0.4f, 0.6f});
  Tensor a = kernel(x, /*seed=*/123, /*dtype=*/0);
  Tensor b = kernel(x, /*seed=*/123, /*dtype=*/0);
  EXPECT_EQ(a.data, b.data);
  Tensor c = kernel(x, /*seed=*/124, /*dtype=*/0);
  // Different seeds are very likely to produce a different draw sequence.
  EXPECT_NE(a.data, c.data);
}

TEST(BackendKernelClass, BernoulliDtypeAttributeOverridesOutputType) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {4}, {0.0f, 1.0f, 0.0f, 1.0f});
  Tensor y = kernel(x, Bernoulli::kNoSeed,
                    /*dtype=*/static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{4}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 0);
  EXPECT_EQ(py[3], 1);
}

TEST(BackendKernelClass, BernoulliRejectsOutOfRangeProbability) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor bad = Tensor::FromFloat("", {2}, {0.5f, 1.5f});
  EXPECT_THROW(kernel(bad), std::invalid_argument);
}

TEST(BackendKernelClass, BernoulliRejectsUnsupportedInputDtype) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor int_in = Tensor::FromInt32("", {2}, {0, 1});
  EXPECT_THROW(kernel(int_in), std::invalid_argument);
}

TEST(BackendKernelClass, RangeFloatPositiveDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromFloat("", {}, {1.0f});
  const Tensor limit = Tensor::FromFloat("", {}, {5.0f});
  const Tensor delta = Tensor::FromFloat("", {}, {2.0f});
  const Tensor y = kernel(start, limit, delta);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
}

TEST(BackendKernelClass, RangeInt32NegativeDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromInt32("", {}, {10});
  const Tensor limit = Tensor::FromInt32("", {}, {6});
  const Tensor delta = Tensor::FromInt32("", {}, {-3});
  const Tensor y = kernel(start, limit, delta);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 10);
  EXPECT_EQ(py[1], 7);
}

TEST(BackendKernelClass, RangeEmptyWhenStartExceedsLimitForPositiveDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromInt64("", {}, {5});
  const Tensor limit = Tensor::FromInt64("", {}, {5});
  const Tensor delta = Tensor::FromInt64("", {}, {1});
  const Tensor y = kernel(start, limit, delta);
  EXPECT_EQ(y.shape, (std::vector<int64_t>{0}));
  EXPECT_EQ(y.element_count(), 0);
}

TEST(BackendKernelClass, RangeRejectsMismatchedDtypes) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromFloat("", {}, {1.0f});
  const Tensor limit = Tensor::FromInt32("", {}, {5});
  const Tensor delta = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW(kernel(start, limit, delta), std::invalid_argument);
}

TEST(BackendKernelClass, RangeRejectsZeroDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromInt64("", {}, {0});
  const Tensor limit = Tensor::FromInt64("", {}, {5});
  const Tensor delta = Tensor::FromInt64("", {}, {0});
  EXPECT_THROW(kernel(start, limit, delta), std::invalid_argument);
}

} // namespace Test
