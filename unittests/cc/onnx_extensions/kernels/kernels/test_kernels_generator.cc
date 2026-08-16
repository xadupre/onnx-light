// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::RuntimeContext;
using core::runtime::Tensor;
using onnx_kernels::Shape;
using onnx_kernels::SimpleRawBufferAllocator;
using onnx_kernels::kernel::Bernoulli;
using onnx_kernels::kernel::Constant;
using onnx_kernels::kernel::ConstantOfShape;
using onnx_kernels::kernel::EyeLike;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::Multinomial;
using onnx_kernels::kernel::RandomNormal;
using onnx_kernels::kernel::RandomNormalLike;
using onnx_kernels::kernel::RandomUniform;
using onnx_kernels::kernel::RandomUniformLike;
using onnx_kernels::kernel::Range;

namespace Test {

namespace {

uint16_t ReadUint16Element(const Tensor &tensor, size_t index) {
  const uint8_t *bytes = tensor.bytes();
  const size_t offset = index * sizeof(uint16_t);
  return std::bit_cast<uint16_t>(
      std::array<uint8_t, sizeof(uint16_t)>{bytes[offset], bytes[offset + 1]});
}

} // namespace

TEST(KernelClass, ConstantClassMatchesReference) {
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
  EXPECT_EQ(y.bytes(), value.bytes());
  EXPECT_TRUE(y.data.empty());
}

TEST(KernelClass, ConstantStringClassBorrowsStringStorage) {
  const KernelContext ctx{DefaultOpset(13)};
  Constant constant_kernel{ctx};
  Tensor value = Tensor::FromStrings("", {2}, {std::string("hello"), std::string("world")});
  Tensor y = constant_kernel(value);
  const Tensor &value_const_ref = value;
  const Tensor &y_const_ref = y;
  EXPECT_EQ(y.shape, value.shape);
  ASSERT_EQ(y_const_ref.AsStrings().size(), 2u);
  EXPECT_EQ(y_const_ref.AsStrings()[0], "hello");
  EXPECT_EQ(y_const_ref.AsStrings()[1], "world");
  EXPECT_EQ(&y_const_ref.AsStrings(), &value_const_ref.AsStrings());
  EXPECT_TRUE(y.string_data.empty());
}

TEST(KernelClass, ConstantTemporaryInputRetainsBytes) {
  const KernelContext ctx{DefaultOpset(13)};
  Constant constant_kernel{ctx};
  Tensor y = constant_kernel(Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.5f, 0.0f}));
  ASSERT_FALSE(y.data.empty());
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], -2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.5f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
}

TEST(KernelClass, ConstantPreallocatedOutputStillCopiesBytes) {
  const KernelContext ctx{DefaultOpset(13)};
  Constant constant_kernel{ctx};
  Tensor value = Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.5f, 0.0f});
  Tensor output("", core::runtime::DataType::FLOAT, {2, 2},
                std::vector<uint8_t>(4 * sizeof(float)));
  constant_kernel(value, output);
  ASSERT_NE(output.bytes(), value.bytes());
  ASSERT_EQ(output.element_count(), value.element_count());
  const float *py = output.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], -2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.5f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
}

TEST(KernelClass, ConstantRejectsMismatchedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Constant constant_kernel{ctx};
  Tensor value = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor bad_shape("", core::runtime::DataType::FLOAT, {3},
                   std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(constant_kernel(value, bad_shape), std::invalid_argument);
  Tensor bad_type("", core::runtime::DataType::INT32, {2},
                  std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(constant_kernel(value, bad_type), std::invalid_argument);
}

TEST(KernelClass, ConstantOfShapeFloatOnes) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {3}, {2, 3, 1});
  const Tensor value = Tensor::FromFloat("", {1}, {1.0f});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 3, 1}));
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(py[i], 1.0f);
  }
}

TEST(KernelClass, ConstantOfShapeInt64Fill) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {2}, {2, 2});
  const Tensor value = Tensor::FromInt64("", {1}, {static_cast<int64_t>(-7)});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  EXPECT_EQ(y.shape, (Shape{2, 2}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], -7);
  EXPECT_EQ(py[1], -7);
  EXPECT_EQ(py[2], -7);
  EXPECT_EQ(py[3], -7);
}

TEST(KernelClass, ConstantOfShapeDefaultValueIsFloatZero) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {1}, {static_cast<int64_t>(4)});
  Tensor y = kernel(shape, Tensor());
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (Shape{4}));
  const float *py = y.AsFloat();
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(py[i], 0.0f);
  }
}

TEST(KernelClass, ConstantOfShapeEmptyShapeProducesScalar) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  // An empty 1-D ``shape`` input produces a scalar output.
  const Tensor shape = Tensor::FromInt64("", {0}, {});
  const Tensor value = Tensor::FromFloat("", {1}, {2.5f});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.shape, (Shape{}));
  ASSERT_EQ(y.element_count(), 1);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 2.5f);
}

TEST(KernelClass, ConstantOfShapeRejectsNonInt64Shape) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor bad_shape = Tensor::FromInt32("", {2}, {2, 3});
  const Tensor value = Tensor::FromFloat("", {1}, {0.0f});
  EXPECT_THROW(kernel(bad_shape, value), std::invalid_argument);
}

TEST(KernelClass, ConstantOfShapeUsesAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {2}, {3, 2});
  const Tensor value = Tensor::FromFloat("", {1}, {1.5f});
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});
  Tensor y = kernel(shape, value, &rt);
  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (Shape{3, 2}));
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(py[i], 1.5f);
  }
}

TEST(KernelClass, EyeLikeDefaultDtypeAndMainDiagonal) {
  const KernelContext ctx{DefaultOpset(22)};
  EyeLike kernel{ctx};
  const Tensor x = Tensor::FromInt32("", {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor y = kernel(x);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::INT32));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 4}));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 1);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[4], 0);
  EXPECT_EQ(py[5], 1);
  EXPECT_EQ(py[10], 1);
  EXPECT_EQ(py[11], 0);
}

TEST(KernelClass, EyeLikeDtypeOverrideAndUpperDiagonal) {
  const KernelContext ctx{DefaultOpset(22)};
  EyeLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  Tensor y = kernel(x, /*k=*/1, static_cast<int32_t>(core::runtime::DataType::INT64));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
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

TEST(KernelClass, EyeLikeFloatOutputUsesOneValue) {
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

TEST(KernelClass, EyeLikeFloat16AndBFloat16UseExpectedBitPatterns) {
  const KernelContext ctx{DefaultOpset(22)};
  EyeLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});

  const Tensor y_float16 =
      kernel(x, /*k=*/0, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  EXPECT_EQ(y_float16.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  EXPECT_EQ(ReadUint16Element(y_float16, 0), 0x3C00);
  EXPECT_EQ(ReadUint16Element(y_float16, 1), 0x0000);
  EXPECT_EQ(ReadUint16Element(y_float16, 2), 0x0000);
  EXPECT_EQ(ReadUint16Element(y_float16, 3), 0x3C00);

  const Tensor y_bfloat16 =
      kernel(x, /*k=*/0, static_cast<int32_t>(core::runtime::DataType::BFLOAT16));
  EXPECT_EQ(y_bfloat16.data_type, static_cast<int32_t>(core::runtime::DataType::BFLOAT16));
  EXPECT_EQ(ReadUint16Element(y_bfloat16, 0), 0x3F80);
  EXPECT_EQ(ReadUint16Element(y_bfloat16, 1), 0x0000);
  EXPECT_EQ(ReadUint16Element(y_bfloat16, 2), 0x0000);
  EXPECT_EQ(ReadUint16Element(y_bfloat16, 3), 0x3F80);
}

TEST(KernelClass, EyeLikeRejectsNonMatrixInput) {
  const KernelContext ctx{DefaultOpset(22)};
  EyeLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_THROW(kernel(x), std::invalid_argument);
}

TEST(KernelClass, BernoulliPreservesShapeAndProducesZeroOrOne) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const std::vector<float> probs = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 0.5f};
  const Tensor x = Tensor::FromFloat("", {2, 3}, probs);
  Tensor y = kernel(x);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
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

TEST(KernelClass, BernoulliIsDeterministicAcrossInvocations) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {8}, {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.2f, 0.4f, 0.6f});
  Tensor a = kernel(x);
  Tensor b = kernel(x);
  ASSERT_EQ(a.data.size(), b.data.size());
  EXPECT_EQ(a.data, b.data);
}

TEST(KernelClass, BernoulliRespectsSeedAttribute) {
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

TEST(KernelClass, BernoulliDtypeAttributeOverridesOutputType) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {4}, {0.0f, 1.0f, 0.0f, 1.0f});
  Tensor y = kernel(x, Bernoulli::kNoSeed,
                    /*dtype=*/static_cast<int32_t>(core::runtime::DataType::INT64));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{4}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 0);
  EXPECT_EQ(py[3], 1);
}

TEST(KernelClass, BernoulliRejectsOutOfRangeProbability) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor bad = Tensor::FromFloat("", {2}, {0.5f, 1.5f});
  EXPECT_THROW(kernel(bad), std::invalid_argument);
}

TEST(KernelClass, BernoulliRejectsUnsupportedInputDtype) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor int_in = Tensor::FromInt32("", {2}, {0, 1});
  EXPECT_THROW(kernel(int_in), std::invalid_argument);
}

TEST(KernelClass, BernoulliRejectsNegativeInputShape) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  Tensor bad;
  bad.data_type = static_cast<int32_t>(core::runtime::DataType::FLOAT);
  bad.shape = {-1};
  EXPECT_THROW(kernel(bad), std::invalid_argument);
}

TEST(KernelClass, BernoulliUsesAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(22)};
  Bernoulli kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {4}, {0.0f, 1.0f, 0.0f, 1.0f});
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});
  Tensor y = kernel(x, Bernoulli::kNoSeed, /*dtype=*/0, &rt);
  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{4}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[1], 1.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 1.0f);
}

TEST(KernelClass, MultinomialProducesInt32SamplesWithExpectedShape) {
  const KernelContext ctx{DefaultOpset(22)};
  Multinomial kernel{ctx};
  // Two batches, three classes; class 1 has overwhelmingly more probability mass.
  const Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 10.0f, 0.0f, 0.0f, 10.0f, 0.0f});
  Tensor y = kernel(x, /*sample_size=*/5);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::INT32));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 5}));
  ASSERT_EQ(y.element_count(), 10);
  const int32_t *py = y.AsInt32();
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(py[i], 1);
  }
}

TEST(KernelClass, MultinomialIsDeterministicForSameSeed) {
  const KernelContext ctx{DefaultOpset(22)};
  Multinomial kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {1, 4}, {0.1f, 0.2f, 0.3f, 0.4f});
  Tensor a = kernel(x, /*sample_size=*/6, /*seed=*/42, /*dtype=*/0);
  Tensor b = kernel(x, /*sample_size=*/6, /*seed=*/42, /*dtype=*/0);
  EXPECT_EQ(a.data, b.data);
  Tensor c = kernel(x, /*sample_size=*/6, /*seed=*/43, /*dtype=*/0);
  EXPECT_NE(a.data, c.data);
}

TEST(KernelClass, MultinomialDtypeAttributeOverridesOutputType) {
  const KernelContext ctx{DefaultOpset(22)};
  Multinomial kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {1, 2}, {10.0f, 0.0f});
  Tensor y = kernel(x, /*sample_size=*/3, Multinomial::kNoSeed,
                    /*dtype=*/static_cast<int32_t>(core::runtime::DataType::INT64));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 3}));
  const int64_t *py = y.AsInt64();
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(py[i], 0);
  }
}

TEST(KernelClass, MultinomialRejectsNon2DInput) {
  const KernelContext ctx{DefaultOpset(22)};
  Multinomial kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  EXPECT_THROW(kernel(x), std::invalid_argument);
}

TEST(KernelClass, MultinomialRejectsUnsupportedOutputDtype) {
  const KernelContext ctx{DefaultOpset(22)};
  Multinomial kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {1, 2}, {1.0f, 1.0f});
  EXPECT_THROW(kernel(x, /*sample_size=*/1, Multinomial::kNoSeed,
                      /*dtype=*/static_cast<int32_t>(core::runtime::DataType::FLOAT)),
               std::invalid_argument);
}

TEST(KernelClass, RandomNormalProducesFloatTensorOfRequestedShape) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomNormal kernel{ctx};
  Tensor y = kernel({2, 3});
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(y.element_count(), 6);
}

TEST(KernelClass, RandomNormalIsDeterministicForSameSeed) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomNormal kernel{ctx};
  Tensor a = kernel({4}, /*mean=*/0.0, /*scale=*/1.0, /*seed=*/42);
  Tensor b = kernel({4}, /*mean=*/0.0, /*scale=*/1.0, /*seed=*/42);
  EXPECT_EQ(a.data, b.data);
  Tensor c = kernel({4}, /*mean=*/0.0, /*scale=*/1.0, /*seed=*/43);
  EXPECT_NE(a.data, c.data);
}

TEST(KernelClass, RandomNormalDtypeOverrideProducesDouble) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomNormal kernel{ctx};
  Tensor y = kernel({3}, 0.0, 1.0, RandomNormal::kNoSeed,
                    static_cast<int32_t>(core::runtime::DataType::DOUBLE));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::DOUBLE));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3}));
}

TEST(KernelClass, RandomNormalRejectsUnsupportedDtype) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomNormal kernel{ctx};
  EXPECT_THROW(kernel({2}, 0.0, 1.0, RandomNormal::kNoSeed,
                      static_cast<int32_t>(core::runtime::DataType::INT32)),
               std::invalid_argument);
}

TEST(KernelClass, RandomUniformProducesFloatInRange) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomUniform kernel{ctx};
  Tensor y = kernel({16}, /*low=*/-2.0, /*high=*/3.0, /*seed=*/7);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  ASSERT_EQ(y.element_count(), 16);
  const float *py = y.AsFloat();
  for (int i = 0; i < 16; ++i) {
    EXPECT_GE(py[i], -2.0f);
    EXPECT_LT(py[i], 3.0f);
  }
}

TEST(KernelClass, RandomUniformIsDeterministicForSameSeed) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomUniform kernel{ctx};
  Tensor a = kernel({5}, 0.0, 1.0, /*seed=*/99);
  Tensor b = kernel({5}, 0.0, 1.0, /*seed=*/99);
  EXPECT_EQ(a.data, b.data);
}

TEST(KernelClass, RandomNormalLikeCopiesInputShape) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomNormalLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {2, 4}, std::vector<float>(8, 0.0f));
  Tensor y = kernel(x);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 4}));
}

TEST(KernelClass, RandomNormalLikeDtypeOverridesOutputType) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomNormalLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});
  Tensor y = kernel(x, 0.0, 1.0, RandomNormalLike::kNoSeed,
                    static_cast<int32_t>(core::runtime::DataType::DOUBLE));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::DOUBLE));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3}));
}

TEST(KernelClass, RandomUniformLikeCopiesInputShapeAndProducesInRange) {
  const KernelContext ctx{DefaultOpset(22)};
  RandomUniformLike kernel{ctx};
  const Tensor x = Tensor::FromFloat("", {6}, std::vector<float>(6, 0.0f));
  Tensor y = kernel(x, /*low=*/0.0, /*high=*/1.0, /*seed=*/11);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{6}));
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_GE(py[i], 0.0f);
    EXPECT_LT(py[i], 1.0f);
  }
}

TEST(KernelClass, RangeFloatPositiveDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromFloat("", {}, {1.0f});
  const Tensor limit = Tensor::FromFloat("", {}, {5.0f});
  const Tensor delta = Tensor::FromFloat("", {}, {2.0f});
  const Tensor y = kernel(start, limit, delta);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
}

TEST(KernelClass, RangeFloat16UsesExpectedBitPatterns) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start("", core::runtime::DataType::FLOAT16, {}, {0x00, 0x3C}); // 1.0f
  const Tensor limit("", core::runtime::DataType::FLOAT16, {}, {0x00, 0x45}); // 5.0f
  const Tensor delta("", core::runtime::DataType::FLOAT16, {}, {0x00, 0x40}); // 2.0f

  const Tensor y = kernel(start, limit, delta);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(ReadUint16Element(y, 0), 0x3C00);
  EXPECT_EQ(ReadUint16Element(y, 1), 0x4200);

  Tensor y_in_place("", core::runtime::DataType::FLOAT16, {2},
                    std::vector<uint8_t>(2 * sizeof(uint16_t)));
  kernel(start, limit, delta, y_in_place);
  EXPECT_EQ(ReadUint16Element(y_in_place, 0), 0x3C00);
  EXPECT_EQ(ReadUint16Element(y_in_place, 1), 0x4200);
}

TEST(KernelClass, RangeUsesAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromFloat("", {}, {1.0f});
  const Tensor limit = Tensor::FromFloat("", {}, {5.0f});
  const Tensor delta = Tensor::FromFloat("", {}, {2.0f});
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});
  const Tensor y = kernel(start, limit, delta, &rt);
  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
}

TEST(KernelClass, RangeInt32NegativeDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromInt32("", {}, {10});
  const Tensor limit = Tensor::FromInt32("", {}, {6});
  const Tensor delta = Tensor::FromInt32("", {}, {-3});
  const Tensor y = kernel(start, limit, delta);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::INT32));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 10);
  EXPECT_EQ(py[1], 7);
}

TEST(KernelClass, RangeEmptyWhenStartExceedsLimitForPositiveDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromInt64("", {}, {5});
  const Tensor limit = Tensor::FromInt64("", {}, {5});
  const Tensor delta = Tensor::FromInt64("", {}, {1});
  const Tensor y = kernel(start, limit, delta);
  EXPECT_EQ(y.shape, (std::vector<int64_t>{0}));
  EXPECT_EQ(y.element_count(), 0);
}

TEST(KernelClass, RangeRejectsMismatchedDtypes) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromFloat("", {}, {1.0f});
  const Tensor limit = Tensor::FromInt32("", {}, {5});
  const Tensor delta = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW(kernel(start, limit, delta), std::invalid_argument);
}

TEST(KernelClass, RangeRejectsZeroDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromInt64("", {}, {0});
  const Tensor limit = Tensor::FromInt64("", {}, {5});
  const Tensor delta = Tensor::FromInt64("", {}, {0});
  EXPECT_THROW(kernel(start, limit, delta), std::invalid_argument);
}

TEST(KernelClass, RangeInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromFloat("", {}, {1.0f});
  const Tensor limit = Tensor::FromFloat("", {}, {5.0f});
  const Tensor delta = Tensor::FromFloat("", {}, {2.0f});
  Tensor y("", core::runtime::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
  kernel(start, limit, delta, y);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
}

TEST(KernelClass, RangeInPlaceInt32NegativeDelta) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromInt32("", {}, {10});
  const Tensor limit = Tensor::FromInt32("", {}, {6});
  const Tensor delta = Tensor::FromInt32("", {}, {-3});
  Tensor y("", core::runtime::DataType::INT32, {2}, std::vector<uint8_t>(2 * sizeof(int32_t)));
  kernel(start, limit, delta, y);
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 10);
  EXPECT_EQ(py[1], 7);
}

TEST(KernelClass, RangeInPlaceEmptyOutput) {
  const KernelContext ctx{DefaultOpset(11)};
  Range kernel{ctx};
  const Tensor start = Tensor::FromInt64("", {}, {5});
  const Tensor limit = Tensor::FromInt64("", {}, {5});
  const Tensor delta = Tensor::FromInt64("", {}, {1});
  Tensor y("", core::runtime::DataType::INT64, {0}, std::vector<uint8_t>(0));
  kernel(start, limit, delta, y);
  EXPECT_EQ(y.element_count(), 0);
}

} // namespace Test
