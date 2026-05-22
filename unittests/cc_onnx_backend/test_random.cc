// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend/random.h"

#include <gtest/gtest.h>

#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(BackendRandomTest, RandIsDeterministicWithSeed) {
  auto first = onnx_backend::Rand({2, 3}, /*seed=*/42);
  auto second = onnx_backend::Rand({2, 3}, /*seed=*/42);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first.size(), 6u);
}

TEST(BackendRandomTest, RandDefaultSeedMatchesPythonReference) {
  // Values produced by the Python reference for ``rand(3, seed=7)``.
  std::vector<double> expected = {
      0.3898297483912715,
      0.01678829452815611,
      0.9007606806068834,
  };
  auto values = onnx_backend::Rand({3}, /*seed=*/7);
  ASSERT_EQ(values.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(values[i], expected[i]);
  }
}

TEST(BackendRandomTest, RandnDefaultSeedMatchesPythonReference) {
  std::vector<double> expected = {
      -1.0009541026316917,
      1.3335403660869787,
      0.5188620948808316,
  };
  auto values = onnx_backend::Randn({3}, /*seed=*/7);
  ASSERT_EQ(values.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(values[i], expected[i]);
  }
}

TEST(BackendRandomTest, RandIntMatchesPythonReference) {
  std::vector<int64_t> expected = {2, 3, 0, 3, 5};
  auto values = onnx_backend::RandInt(0, 7, {5}, /*seed=*/7);
  EXPECT_EQ(values, expected);
}

TEST(BackendRandomTest, RandIntBoundsAreRespected) {
  auto values = onnx_backend::RandInt(0, 5, {100}, /*seed=*/12);
  ASSERT_EQ(values.size(), 100u);
  for (int64_t v : values) {
    EXPECT_GE(v, 0);
    EXPECT_LT(v, 5);
  }
}

TEST(BackendRandomTest, RandIntRejectsInvalidRange) {
  EXPECT_THROW(onnx_backend::RandInt(5, 5, {3}), std::invalid_argument);
  EXPECT_THROW(onnx_backend::RandInt(10, 2, {3}), std::invalid_argument);
}

TEST(BackendRandomTest, NegativeShapeRejected) {
  EXPECT_THROW(onnx_backend::Rand({-1, 2}), std::invalid_argument);
  EXPECT_THROW(onnx_backend::Randn({2, -3}), std::invalid_argument);
  EXPECT_THROW(onnx_backend::RandInt(0, 2, {-1}), std::invalid_argument);
}

TEST(BackendRandomTest, EmptyShapeReturnsSingleValue) {
  EXPECT_EQ(onnx_backend::Rand({}).size(), 1u);
  EXPECT_EQ(onnx_backend::Randn({}).size(), 1u);
}

} // namespace Test
