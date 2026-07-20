// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/symbolic/sym_map.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(OnnxCoreSymMap, DefaultConstructedIsUnknown) {
  core::symbolic::SymMap m;
  EXPECT_FALSE(m.HasKeyType());
  EXPECT_FALSE(m.HasValueDtype());
  EXPECT_EQ(m.KeyType(), core::symbolic::TensorType::kUndefined);
  EXPECT_EQ(m.ValueDtype(), core::symbolic::TensorType::kUndefined);
  EXPECT_TRUE(m.ValueShape().Empty());
}

TEST(OnnxCoreSymMap, ConstructsWithKeyAndValueType) {
  core::symbolic::SymMap m(core::symbolic::TensorType::kInt64, core::symbolic::TensorType::kFloat);
  EXPECT_TRUE(m.HasKeyType());
  EXPECT_TRUE(m.HasValueDtype());
  EXPECT_EQ(m.KeyType(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(m.ValueDtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxCoreSymMap, ConstructsWithValueShape) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(3)};
  core::symbolic::SymMap m(core::symbolic::TensorType::kString, core::symbolic::TensorType::kFloat,
                           shape);
  EXPECT_EQ(m.ValueShape(), shape);
}

TEST(OnnxCoreSymMap, SettersUpdateFieldsAndFlags) {
  core::symbolic::SymMap m;
  m.SetKeyType(core::symbolic::TensorType::kInt64);
  m.SetValueDtype(core::symbolic::TensorType::kFloat);
  m.SetValueShape(core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_TRUE(m.HasKeyType());
  EXPECT_TRUE(m.HasValueDtype());
  EXPECT_EQ(m.ValueShape().Rank(), 1u);

  m.SetKeyType(core::symbolic::TensorType::kUndefined);
  EXPECT_FALSE(m.HasKeyType());
}

TEST(OnnxCoreSymMap, EqualityComparesAllFields) {
  core::symbolic::SymMap a(core::symbolic::TensorType::kInt64, core::symbolic::TensorType::kFloat);
  core::symbolic::SymMap b(core::symbolic::TensorType::kInt64, core::symbolic::TensorType::kFloat);
  core::symbolic::SymMap c(core::symbolic::TensorType::kString, core::symbolic::TensorType::kFloat);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

} // namespace Test
