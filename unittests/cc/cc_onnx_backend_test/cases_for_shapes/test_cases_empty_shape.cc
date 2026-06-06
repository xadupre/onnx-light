// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectEmptyShapeTestCases;
using onnx_backend_test::TestCase;

namespace {

std::vector<TestCase> Collect(const std::string &op_type = "") {
  std::vector<TestCase> registry;
  CollectEmptyShapeTestCases(registry, op_type);
  return registry;
}

const TestCase *Find(const std::vector<TestCase> &cases, const std::string &name) {
  for (const auto &c : cases) {
    if (c.name == name) {
      return &c;
    }
  }
  return nullptr;
}

} // namespace

namespace Test {

TEST(EmptyShapeCases, CollectorReturnsCases) {
  const auto cases = Collect();
  EXPECT_FALSE(cases.empty());
}

TEST(EmptyShapeCases, FilterByOpTypeAdd) {
  const auto cases = Collect("Add");
  ASSERT_FALSE(cases.empty());
  for (const auto &tc : cases) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "Add");
  }
}

TEST(EmptyShapeCases, FilterByOpTypeCompress) {
  const auto cases = Collect("Compress");
  ASSERT_FALSE(cases.empty());
  for (const auto &tc : cases) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "Compress");
  }
}

TEST(EmptyShapeCases, AddScalarsProducesScalarSum) {
  const auto cases = Collect("Add");
  const TestCase *tc = Find(cases, "test_cc_add_empty_shape_scalars");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_TRUE(ds.outputs[0].shape.empty());
  ASSERT_EQ(ds.outputs[0].element_count(), 1);
  EXPECT_FLOAT_EQ(ds.outputs[0].AsFloat()[0],
                  ds.inputs[0].AsFloat()[0] + ds.inputs[1].AsFloat()[0]);
}

TEST(EmptyShapeCases, AddZeroDimProducesZeroElementOutput) {
  const auto cases = Collect("Add");
  const TestCase *tc = Find(cases, "test_cc_add_empty_shape_zero_dim");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].shape, std::vector<int64_t>({0}));
  EXPECT_EQ(ds.outputs[0].element_count(), 0);
}

TEST(EmptyShapeCases, AddZeroDim2DProducesZeroElementOutput) {
  const auto cases = Collect("Add");
  const TestCase *tc = Find(cases, "test_cc_add_empty_shape_zero_dim_2d");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{0, 3}));
  EXPECT_EQ(ds.outputs[0].element_count(), 0);
}

TEST(EmptyShapeCases, CompressAllFalseNoAxisProducesEmptyOutput) {
  const auto cases = Collect("Compress");
  const TestCase *tc = Find(cases, "test_cc_compress_empty_shape_no_axis_all_false");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].shape, std::vector<int64_t>({0}));
  EXPECT_EQ(ds.outputs[0].element_count(), 0);
}

TEST(EmptyShapeCases, CompressAllFalseAxis0ProducesEmptyRowOutput) {
  const auto cases = Collect("Compress");
  const TestCase *tc = Find(cases, "test_cc_compress_empty_shape_axis0_all_false");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(ds.outputs[0].element_count(), 0);
}

TEST(EmptyShapeCases, CompressInputZeroDimProducesEmptyOutput) {
  const auto cases = Collect("Compress");
  const TestCase *tc = Find(cases, "test_cc_compress_empty_shape_input_zero_dim");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(ds.outputs[0].element_count(), 0);
}

} // namespace Test
