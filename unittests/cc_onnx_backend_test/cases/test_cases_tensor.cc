// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::TestCase;

namespace Test {

namespace {

const TestCase *FindCase(const std::vector<TestCase> &cases, const std::string &name) {
  for (const auto &c : cases) {
    if (c.name == name) {
      return &c;
    }
  }
  return nullptr;
}

void CheckCastCasePresent(const std::vector<TestCase> &cases, const std::string &name,
                          TensorProto::DataType expected_output_dtype) {
  const TestCase *tc = FindCase(cases, name);
  ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

  const GraphProto &graph = tc->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Cast");
  EXPECT_EQ(graph.ref_input().size(), 1u);
  ASSERT_EQ(graph.ref_output().size(), 1u);

  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(expected_output_dtype));
  EXPECT_EQ(ds.inputs[0].shape, ds.outputs[0].shape);
}

struct DtypeNameEntry {
  TensorProto::DataType dtype;
  const char *name;
};

const std::vector<DtypeNameEntry> &SupportedDtypeNames() {
  static const std::vector<DtypeNameEntry> kEntries = {
      {TensorProto::DataType::FLOAT, "FLOAT"}, {TensorProto::DataType::DOUBLE, "DOUBLE"},
      {TensorProto::DataType::INT32, "INT32"}, {TensorProto::DataType::INT64, "INT64"},
      {TensorProto::DataType::INT8, "INT8"},   {TensorProto::DataType::UINT8, "UINT8"},
      {TensorProto::DataType::INT16, "INT16"}, {TensorProto::DataType::UINT16, "UINT16"},
      {TensorProto::DataType::BOOL, "BOOL"},   {TensorProto::DataType::STRING, "STRING"},
  };
  return kEntries;
}

} // namespace

TEST(BackendTestCase, CastAllSupportedDataTypePairsRegistered) {
  const auto cases = CollectTestCases();
  const auto &dtypes = SupportedDtypeNames();
  for (const auto &from : dtypes) {
    for (const auto &to : dtypes) {
      if (from.dtype == to.dtype)
        continue;
      const std::string name = std::string("test_cc_cast_") + from.name + "_to_" + to.name;
      CheckCastCasePresent(cases, name, to.dtype);
    }
  }
}

TEST(BackendTestCase, CastFloatToInt32TruncatesTowardZero) {
  const auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_cc_cast_FLOAT_to_INT32");
  ASSERT_NE(tc, nullptr);

  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
  const std::vector<int64_t> expected_shape = {4};
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);

  const int32_t *py = reinterpret_cast<const int32_t *>(ds.outputs[0].data.data());
  EXPECT_EQ(py[0], -1);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 2);
  EXPECT_EQ(py[3], 4);
}

TEST(BackendTestCase, CastBoolToInt32MapsTrueToOne) {
  const auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_cc_cast_BOOL_to_INT32");
  ASSERT_NE(tc, nullptr);

  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
  const int32_t *py = reinterpret_cast<const int32_t *>(ds.outputs[0].data.data());
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 1);
  EXPECT_EQ(py[3], 0);
}

TEST(BackendTestCase, CastInt32ToStringFormatsDecimal) {
  const auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_cc_cast_INT32_to_STRING");
  ASSERT_NE(tc, nullptr);

  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::STRING));
  ASSERT_EQ(ds.outputs[0].string_data.size(), 4u);
  EXPECT_EQ(ds.outputs[0].string_data[0], "-3");
  EXPECT_EQ(ds.outputs[0].string_data[1], "0");
  EXPECT_EQ(ds.outputs[0].string_data[2], "7");
  EXPECT_EQ(ds.outputs[0].string_data[3], "42");
}

TEST(BackendTestCase, CastStringToInt32ParsesDecimal) {
  const auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_cc_cast_STRING_to_INT32");
  ASSERT_NE(tc, nullptr);

  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
  const int32_t *py = reinterpret_cast<const int32_t *>(ds.outputs[0].data.data());
  EXPECT_EQ(py[0], -3);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 7);
  EXPECT_EQ(py[3], 42);
}

} // namespace Test
