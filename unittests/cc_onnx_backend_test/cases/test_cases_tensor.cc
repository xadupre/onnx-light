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

namespace {

struct Float8Expectation {
  const char *name;
  TensorProto::DataType dtype;
  std::vector<uint8_t> expected_bytes;
};

const std::vector<Float8Expectation> &Float8Expectations() {
  // Byte values produced by ``onnx.numpy_helper.saturate_cast`` for the
  // 15-element FP32 vector that the upstream ``test_cast_FLOAT_to_FLOAT8*``
  // node tests exercise.
  static const std::vector<Float8Expectation> kExpectations = {
      {"FLOAT8E4M3FN",
       TensorProto::DataType::FLOAT8E4M3FN,
       {0x2F, 0x2F, 0x30, 0x35, 0x2F, 0x34, 0x7E, 0x00, 0x7F, 0x7E, 0x7E, 0xFE, 0x80, 0x00, 0xFE}},
      {"FLOAT8E4M3FNUZ",
       TensorProto::DataType::FLOAT8E4M3FNUZ,
       {0x37, 0x37, 0x38, 0x3D, 0x37, 0x3C, 0x7F, 0x00, 0x80, 0x7F, 0x7F, 0xFF, 0x00, 0x00, 0xFF}},
      {"FLOAT8E5M2",
       TensorProto::DataType::FLOAT8E5M2,
       {0x38, 0x38, 0x38, 0x3B, 0x38, 0x3A, 0x7B, 0x00, 0x7E, 0x7B, 0x7B, 0xFB, 0x80, 0x00, 0xFB}},
      {"FLOAT8E5M2FNUZ",
       TensorProto::DataType::FLOAT8E5M2FNUZ,
       {0x3C, 0x3C, 0x3C, 0x3F, 0x3C, 0x3E, 0x7F, 0x00, 0x80, 0x7F, 0x7F, 0xFF, 0x00, 0x00, 0xFF}},
  };
  return kExpectations;
}

} // namespace

TEST(BackendTestCase, CastFloatToFloat8RegistersExpectedBytes) {
  const auto cases = CollectTestCases();
  for (const auto &f8 : Float8Expectations()) {
    const std::string name = std::string("test_cc_cast_FLOAT_to_") + f8.name;
    const TestCase *tc = FindCase(cases, name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(f8.dtype));
    const std::vector<int64_t> expected_shape = {3, 5};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    ASSERT_EQ(ds.outputs[0].data.size(), f8.expected_bytes.size()) << "for case " << name;
    for (size_t i = 0; i < f8.expected_bytes.size(); ++i) {
      EXPECT_EQ(ds.outputs[0].data[i], f8.expected_bytes[i]) << "byte " << i << " of " << name;
    }
  }
}

TEST(BackendTestCase, CastFloat8ToFloatInputMatchesSaturatedEncoding) {
  const auto cases = CollectTestCases();
  for (const auto &f8 : Float8Expectations()) {
    const std::string name = std::string("test_cc_cast_") + f8.name + "_to_FLOAT";
    const TestCase *tc = FindCase(cases, name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(f8.dtype));
    ASSERT_EQ(ds.inputs[0].data.size(), f8.expected_bytes.size()) << "for case " << name;
    for (size_t i = 0; i < f8.expected_bytes.size(); ++i) {
      EXPECT_EQ(ds.inputs[0].data[i], f8.expected_bytes[i]) << "byte " << i << " of " << name;
    }
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].data.size(), f8.expected_bytes.size() * sizeof(float));
  }
}

TEST(BackendTestCase, ConcatAllUpstreamCasesRegistered) {
  const auto cases = CollectTestCases();
  // Mirrors the shape/axis grid in ``onnx/backend/test/case/node/concat.py``:
  // for each input rank ``r`` in {1, 2, 3} we expect every positive axis in
  // ``[0, r-1]`` and every negative axis in ``[-r, -1]``.
  struct ShapeEntry {
    const char *label;
    int64_t rank;
  };
  const ShapeEntry kShapes[] = {{"1d", 1}, {"2d", 2}, {"3d", 3}};
  for (const auto &s : kShapes) {
    for (int64_t axis = 0; axis < s.rank; ++axis) {
      const std::string name =
          std::string("test_cc_concat_") + s.label + "_axis_" + std::to_string(axis);
      const TestCase *tc = FindCase(cases, name);
      ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;
      ASSERT_EQ(tc->data_sets.size(), 1u);
      EXPECT_EQ(tc->data_sets[0].inputs.size(), 2u);
      EXPECT_EQ(tc->data_sets[0].outputs.size(), 1u);
    }
    for (int64_t axis = 1; axis <= s.rank; ++axis) {
      const std::string name =
          std::string("test_cc_concat_") + s.label + "_axis_negative_" + std::to_string(axis);
      const TestCase *tc = FindCase(cases, name);
      ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;
      ASSERT_EQ(tc->data_sets.size(), 1u);
      EXPECT_EQ(tc->data_sets[0].inputs.size(), 2u);
      EXPECT_EQ(tc->data_sets[0].outputs.size(), 1u);
    }
  }
}

} // namespace Test
