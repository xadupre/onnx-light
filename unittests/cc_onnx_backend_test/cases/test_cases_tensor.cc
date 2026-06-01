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
using onnx_backend_test::CollectTensorTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectTensorTestCases(registry, op_type);
  return registry;
}
} // namespace
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
                          onnx_backend_test::DataType expected_output_dtype) {
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
  onnx_backend_test::DataType dtype;
  const char *name;
};

const std::vector<DtypeNameEntry> &SupportedDtypeNames() {
  static const std::vector<DtypeNameEntry> kEntries = {
      {onnx_backend_test::DataType::FLOAT, "FLOAT"},
      {onnx_backend_test::DataType::DOUBLE, "DOUBLE"},
      {onnx_backend_test::DataType::INT32, "INT32"},
      {onnx_backend_test::DataType::INT64, "INT64"},
      {onnx_backend_test::DataType::INT8, "INT8"},
      {onnx_backend_test::DataType::UINT8, "UINT8"},
      {onnx_backend_test::DataType::INT16, "INT16"},
      {onnx_backend_test::DataType::UINT16, "UINT16"},
      {onnx_backend_test::DataType::BOOL, "BOOL"},
      {onnx_backend_test::DataType::STRING, "STRING"},
  };
  return kEntries;
}

} // namespace

TEST(BackendTestCase, CastAllSupportedDataTypePairsRegistered) {
  const auto cases = CollectTestCases("Cast");
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
  const auto cases = CollectTestCases("Cast");
  const TestCase *tc = FindCase(cases, "test_cc_cast_FLOAT_to_INT32");
  ASSERT_NE(tc, nullptr);

  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  const std::vector<int64_t> expected_shape = {4};
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);

  const int32_t *py = reinterpret_cast<const int32_t *>(ds.outputs[0].data.data());
  EXPECT_EQ(py[0], -1);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 2);
  EXPECT_EQ(py[3], 4);
}

TEST(BackendTestCase, CastBoolToInt32MapsTrueToOne) {
  const auto cases = CollectTestCases("Cast");
  const TestCase *tc = FindCase(cases, "test_cc_cast_BOOL_to_INT32");
  ASSERT_NE(tc, nullptr);

  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  const int32_t *py = reinterpret_cast<const int32_t *>(ds.outputs[0].data.data());
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 1);
  EXPECT_EQ(py[3], 0);
}

TEST(BackendTestCase, CastInt32ToStringFormatsDecimal) {
  const auto cases = CollectTestCases("Cast");
  const TestCase *tc = FindCase(cases, "test_cc_cast_INT32_to_STRING");
  ASSERT_NE(tc, nullptr);

  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::STRING));
  ASSERT_EQ(ds.outputs[0].string_data.size(), 4u);
  EXPECT_EQ(ds.outputs[0].string_data[0], "-3");
  EXPECT_EQ(ds.outputs[0].string_data[1], "0");
  EXPECT_EQ(ds.outputs[0].string_data[2], "7");
  EXPECT_EQ(ds.outputs[0].string_data[3], "42");
}

TEST(BackendTestCase, CastStringToInt32ParsesDecimal) {
  const auto cases = CollectTestCases("Cast");
  const TestCase *tc = FindCase(cases, "test_cc_cast_STRING_to_INT32");
  ASSERT_NE(tc, nullptr);

  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  const int32_t *py = reinterpret_cast<const int32_t *>(ds.outputs[0].data.data());
  EXPECT_EQ(py[0], -3);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 7);
  EXPECT_EQ(py[3], 42);
}

// ---------------------------------------------------------------------------
// CastLike — backend test case registration tests.
//
// CastLike test cases mirror the Cast cases over the same source/destination
// dtype matrix. Each registered case must wrap a single ``CastLike`` node
// with two inputs (``input`` and ``target_type``) and exactly one output.
// ---------------------------------------------------------------------------

namespace {

void CheckCastLikeCasePresent(const std::vector<TestCase> &cases, const std::string &name,
                              onnx_backend_test::DataType expected_output_dtype) {
  const TestCase *tc = FindCase(cases, name);
  ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

  const GraphProto &graph = tc->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), "CastLike");
  EXPECT_EQ(graph.ref_input().size(), 2u);
  ASSERT_EQ(graph.ref_output().size(), 1u);

  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(expected_output_dtype));
  EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(expected_output_dtype));
  EXPECT_EQ(ds.inputs[0].shape, ds.outputs[0].shape);
}

} // namespace

TEST(BackendTestCase, CastLikeAllSupportedDataTypePairsRegistered) {
  const auto cases = CollectTestCases();
  const auto &dtypes = SupportedDtypeNames();
  for (const auto &from : dtypes) {
    for (const auto &to : dtypes) {
      if (from.dtype == to.dtype)
        continue;
      const std::string name = std::string("test_cc_castlike_") + from.name + "_to_" + to.name;
      CheckCastLikeCasePresent(cases, name, to.dtype);
    }
  }
}

TEST(BackendTestCase, CastLikeFloatToInt32MatchesCast) {
  const auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_cc_castlike_FLOAT_to_INT32");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  const std::vector<int64_t> expected_shape = {4};
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const int32_t *py = reinterpret_cast<const int32_t *>(ds.outputs[0].data.data());
  // FLOAT->INT32 truncates toward zero: {-1.5, 0.0, 2.75, 4.0} -> {-1, 0, 2, 4}.
  EXPECT_EQ(py[0], -1);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 2);
  EXPECT_EQ(py[3], 4);
}

namespace {

struct Float8Expectation {
  const char *name;
  onnx_backend_test::DataType dtype;
  std::vector<uint8_t> expected_bytes;
};

const std::vector<Float8Expectation> &Float8Expectations() {
  // Byte values produced by ``onnx.numpy_helper.saturate_cast`` for the
  // 15-element FP32 vector that the upstream ``test_cast_FLOAT_to_FLOAT8*``
  // node tests exercise.
  static const std::vector<Float8Expectation> kExpectations = {
      {"FLOAT8E4M3FN",
       onnx_backend_test::DataType::FLOAT8E4M3FN,
       {0x2F, 0x2F, 0x30, 0x35, 0x2F, 0x34, 0x7E, 0x00, 0x7F, 0x7E, 0x7E, 0xFE, 0x80, 0x00, 0xFE}},
      {"FLOAT8E4M3FNUZ",
       onnx_backend_test::DataType::FLOAT8E4M3FNUZ,
       {0x37, 0x37, 0x38, 0x3D, 0x37, 0x3C, 0x7F, 0x00, 0x80, 0x7F, 0x7F, 0xFF, 0x00, 0x00, 0xFF}},
      {"FLOAT8E5M2",
       onnx_backend_test::DataType::FLOAT8E5M2,
       {0x38, 0x38, 0x38, 0x3B, 0x38, 0x3A, 0x7B, 0x00, 0x7E, 0x7B, 0x7B, 0xFB, 0x80, 0x00, 0xFB}},
      {"FLOAT8E5M2FNUZ",
       onnx_backend_test::DataType::FLOAT8E5M2FNUZ,
       {0x3C, 0x3C, 0x3C, 0x3F, 0x3C, 0x3E, 0x7F, 0x00, 0x80, 0x7F, 0x7F, 0xFF, 0x00, 0x00, 0xFF}},
  };
  return kExpectations;
}

} // namespace

TEST(BackendTestCase, CastFloatToFloat8RegistersExpectedBytes) {
  const auto cases = CollectTestCases("Cast");
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
  const auto cases = CollectTestCases("Cast");
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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].data.size(), f8.expected_bytes.size() * sizeof(float));
  }
}

// Verifies that the four upstream ``test_affine_grid_*`` cases imported
// from ``onnx/backend/test/case/node/affinegrid.py`` are registered with
// the right op_type, input/output cardinality and shapes.
TEST(BackendTestCase, AffineGridUpstreamCasesArePresent) {
  const auto cases = CollectTestCases("AffineGrid");

  struct Expected {
    const char *name;
    std::vector<int64_t> theta_shape;
    std::vector<int64_t> size_shape;
    std::vector<int64_t> grid_shape;
  };
  const std::vector<Expected> expected{
      {"test_affine_grid_2d", {2, 2, 3}, {4}, {2, 5, 6, 2}},
      {"test_affine_grid_2d_align_corners", {2, 2, 3}, {4}, {2, 5, 6, 2}},
      {"test_affine_grid_3d", {2, 3, 4}, {5}, {2, 4, 5, 6, 3}},
      {"test_affine_grid_3d_align_corners", {2, 3, 4}, {5}, {2, 4, 5, 6, 3}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "AffineGrid");
    EXPECT_EQ(graph.ref_input().size(), 2u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.theta_shape);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.inputs[1].shape, exp.size_shape);
    EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].shape, exp.grid_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  }
}

// Verifies that the upstream ``test_gridsample_*`` cases imported from
// ``onnx/backend/test/case/node/gridsample.py`` are registered with the
// right op_type, input/output cardinality and shapes.
TEST(BackendTestCase, GridSampleUpstreamCasesArePresent) {
  const auto cases = CollectTestCases("GridSample");

  struct Expected {
    const char *name;
    std::vector<int64_t> x_shape;
    std::vector<int64_t> grid_shape;
    std::vector<int64_t> y_shape;
  };
  const std::vector<Expected> expected{
      {"test_gridsample", {1, 1, 4, 4}, {1, 6, 6, 2}, {1, 1, 6, 6}},
      {"test_gridsample_zeros_padding", {1, 1, 3, 2}, {1, 2, 4, 2}, {1, 1, 2, 4}},
      {"test_gridsample_border_padding", {1, 1, 3, 2}, {1, 2, 4, 2}, {1, 1, 2, 4}},
      {"test_gridsample_reflection_padding", {1, 1, 3, 2}, {1, 2, 4, 2}, {1, 1, 2, 4}},
      {"test_gridsample_bilinear", {1, 1, 3, 2}, {1, 2, 4, 2}, {1, 1, 2, 4}},
      {"test_gridsample_aligncorners_true", {1, 1, 3, 2}, {1, 2, 4, 2}, {1, 1, 2, 4}},
      {"test_gridsample_nearest", {1, 1, 3, 2}, {1, 2, 4, 2}, {1, 1, 2, 4}},
      {"test_gridsample_bicubic", {1, 1, 3, 2}, {1, 2, 4, 2}, {1, 1, 2, 4}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "GridSample");
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.x_shape);
    EXPECT_EQ(ds.inputs[1].shape, exp.grid_shape);
    EXPECT_EQ(ds.outputs[0].shape, exp.y_shape);
  }
}

namespace {

struct SubByteExpectation {
  const char *name;
  onnx_backend_test::DataType dtype;
  std::vector<int64_t> shape;
  // Packed wire bytes the upstream ONNX ``test_cast_FLOAT_to_<NAME>`` node
  // test exercises: saturating cast of the input ``np.arange`` vector into
  // the destination sub-byte dtype, packed two-per-byte for 4-bit dtypes
  // and four-per-byte for 2-bit dtypes (low elements first).
  std::vector<uint8_t> expected_bytes;
  // Unpacked, sign-extended values of ``expected_bytes`` — the output of
  // ``test_cc_cast_<NAME>_to_<wide-int>`` (``INT8`` for the signed
  // variants, ``UINT8`` for the unsigned ones). Stored as int32 so the
  // expectation vector is dtype-agnostic.
  std::vector<int32_t> unpacked_values;
  onnx_backend_test::DataType wide_int_dtype;
  const char *wide_int_name;
};

const std::vector<SubByteExpectation> &SubByteExpectations() {
  // INT4 — saturating cast of ``np.arange(-9, 16)``: values <= -8 clip to
  // -8 (0x8 nibble), values >= 7 clip to 7 (0x7 nibble). Negative values
  // use two's-complement nibbles 0x8..0xF.
  static const std::vector<SubByteExpectation> kEntries = {
      // UINT4: input values -9..15 → saturate to [0, 15] → 0,0,...,0,1,2,...,15
      // (10 zeros then 1..15 = 25 values). Packed nibbles: low first.
      {"UINT4",
       onnx_backend_test::DataType::UINT4,
       {5, 5},
       {0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       onnx_backend_test::DataType::UINT8,
       "UINT8"},
      // INT4: input values -9..15 → saturate to [-8, 7] → -8,-8,-7,...,7,7,...,7
      // ("-8" repeated twice for -9/-8, then -7..6, then 7 repeated 9 times).
      {"INT4",
       onnx_backend_test::DataType::INT4,
       {5, 5},
       {0x88, 0xA9, 0xCB, 0xED, 0x0F, 0x21, 0x43, 0x65, 0x77, 0x77, 0x77, 0x77, 0x07},
       {-8, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7},
       onnx_backend_test::DataType::INT8,
       "INT8"},
      // UINT2: input values -3..3 → saturate to [0, 3] → 0,0,0,0,1,2,3.
      // Packed 4 elements per byte (low pair first):
      //   byte0 = 0|0<<2|0<<4|0<<6 = 0x00
      //   byte1 = 1|2<<2|3<<4|0<<6 = 0x39
      {"UINT2",
       onnx_backend_test::DataType::UINT2,
       {7, 1},
       {0x00, 0x39},
       {0, 0, 0, 0, 1, 2, 3},
       onnx_backend_test::DataType::UINT8,
       "UINT8"},
      // INT2: input values -3..3 → saturate to [-2, 1] → -2,-2,-1,0,1,1,1.
      // Two's-complement 2-bit values: -2=0x2, -1=0x3, 0=0x0, 1=0x1.
      //   byte0 = 2|2<<2|3<<4|0<<6 = 0x3A
      //   byte1 = 1|1<<2|1<<4|0<<6 = 0x15
      {"INT2",
       onnx_backend_test::DataType::INT2,
       {7, 1},
       {0x3A, 0x15},
       {-2, -2, -1, 0, 1, 1, 1},
       onnx_backend_test::DataType::INT8,
       "INT8"},
  };
  return kEntries;
}

} // namespace

TEST(BackendTestCase, CastFloatToSubByteRegistersExpectedPackedBytes) {
  const auto cases = CollectTestCases("Cast");
  for (const auto &e : SubByteExpectations()) {
    const std::string name = std::string("test_cc_cast_FLOAT_to_") + e.name;
    const TestCase *tc = FindCase(cases, name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(e.dtype));
    EXPECT_EQ(ds.outputs[0].shape, e.shape);
    ASSERT_EQ(ds.outputs[0].data.size(), e.expected_bytes.size()) << "for case " << name;
    for (size_t i = 0; i < e.expected_bytes.size(); ++i) {
      EXPECT_EQ(ds.outputs[0].data[i], e.expected_bytes[i]) << "byte " << i << " of " << name;
    }
  }
}

TEST(BackendTestCase, CastSubByteToFloatInputMatchesSaturatedPackedEncoding) {
  const auto cases = CollectTestCases("Cast");
  for (const auto &e : SubByteExpectations()) {
    const std::string name = std::string("test_cc_cast_") + e.name + "_to_FLOAT";
    const TestCase *tc = FindCase(cases, name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(e.dtype));
    EXPECT_EQ(ds.inputs[0].shape, e.shape);
    ASSERT_EQ(ds.inputs[0].data.size(), e.expected_bytes.size()) << "for case " << name;
    for (size_t i = 0; i < e.expected_bytes.size(); ++i) {
      EXPECT_EQ(ds.inputs[0].data[i], e.expected_bytes[i]) << "byte " << i << " of " << name;
    }
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    ASSERT_EQ(ds.outputs[0].data.size(), e.unpacked_values.size() * sizeof(float));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    for (size_t i = 0; i < e.unpacked_values.size(); ++i) {
      EXPECT_EQ(py[i], static_cast<float>(e.unpacked_values[i]))
          << "element " << i << " of " << name;
    }
  }
}

TEST(BackendTestCase, CastSubByteToWideIntegerProducesUnpackedSaturatedValues) {
  const auto cases = CollectTestCases("Cast");
  for (const auto &e : SubByteExpectations()) {
    const std::string name = std::string("test_cc_cast_") + e.name + "_to_" + e.wide_int_name;
    const TestCase *tc = FindCase(cases, name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(e.wide_int_dtype));
    EXPECT_EQ(ds.outputs[0].shape, e.shape);
    ASSERT_EQ(ds.outputs[0].data.size(), e.unpacked_values.size());
    if (e.wide_int_dtype == onnx_backend_test::DataType::INT8) {
      const int8_t *py = reinterpret_cast<const int8_t *>(ds.outputs[0].data.data());
      for (size_t i = 0; i < e.unpacked_values.size(); ++i) {
        EXPECT_EQ(static_cast<int32_t>(py[i]), e.unpacked_values[i])
            << "element " << i << " of " << name;
      }
    } else {
      const uint8_t *py = ds.outputs[0].data.data();
      for (size_t i = 0; i < e.unpacked_values.size(); ++i) {
        EXPECT_EQ(static_cast<int32_t>(py[i]), e.unpacked_values[i])
            << "element " << i << " of " << name;
      }
    }
  }
}

TEST(BackendTestCase, ConcatAllUpstreamCasesRegistered) {
  const auto cases = CollectTestCases("Concat");
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

TEST(BackendTestCase, ExpandDimChangedAndDimUnchangedCasesRegistered) {
  const auto cases = CollectTestCases("Expand");

  struct Expected {
    const char *name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
  };
  const std::vector<Expected> expected{
      {"test_cc_expand_dim_changed", {3, 1}, {2, 3, 6}},
      {"test_cc_expand_dim_unchanged", {3, 1}, {3, 4}},
      {"test_cc_expand_1d_to_2d", {4}, {3, 4}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Expand");
    EXPECT_EQ(graph.ref_input().size(), 2u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.input_shape);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].shape, exp.output_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  }
}

TEST(BackendTestCase, ReshapeCasesRegistered) {
  const auto cases = CollectTestCases("Reshape");

  struct Expected {
    const char *name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
  };
  const std::vector<Expected> expected{
      {"test_cc_reshape_reordered", {2, 3}, {3, 2}},
      {"test_cc_reshape_allowzero_literal_zero", {0, 2}, {0, 2}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Reshape");

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.input_shape);
    EXPECT_EQ(ds.outputs[0].shape, exp.output_shape);
    EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  }
}

TEST(BackendTestCase, SliceCasesRegistered) {
  const auto cases = CollectTestCases("Slice");

  struct Expected {
    const char *name;
    std::vector<int64_t> output_shape;
    size_t input_count;
  };
  const std::vector<Expected> expected{
      {"test_cc_slice_axes_steps", {1, 2}, 5u},
      {"test_cc_slice_default_axes_steps", {1, 3}, 3u},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Slice");

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), exp.input_count);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, exp.output_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  }
}

TEST(BackendTestCase, DepthToSpaceCasesRegistered) {
  const auto cases = CollectTestCases("DepthToSpace");

  struct Expected {
    const char *name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
  };
  const std::vector<Expected> expected{
      {"test_cc_depthtospace_dcr", {1, 8, 2, 3}, {1, 2, 4, 6}},
      {"test_cc_depthtospace_crd", {1, 8, 2, 3}, {1, 2, 4, 6}},
      {"test_cc_depthtospace_default_mode", {1, 4, 2, 2}, {1, 1, 4, 4}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "DepthToSpace");
    EXPECT_EQ(graph.ref_input().size(), 1u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.input_shape);
    EXPECT_EQ(ds.outputs[0].shape, exp.output_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  }
}

TEST(BackendTestCase, TransposeDefaultAndPermCasesRegistered) {
  const auto cases = CollectTestCases("Transpose");

  struct Expected {
    const char *name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
  };
  const std::vector<Expected> expected{
      {"test_cc_transpose_default_perm", {2, 3}, {3, 2}},
      {"test_cc_transpose_permuted_axes", {2, 3, 4}, {3, 2, 4}},
      {"test_cc_transpose_permuted_axes_2", {1, 2, 3}, {2, 3, 1}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Transpose");
    EXPECT_EQ(graph.ref_input().size(), 1u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.input_shape);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, exp.output_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  }
}

TEST(BackendTestCase, TilePrecomputedAndOtherCasesRegistered) {
  const auto cases = CollectTestCases("Tile");

  struct Expected {
    const char *name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
  };
  const std::vector<Expected> expected{
      {"test_cc_tile_precomputed", {2, 2}, {4, 4}},
      {"test_cc_tile_1d", {3}, {9}},
      {"test_cc_tile_repeats_one", {2, 2}, {2, 2}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Tile");
    EXPECT_EQ(graph.ref_input().size(), 2u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.input_shape);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
    EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].shape, exp.output_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  }

  // Verify that the precomputed case produces the documented tiled values.
  const TestCase *tc = FindCase(cases, "test_cc_tile_precomputed");
  ASSERT_NE(tc, nullptr);
  const auto &out = tc->data_sets[0].outputs[0];
  ASSERT_EQ(out.data.size(), 16u * sizeof(float));
  const float *vals = reinterpret_cast<const float *>(out.data.data());
  const std::vector<float> expected_vals = {0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3};
  for (std::size_t i = 0; i < expected_vals.size(); ++i) {
    EXPECT_FLOAT_EQ(vals[i], expected_vals[i]) << "at index " << i;
  }
}

TEST(BackendTestCase, SqueezeCasesRegistered) {
  const auto cases = CollectTestCases("Squeeze");

  struct Expected {
    const char *name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
  };
  const std::vector<Expected> expected{
      {"test_cc_squeeze_axes", {2, 1, 3, 1}, {2, 3}},
      {"test_cc_squeeze_all_singleton", {1, 2, 1, 3}, {2, 3}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Squeeze");
    EXPECT_EQ(graph.ref_input().size(), 2u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.input_shape);
    EXPECT_EQ(ds.outputs[0].shape, exp.output_shape);
    EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  }
}

TEST(BackendTestCase, UnsqueezeCasesRegistered) {
  const auto cases = CollectTestCases("Unsqueeze");
  const TestCase *tc = FindCase(cases, "test_cc_unsqueeze_axes");
  ASSERT_NE(tc, nullptr);

  const GraphProto &graph = tc->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Unsqueeze");
  EXPECT_EQ(graph.ref_input().size(), 2u);
  ASSERT_EQ(graph.ref_output().size(), 1u);

  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 2, 1, 3}));
  EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
}

TEST(BackendTestCase, NonZeroCasesRegistered) {
  const auto cases = CollectTestCases("NonZero");

  struct Expected {
    const char *name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
  };
  const std::vector<Expected> expected{
      {"test_cc_nonzero_2d", {2, 2}, {2, 3}},
      {"test_cc_nonzero_1d", {5}, {1, 3}},
      {"test_cc_nonzero_bool", {2, 3}, {2, 3}},
      {"test_cc_nonzero_int64", {2, 3}, {2, 3}},
  };

  for (const Expected &exp : expected) {
    const TestCase *tc = FindCase(cases, exp.name);
    ASSERT_NE(tc, nullptr) << "missing backend test case: " << exp.name;

    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "NonZero");

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].shape, exp.input_shape);
    EXPECT_EQ(ds.outputs[0].shape, exp.output_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  }
}

} // namespace Test
