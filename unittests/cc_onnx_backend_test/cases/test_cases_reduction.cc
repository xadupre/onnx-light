// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectReductionTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectReductionTestCases(registry, op_type);
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

void CheckReduceSumCasePresent(const std::vector<TestCase> &cases, const std::string &name,
                               size_t expected_inputs, const std::vector<int64_t> &expected_shape) {
  const TestCase *tc = FindCase(cases, name);
  ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

  const GraphProto &graph = tc->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), "ReduceSum");

  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), expected_inputs);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
}

} // namespace

TEST(BackendTestCase, ReduceSumAllUpstreamCasesRegistered) {
  const auto cases = CollectTestCases("ReduceSum");

  // ``axes`` omitted, default ``keepdims = 1``: shape collapses to all-1s.
  CheckReduceSumCasePresent(cases, "test_cc_reducesum_default_axes_keepdims",
                            /*expected_inputs=*/1, /*expected_shape=*/{1, 1, 1});

  // Explicit ``axes = [1]`` with ``keepdims = 1``: preserves reduced dim as 1.
  CheckReduceSumCasePresent(cases, "test_cc_reducesum_keepdims",
                            /*expected_inputs=*/2, /*expected_shape=*/{3, 1, 2});

  // Explicit ``axes = [1]`` with ``keepdims = 0``: drops the reduced dim.
  CheckReduceSumCasePresent(cases, "test_cc_reducesum_do_not_keepdims",
                            /*expected_inputs=*/2, /*expected_shape=*/{3, 2});

  // Negative ``axes = [-2]`` with ``keepdims = 1``.
  CheckReduceSumCasePresent(cases, "test_cc_reducesum_negative_axes_keepdims",
                            /*expected_inputs=*/2, /*expected_shape=*/{3, 1, 2});

  // Empty ``axes`` input + ``noop_with_empty_axes = 1`` is an identity copy.
  CheckReduceSumCasePresent(cases, "test_cc_reducesum_empty_axes_input_noop",
                            /*expected_inputs=*/2, /*expected_shape=*/{3, 2, 2});

  // Reducing over a size-0 axis: ``[2, 0, 4]`` with axes=[1] -> ``[2, 1, 4]``.
  CheckReduceSumCasePresent(cases, "test_cc_reducesum_empty_set",
                            /*expected_inputs=*/2, /*expected_shape=*/{2, 1, 4});

  // Non-reduced axis of size 0: ``[2, 0, 4]`` with axes=[2] -> ``[2, 0, 1]``.
  CheckReduceSumCasePresent(cases, "test_cc_reducesum_empty_set_non_reduced_axis_zero",
                            /*expected_inputs=*/2, /*expected_shape=*/{2, 0, 1});
}

TEST(BackendTestCase, ReduceSumDefaultAxesKeepdimsSumIs78) {
  const auto cases = CollectTestCases("ReduceSum");
  const TestCase *tc = FindCase(cases, "test_cc_reducesum_default_axes_keepdims");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data.size(), sizeof(float));
  const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
  EXPECT_FLOAT_EQ(py[0], 78.0f);
}

TEST(BackendTestCase, ReduceSumEmptyAxesInputNoopIsIdentity) {
  const auto cases = CollectTestCases("ReduceSum");
  const TestCase *tc = FindCase(cases, "test_cc_reducesum_empty_axes_input_noop");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs[0].data, ds.outputs[0].data);
}

TEST(BackendTestCase, ReduceSumEmptySetIsZeroFilled) {
  const auto cases = CollectTestCases("ReduceSum");
  const TestCase *tc = FindCase(cases, "test_cc_reducesum_empty_set");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].data.size(), 2u * 1u * 4u * sizeof(float));
  const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(py[i], 0.0f) << "at index " << i;
  }
}

TEST(BackendTestCase, ReduceSumEmptySetNonReducedAxisZeroHasNoElements) {
  const auto cases = CollectTestCases("ReduceSum");
  const TestCase *tc = FindCase(cases, "test_cc_reducesum_empty_set_non_reduced_axis_zero");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  EXPECT_EQ(ds.outputs[0].data.size(), 0u);
}

TEST(BackendTestCase, ReduceMaxCasesRegistered) {
  const auto cases = CollectTestCases("ReduceMax");
  const TestCase *keepdims = FindCase(cases, "test_cc_reducemax_keepdims");
  ASSERT_NE(keepdims, nullptr);
  const GraphProto &graph = keepdims->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op = node.ref_op_type();
  EXPECT_EQ(std::string(op.data(), op.size()), "ReduceMax");

  ASSERT_EQ(keepdims->data_sets.size(), 1u);
  const auto &ds = keepdims->data_sets[0];
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 1, 2}));
  const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
  EXPECT_FLOAT_EQ(py[0], 3.0f);
  EXPECT_FLOAT_EQ(py[1], 4.0f);
  EXPECT_FLOAT_EQ(py[2], 7.0f);
  EXPECT_FLOAT_EQ(py[3], 8.0f);
  EXPECT_FLOAT_EQ(py[4], 11.0f);
  EXPECT_FLOAT_EQ(py[5], 12.0f);

  EXPECT_NE(FindCase(cases, "test_cc_reducemax_do_not_keepdims"), nullptr);
  EXPECT_NE(FindCase(cases, "test_cc_reducemax_negative_axes_keepdims"), nullptr);
  EXPECT_NE(FindCase(cases, "test_cc_reducemax_empty_axes_input_noop"), nullptr);

  // ``axes`` omitted: reduces to a single ``[1, 1, 1]`` tensor whose value is
  // the maximum of ``data`` (12 for the standard ``[3, 2, 2]`` input).
  const TestCase *default_axes = FindCase(cases, "test_cc_reducemax_default_axes_keepdims");
  ASSERT_NE(default_axes, nullptr);
  ASSERT_EQ(default_axes->data_sets.size(), 1u);
  const auto &da_ds = default_axes->data_sets[0];
  ASSERT_EQ(da_ds.inputs.size(), 1u);
  EXPECT_EQ(da_ds.outputs[0].shape, (std::vector<int64_t>{1, 1, 1}));
  const float *pda = reinterpret_cast<const float *>(da_ds.outputs[0].data.data());
  EXPECT_FLOAT_EQ(pda[0], 12.0f);

  // Reducing over an axis of size 0: result is ``-inf`` (ReduceMax identity).
  const TestCase *empty = FindCase(cases, "test_cc_reducemax_empty_set");
  ASSERT_NE(empty, nullptr);
  const auto &e_ds = empty->data_sets[0];
  EXPECT_EQ(e_ds.outputs[0].shape, (std::vector<int64_t>{2, 1, 4}));
  ASSERT_EQ(e_ds.outputs[0].data.size(), 2u * 1u * 4u * sizeof(float));
  const float *pe = reinterpret_cast<const float *>(e_ds.outputs[0].data.data());
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_TRUE(std::isinf(pe[i]) && pe[i] < 0.0f) << "at index " << i;
  }

  const TestCase *empty_nr = FindCase(cases, "test_cc_reducemax_empty_set_non_reduced_axis_zero");
  ASSERT_NE(empty_nr, nullptr);
  EXPECT_EQ(empty_nr->data_sets[0].outputs[0].data.size(), 0u);
}

TEST(BackendTestCase, ReduceMinCasesRegistered) {
  const auto cases = CollectTestCases("ReduceMin");
  const TestCase *keepdims = FindCase(cases, "test_cc_reducemin_keepdims");
  ASSERT_NE(keepdims, nullptr);
  const GraphProto &graph = keepdims->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op = node.ref_op_type();
  EXPECT_EQ(std::string(op.data(), op.size()), "ReduceMin");

  ASSERT_EQ(keepdims->data_sets.size(), 1u);
  const auto &ds = keepdims->data_sets[0];
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 1, 2}));
  const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
  EXPECT_FLOAT_EQ(py[2], 5.0f);
  EXPECT_FLOAT_EQ(py[3], 6.0f);
  EXPECT_FLOAT_EQ(py[4], 9.0f);
  EXPECT_FLOAT_EQ(py[5], 10.0f);

  const TestCase *noop = FindCase(cases, "test_cc_reducemin_empty_axes_input_noop");
  ASSERT_NE(noop, nullptr);
  ASSERT_EQ(noop->data_sets.size(), 1u);
  EXPECT_EQ(noop->data_sets[0].inputs[0].data, noop->data_sets[0].outputs[0].data);

  EXPECT_NE(FindCase(cases, "test_cc_reducemin_do_not_keepdims"), nullptr);
  EXPECT_NE(FindCase(cases, "test_cc_reducemin_negative_axes_keepdims"), nullptr);

  // ``axes`` omitted: reduces to a single ``[1, 1, 1]`` tensor whose value is
  // the minimum of ``data`` (1 for the standard ``[3, 2, 2]`` input).
  const TestCase *default_axes = FindCase(cases, "test_cc_reducemin_default_axes_keepdims");
  ASSERT_NE(default_axes, nullptr);
  const auto &da_ds = default_axes->data_sets[0];
  ASSERT_EQ(da_ds.inputs.size(), 1u);
  EXPECT_EQ(da_ds.outputs[0].shape, (std::vector<int64_t>{1, 1, 1}));
  const float *pda = reinterpret_cast<const float *>(da_ds.outputs[0].data.data());
  EXPECT_FLOAT_EQ(pda[0], 1.0f);

  // Reducing over an axis of size 0: result is ``+inf`` (ReduceMin identity).
  const TestCase *empty = FindCase(cases, "test_cc_reducemin_empty_set");
  ASSERT_NE(empty, nullptr);
  const auto &e_ds = empty->data_sets[0];
  EXPECT_EQ(e_ds.outputs[0].shape, (std::vector<int64_t>{2, 1, 4}));
  ASSERT_EQ(e_ds.outputs[0].data.size(), 2u * 1u * 4u * sizeof(float));
  const float *pe = reinterpret_cast<const float *>(e_ds.outputs[0].data.data());
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_TRUE(std::isinf(pe[i]) && pe[i] > 0.0f) << "at index " << i;
  }

  const TestCase *empty_nr = FindCase(cases, "test_cc_reducemin_empty_set_non_reduced_axis_zero");
  ASSERT_NE(empty_nr, nullptr);
  EXPECT_EQ(empty_nr->data_sets[0].outputs[0].data.size(), 0u);
}

namespace {

void CheckArgReduceCasePresent(const std::vector<TestCase> &cases, const std::string &name,
                               const std::string &op_type,
                               const std::vector<int64_t> &expected_shape,
                               const std::vector<int64_t> &expected_values) {
  const TestCase *tc = FindCase(cases, name);
  ASSERT_NE(tc, nullptr) << "missing backend test case: " << name;

  const GraphProto &graph = tc->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op = node.ref_op_type();
  EXPECT_EQ(std::string(op.data(), op.size()), op_type);

  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  ASSERT_EQ(ds.outputs[0].data.size(), expected_values.size() * sizeof(int64_t));
  const int64_t *py = reinterpret_cast<const int64_t *>(ds.outputs[0].data.data());
  for (size_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_EQ(py[i], expected_values[i]) << "at index " << i;
  }
}

} // namespace

TEST(BackendTestCase, ArgMaxAllUpstreamCasesRegistered) {
  const auto cases = CollectTestCases();
  // Shared input [[2, 2], [3, 10]]: argmax along axis=1 is [0, 1]; along
  // axis=0 (default) is [1, 1].
  CheckArgReduceCasePresent(cases, "test_cc_argmax_no_keepdims", "ArgMax", {2}, {0, 1});
  CheckArgReduceCasePresent(cases, "test_cc_argmax_keepdims", "ArgMax", {2, 1}, {0, 1});
  CheckArgReduceCasePresent(cases, "test_cc_argmax_default_axis", "ArgMax", {1, 2}, {1, 1});
  CheckArgReduceCasePresent(cases, "test_cc_argmax_negative_axis_keepdims", "ArgMax", {2, 1},
                            {0, 1});

  // select_last_index=1 with the tie at (row 0, col 0) and (row 0, col 1):
  // axis=1 yields [1, 1] (the last index of the tied value).
  CheckArgReduceCasePresent(cases, "test_cc_argmax_no_keepdims_select_last_index", "ArgMax", {2},
                            {1, 1});
  CheckArgReduceCasePresent(cases, "test_cc_argmax_keepdims_select_last_index", "ArgMax", {2, 1},
                            {1, 1});
  CheckArgReduceCasePresent(cases, "test_cc_argmax_default_axis_select_last_index", "ArgMax",
                            {1, 2}, {1, 1});
  CheckArgReduceCasePresent(cases, "test_cc_argmax_negative_axis_keepdims_select_last_index",
                            "ArgMax", {2, 1}, {1, 1});
}

TEST(BackendTestCase, ArgMinAllUpstreamCasesRegistered) {
  const auto cases = CollectTestCases();
  // Shared input [[2, 2], [3, 10]]: argmin along axis=1 is [0, 0] (first
  // occurrence); along axis=0 (default) is [0, 0].
  CheckArgReduceCasePresent(cases, "test_cc_argmin_no_keepdims", "ArgMin", {2}, {0, 0});
  CheckArgReduceCasePresent(cases, "test_cc_argmin_keepdims", "ArgMin", {2, 1}, {0, 0});
  CheckArgReduceCasePresent(cases, "test_cc_argmin_default_axis", "ArgMin", {1, 2}, {0, 0});
  CheckArgReduceCasePresent(cases, "test_cc_argmin_negative_axis_keepdims", "ArgMin", {2, 1},
                            {0, 0});

  // select_last_index=1: axis=1 yields [1, 0] — row 0 ties at 2 so the last
  // occurrence is at index 1; row 1 has a unique min at index 0.
  CheckArgReduceCasePresent(cases, "test_cc_argmin_no_keepdims_select_last_index", "ArgMin", {2},
                            {1, 0});
  CheckArgReduceCasePresent(cases, "test_cc_argmin_keepdims_select_last_index", "ArgMin", {2, 1},
                            {1, 0});
  // Default axis=0 with select_last_index: column 0 ties at 2 (row 0) vs 3
  // (row 1) -> unique min row 0; column 1 ties at 2 (row 0) vs 10 (row 1)
  // -> unique min row 0 -> [[0, 0]].
  CheckArgReduceCasePresent(cases, "test_cc_argmin_default_axis_select_last_index", "ArgMin",
                            {1, 2}, {0, 0});
  CheckArgReduceCasePresent(cases, "test_cc_argmin_negative_axis_keepdims_select_last_index",
                            "ArgMin", {2, 1}, {1, 0});
}

} // namespace Test
