// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../defs/generator/utils.h"
#include "onnx.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

// Builds a scalar TensorProto of the given data type from a single value.
template <typename T> TensorProto MakeScalar(T value);

template <> TensorProto MakeScalar<float>(float value) {
  TensorProto tp;
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.add_float_data(value);
  return tp;
}

template <> TensorProto MakeScalar<double>(double value) {
  TensorProto tp;
  tp.set_data_type(TensorProto::DataType::DOUBLE);
  tp.add_double_data(value);
  return tp;
}

template <> TensorProto MakeScalar<int32_t>(int32_t value) {
  TensorProto tp;
  tp.set_data_type(TensorProto::DataType::INT32);
  tp.add_int32_data(value);
  return tp;
}

template <> TensorProto MakeScalar<int64_t>(int64_t value) {
  TensorProto tp;
  tp.set_data_type(TensorProto::DataType::INT64);
  tp.add_int64_data(value);
  return tp;
}

} // namespace

// ===========================================================================
// compute_output_dim_for_range – int32
// ===========================================================================

TEST(compute_output_dim_for_range, Int32_PositiveDelta_TwoElements) {
  // Range(3, 9, 3) -> [3, 6] -> 2 elements
  auto start = MakeScalar<int32_t>(3);
  auto limit = MakeScalar<int32_t>(9);
  auto delta = MakeScalar<int32_t>(3);
  EXPECT_EQ(compute_output_dim_for_range<int32_t>(&start, &limit, &delta), int64_t{2});
}

TEST(compute_output_dim_for_range, Int32_NegativeDelta_ThreeElements) {
  // Range(10, 4, -2) -> [10, 8, 6] -> 3 elements
  auto start = MakeScalar<int32_t>(10);
  auto limit = MakeScalar<int32_t>(4);
  auto delta = MakeScalar<int32_t>(-2);
  EXPECT_EQ(compute_output_dim_for_range<int32_t>(&start, &limit, &delta), int64_t{3});
}

TEST(compute_output_dim_for_range, Int32_EmptyRange_LimitEqualsStart) {
  // Range(5, 5, 1) -> [] -> 0 elements
  auto start = MakeScalar<int32_t>(5);
  auto limit = MakeScalar<int32_t>(5);
  auto delta = MakeScalar<int32_t>(1);
  EXPECT_EQ(compute_output_dim_for_range<int32_t>(&start, &limit, &delta), int64_t{0});
}

TEST(compute_output_dim_for_range, Int32_EmptyRange_NegativeDeltaWithPositiveDirection) {
  // Range(1, 5, -1) -> [] -> 0 elements (delta goes wrong way)
  auto start = MakeScalar<int32_t>(1);
  auto limit = MakeScalar<int32_t>(5);
  auto delta = MakeScalar<int32_t>(-1);
  EXPECT_EQ(compute_output_dim_for_range<int32_t>(&start, &limit, &delta), int64_t{0});
}

TEST(compute_output_dim_for_range, Int32_SingleElement) {
  // Range(0, 1, 1) -> [0] -> 1 element
  auto start = MakeScalar<int32_t>(0);
  auto limit = MakeScalar<int32_t>(1);
  auto delta = MakeScalar<int32_t>(1);
  EXPECT_EQ(compute_output_dim_for_range<int32_t>(&start, &limit, &delta), int64_t{1});
}

// ===========================================================================
// compute_output_dim_for_range – int64
// ===========================================================================

TEST(compute_output_dim_for_range, Int64_PositiveDelta) {
  // Range(0, 10, 2) -> [0, 2, 4, 6, 8] -> 5 elements
  auto start = MakeScalar<int64_t>(0);
  auto limit = MakeScalar<int64_t>(10);
  auto delta = MakeScalar<int64_t>(2);
  EXPECT_EQ(compute_output_dim_for_range<int64_t>(&start, &limit, &delta), int64_t{5});
}

TEST(compute_output_dim_for_range, Int64_NegativeDelta) {
  // Range(5, 0, -1) -> [5, 4, 3, 2, 1] -> 5 elements
  auto start = MakeScalar<int64_t>(5);
  auto limit = MakeScalar<int64_t>(0);
  auto delta = MakeScalar<int64_t>(-1);
  EXPECT_EQ(compute_output_dim_for_range<int64_t>(&start, &limit, &delta), int64_t{5});
}

TEST(compute_output_dim_for_range, Int64_EmptyRange) {
  // Range(3, 3, 1) -> [] -> 0 elements
  auto start = MakeScalar<int64_t>(3);
  auto limit = MakeScalar<int64_t>(3);
  auto delta = MakeScalar<int64_t>(1);
  EXPECT_EQ(compute_output_dim_for_range<int64_t>(&start, &limit, &delta), int64_t{0});
}

// ===========================================================================
// compute_output_dim_for_range – float
// ===========================================================================

TEST(compute_output_dim_for_range, Float_PositiveDelta) {
  // Range(1.0, 5.0, 2.0) -> [1.0, 3.0] -> 2 elements
  auto start = MakeScalar<float>(1.0f);
  auto limit = MakeScalar<float>(5.0f);
  auto delta = MakeScalar<float>(2.0f);
  EXPECT_EQ(compute_output_dim_for_range<float>(&start, &limit, &delta), int64_t{2});
}

TEST(compute_output_dim_for_range, Float_NegativeDelta) {
  // Range(10.0, 4.0, -2.0) -> [10.0, 8.0, 6.0] -> 3 elements
  auto start = MakeScalar<float>(10.0f);
  auto limit = MakeScalar<float>(4.0f);
  auto delta = MakeScalar<float>(-2.0f);
  EXPECT_EQ(compute_output_dim_for_range<float>(&start, &limit, &delta), int64_t{3});
}

TEST(compute_output_dim_for_range, Float_NonIntegerStep) {
  // Range(0.0, 1.0, 0.25) -> [0.0, 0.25, 0.5, 0.75] -> 4 elements
  auto start = MakeScalar<float>(0.0f);
  auto limit = MakeScalar<float>(1.0f);
  auto delta = MakeScalar<float>(0.25f);
  EXPECT_EQ(compute_output_dim_for_range<float>(&start, &limit, &delta), int64_t{4});
}

TEST(compute_output_dim_for_range, Float_EmptyRange) {
  // Range(1.0, 1.0, 1.0) -> [] -> 0 elements
  auto start = MakeScalar<float>(1.0f);
  auto limit = MakeScalar<float>(1.0f);
  auto delta = MakeScalar<float>(1.0f);
  EXPECT_EQ(compute_output_dim_for_range<float>(&start, &limit, &delta), int64_t{0});
}

// ===========================================================================
// compute_output_dim_for_range – double
// ===========================================================================

TEST(compute_output_dim_for_range, Double_PositiveDelta) {
  // Range(3.0, 9.0, 3.0) -> [3.0, 6.0] -> 2 elements
  auto start = MakeScalar<double>(3.0);
  auto limit = MakeScalar<double>(9.0);
  auto delta = MakeScalar<double>(3.0);
  EXPECT_EQ(compute_output_dim_for_range<double>(&start, &limit, &delta), int64_t{2});
}

TEST(compute_output_dim_for_range, Double_NegativeDelta) {
  // Range(0.0, -5.0, -1.0) -> [0.0, -1.0, -2.0, -3.0, -4.0] -> 5 elements
  auto start = MakeScalar<double>(0.0);
  auto limit = MakeScalar<double>(-5.0);
  auto delta = MakeScalar<double>(-1.0);
  EXPECT_EQ(compute_output_dim_for_range<double>(&start, &limit, &delta), int64_t{5});
}

TEST(compute_output_dim_for_range, Double_EmptyRange) {
  // Range(2.0, 0.0, 1.0) -> [] -> 0 elements (delta goes wrong way)
  auto start = MakeScalar<double>(2.0);
  auto limit = MakeScalar<double>(0.0);
  auto delta = MakeScalar<double>(1.0);
  EXPECT_EQ(compute_output_dim_for_range<double>(&start, &limit, &delta), int64_t{0});
}

// ===========================================================================
// compute_output_dim_for_range – non-scalar input triggers inference error
// ===========================================================================

TEST(compute_output_dim_for_range, NonScalarInput_ThrowsInferenceError) {
  TensorProto start;
  start.set_data_type(TensorProto::DataType::INT32);
  start.add_dims(1); // non-scalar: shape [1]
  start.add_int32_data(0);

  auto limit = MakeScalar<int32_t>(5);
  auto delta = MakeScalar<int32_t>(1);

  EXPECT_THROW(compute_output_dim_for_range<int32_t>(&start, &limit, &delta),
               InferenceError);
}
