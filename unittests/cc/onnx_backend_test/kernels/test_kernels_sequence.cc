// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_kernels/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::DefaultOpset;
using onnx_kernels::Sequence;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::SequenceAt;
using onnx_kernels::kernel::SequenceConstruct;
using onnx_kernels::kernel::SequenceEmpty;
using onnx_kernels::kernel::SequenceErase;
using onnx_kernels::kernel::SequenceInsert;
using onnx_kernels::kernel::SequenceLength;
using onnx_kernels::kernel::SplitToSequence;

namespace Test {

TEST(BackendKernelClass, SequenceConstructStacksInputsAlongNewAxis) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceConstruct seq{ctx};
  Tensor a = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  Tensor b = Tensor::FromFloat("", {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  Tensor c = Tensor::FromFloat("", {2, 3}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});
  Tensor out = seq({a, b, c});

  EXPECT_EQ(out.data_type, a.data_type);
  const std::vector<int64_t> expected_shape = {3, 2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.data.size(), a.data.size() + b.data.size() + c.data.size());
  // Bytes are the concatenation of the per-input buffers (row-major).
  std::vector<uint8_t> expected_bytes;
  expected_bytes.insert(expected_bytes.end(), a.data.begin(), a.data.end());
  expected_bytes.insert(expected_bytes.end(), b.data.begin(), b.data.end());
  expected_bytes.insert(expected_bytes.end(), c.data.begin(), c.data.end());
  EXPECT_EQ(out.data, expected_bytes);
}

TEST(BackendKernelClass, SequenceConstructSingleInputProducesUnitOuterDim) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceConstruct seq{ctx};
  Tensor a = Tensor::FromInt64("", {4}, {-1, 0, 1, 2});
  Tensor out = seq({a});
  const std::vector<int64_t> expected_shape = {1, 4};
  EXPECT_EQ(out.shape, expected_shape);
  EXPECT_EQ(out.data_type, a.data_type);
  EXPECT_EQ(out.data, a.data);
}

TEST(BackendKernelClass, SequenceConstructEmptyInputsProducesEmptySequence) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceConstruct seq{ctx};
  Tensor out = seq({});
  // Element type cannot be inferred from inputs.
  EXPECT_EQ(out.data_type, 0);
  const std::vector<int64_t> expected_shape = {0};
  EXPECT_EQ(out.shape, expected_shape);
  EXPECT_TRUE(out.data.empty());
}

TEST(BackendKernelClass, SequenceConstructRejectsBadInputsAndMismatchedOutput) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceConstruct seq{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Mismatched data_type across inputs.
  Tensor bad_dtype = Tensor::FromInt32("", {2}, {1, 2});
  EXPECT_THROW(seq({a, bad_dtype}), std::invalid_argument);

  // Mismatched shape across inputs.
  Tensor bad_shape = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  EXPECT_THROW(seq({a, bad_shape}), std::invalid_argument);

  // Undefined element type.
  Tensor bad_first;
  EXPECT_THROW(seq({bad_first}), std::invalid_argument);

  // In-place overload with a mismatched output buffer is rejected.
  Tensor bad_out_dtype("", static_cast<int32_t>(onnx_kernels::DataType::INT32), {2, 2},
                       std::vector<uint8_t>(4 * sizeof(int32_t)));
  EXPECT_THROW(seq({a, b}, bad_out_dtype), std::invalid_argument);

  Tensor bad_out_shape("", a.data_type, {3, 2}, std::vector<uint8_t>(6 * sizeof(float)));
  EXPECT_THROW(seq({a, b}, bad_out_shape), std::invalid_argument);
}

TEST(BackendKernelClass, SequenceConstructAsSequenceBuildsSequenceValue) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceConstruct seq{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {3}, {3.0f, 4.0f, 5.0f});

  Sequence out = seq.AsSequence({a, b});

  EXPECT_EQ(out.elem_type, a.data_type);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out.at(0).shape, a.shape);
  EXPECT_EQ(out.at(0).data, a.data);
  EXPECT_EQ(out.at(1).shape, b.shape);
  EXPECT_EQ(out.at(1).data, b.data);
}

TEST(BackendKernelClass, SequenceConstructAsSequenceEmptyIsUndefinedElemType) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceConstruct seq{ctx};
  Sequence out = seq.AsSequence({});
  EXPECT_EQ(out.elem_type, 0);
  EXPECT_TRUE(out.empty());
}

TEST(BackendKernelClass, SequenceConstructAsSequenceRejectsDtypeMismatch) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceConstruct seq{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor bad = Tensor::FromInt32("", {2}, {1, 2});
  EXPECT_THROW(seq.AsSequence({a, bad}), std::invalid_argument);
}

TEST(BackendKernelClass, SequenceLengthReturnsScalarInt64Count) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceLength op{ctx};
  Sequence seq("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT),
               {Tensor::FromFloat("", {2}, {1.0f, 2.0f}), Tensor::FromFloat("", {1}, {3.0f}),
                Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f})});

  Tensor out = op(seq);

  EXPECT_EQ(out.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
  EXPECT_TRUE(out.shape.empty());
  ASSERT_EQ(out.data.size(), sizeof(int64_t));
  EXPECT_EQ(*out.AsInt64(), 3);
}

TEST(BackendKernelClass, SequenceLengthHandlesEmptySequence) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceLength op{ctx};
  Sequence empty("", 0, {});

  Tensor out = op(empty);

  EXPECT_EQ(out.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
  EXPECT_TRUE(out.shape.empty());
  ASSERT_EQ(out.data.size(), sizeof(int64_t));
  EXPECT_EQ(*out.AsInt64(), 0);
}

// ──────────────────────────────────────────────────────────────────────
// SequenceEmpty kernel tests.
// ──────────────────────────────────────────────────────────────────────

TEST(BackendKernelClass, SequenceEmptyDefaultDtypeIsFloat) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceEmpty op{ctx};
  Sequence out = op();
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(out.size(), 0u);
  EXPECT_EQ(out.elem_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
}

TEST(BackendKernelClass, SequenceEmptyUndefinedDtypeFallsBackToFloat) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceEmpty op{ctx};
  Sequence out = op(static_cast<int32_t>(onnx_kernels::DataType::UNDEFINED));
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(out.elem_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
}

TEST(BackendKernelClass, SequenceEmptyHonoursExplicitDtype) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceEmpty op{ctx};
  Sequence out = op(static_cast<int32_t>(onnx_kernels::DataType::INT64));
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(out.elem_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
}

// ──────────────────────────────────────────────────────────────────────
// ConcatFromSequence kernel tests.
// ──────────────────────────────────────────────────────────────────────

using onnx_kernels::kernel::ConcatFromSequence;

TEST(BackendKernelClass, ConcatFromSequenceAxis0ConcatenatesAlongLeadingAxis) {
  const KernelContext ctx{DefaultOpset(11)};
  ConcatFromSequence op{ctx};
  Tensor a = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b = Tensor::FromFloat("", {1, 3}, {7.0f, 8.0f, 9.0f});
  Tensor c = Tensor::FromFloat("", {2, 3}, {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f});

  Tensor out = op({a, b, c}, /*axis=*/0);

  EXPECT_EQ(out.data_type, a.data_type);
  const std::vector<int64_t> expected_shape = {5, 3};
  EXPECT_EQ(out.shape, expected_shape);
  // Bytes are simple concatenation along the leading axis.
  std::vector<uint8_t> expected_bytes;
  expected_bytes.insert(expected_bytes.end(), a.data.begin(), a.data.end());
  expected_bytes.insert(expected_bytes.end(), b.data.begin(), b.data.end());
  expected_bytes.insert(expected_bytes.end(), c.data.begin(), c.data.end());
  EXPECT_EQ(out.data, expected_bytes);
}

TEST(BackendKernelClass, ConcatFromSequenceAxis1InterleavesPerOuterRow) {
  const KernelContext ctx{DefaultOpset(11)};
  ConcatFromSequence op{ctx};
  Tensor a = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor b = Tensor::FromFloat("", {2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});

  Tensor out = op({a, b}, /*axis=*/1);

  const std::vector<int64_t> expected_shape = {2, 5};
  EXPECT_EQ(out.shape, expected_shape);
  const std::vector<float> expected_values = {1.0f, 2.0f, 5.0f, 6.0f, 7.0f,
                                              3.0f, 4.0f, 8.0f, 9.0f, 10.0f};
  ASSERT_EQ(out.data.size(), expected_values.size() * sizeof(float));
  std::vector<float> got(expected_values.size());
  std::memcpy(got.data(), out.data.data(), out.data.size());
  EXPECT_EQ(got, expected_values);
}

TEST(BackendKernelClass, ConcatFromSequenceNegativeAxisResolvesAgainstRank) {
  const KernelContext ctx{DefaultOpset(11)};
  ConcatFromSequence op{ctx};
  Tensor a = Tensor::FromInt64("", {2, 2}, {1, 2, 3, 4});
  Tensor b = Tensor::FromInt64("", {2, 1}, {5, 6});

  Tensor out = op({a, b}, /*axis=*/-1);

  const std::vector<int64_t> expected_shape = {2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  const std::vector<int64_t> expected_values = {1, 2, 5, 3, 4, 6};
  std::vector<int64_t> got(expected_values.size());
  std::memcpy(got.data(), out.data.data(), out.data.size());
  EXPECT_EQ(got, expected_values);
}

TEST(BackendKernelClass, ConcatFromSequenceNewAxisStacksAlongNewLeadingAxis) {
  const KernelContext ctx{DefaultOpset(11)};
  ConcatFromSequence op{ctx};
  Tensor a = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b = Tensor::FromFloat("", {2, 3}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  Tensor c = Tensor::FromFloat("", {2, 3}, {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f});

  Tensor out = op({a, b, c}, /*axis=*/0, /*new_axis=*/1);

  const std::vector<int64_t> expected_shape = {3, 2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  std::vector<uint8_t> expected_bytes;
  expected_bytes.insert(expected_bytes.end(), a.data.begin(), a.data.end());
  expected_bytes.insert(expected_bytes.end(), b.data.begin(), b.data.end());
  expected_bytes.insert(expected_bytes.end(), c.data.begin(), c.data.end());
  EXPECT_EQ(out.data, expected_bytes);
}

TEST(BackendKernelClass, ConcatFromSequenceNewAxisAtTailAppendsLengthDim) {
  const KernelContext ctx{DefaultOpset(11)};
  ConcatFromSequence op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor c = Tensor::FromFloat("", {2}, {5.0f, 6.0f});

  Tensor out = op({a, b, c}, /*axis=*/-1, /*new_axis=*/1);

  const std::vector<int64_t> expected_shape = {2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  const std::vector<float> expected_values = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};
  std::vector<float> got(expected_values.size());
  std::memcpy(got.data(), out.data.data(), out.data.size());
  EXPECT_EQ(got, expected_values);
}

TEST(BackendKernelClass, ConcatFromSequenceRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(11)};
  ConcatFromSequence op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Empty sequence.
  EXPECT_THROW(op({}, /*axis=*/0), std::invalid_argument);

  // axis out of range.
  EXPECT_THROW(op({a}, /*axis=*/2), std::invalid_argument);

  // new_axis must be 0 or 1.
  EXPECT_THROW(op({a, b}, /*axis=*/0, /*new_axis=*/2), std::invalid_argument);

  // Mismatched dtype.
  Tensor bad_dtype = Tensor::FromInt32("", {2}, {1, 2});
  EXPECT_THROW(op({a, bad_dtype}, /*axis=*/0), std::invalid_argument);

  // For new_axis=1, all inputs must share the exact same shape.
  Tensor bad_shape = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  EXPECT_THROW(op({a, bad_shape}, /*axis=*/0, /*new_axis=*/1), std::invalid_argument);

  // For new_axis=0, non-concat dimensions must match.
  Tensor mismatched_other = Tensor::FromFloat("", {3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  Tensor base = Tensor::FromFloat("", {2, 3}, {0, 0, 0, 0, 0, 0});
  EXPECT_THROW(op({base, mismatched_other}, /*axis=*/1), std::invalid_argument);
}

TEST(BackendKernelClass, ConcatFromSequenceInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(11)};
  ConcatFromSequence op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {3}, {3.0f, 4.0f, 5.0f});

  Tensor out("", a.data_type, {5}, std::vector<uint8_t>(5 * sizeof(float)));
  op({a, b}, /*axis=*/0, /*new_axis=*/0, out);

  std::vector<uint8_t> expected_bytes;
  expected_bytes.insert(expected_bytes.end(), a.data.begin(), a.data.end());
  expected_bytes.insert(expected_bytes.end(), b.data.begin(), b.data.end());
  EXPECT_EQ(out.data, expected_bytes);

  // Mismatched preallocated output shape is rejected.
  Tensor bad_shape("", a.data_type, {4}, std::vector<uint8_t>(4 * sizeof(float)));
  EXPECT_THROW(op({a, b}, 0, 0, bad_shape), std::invalid_argument);
}

// ──────────────────────────────────────────────────────────────────────
// SequenceErase kernel tests.
// ──────────────────────────────────────────────────────────────────────

TEST(BackendKernelClass, SequenceEraseDefaultRemovesLastElement) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceErase op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor c = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b, c});

  Sequence out = op(seq);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out.elem_type, a.data_type);
  EXPECT_EQ(out.at(0).data, a.data);
  EXPECT_EQ(out.at(1).data, b.data);
}

TEST(BackendKernelClass, SequenceErasePositivePositionRemovesMiddleElement) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceErase op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor c = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b, c});
  Tensor pos = Tensor::FromInt64("", {}, {1});

  Sequence out = op(seq, &pos);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out.at(0).data, a.data);
  EXPECT_EQ(out.at(1).data, c.data);
}

TEST(BackendKernelClass, SequenceEraseNegativePositionCountsFromBack) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceErase op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor c = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b, c});
  Tensor pos = Tensor::FromInt64("", {}, {-2}); // index 1

  Sequence out = op(seq, &pos);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out.at(0).data, a.data);
  EXPECT_EQ(out.at(1).data, c.data);
}

TEST(BackendKernelClass, SequenceEraseInt32PositionIsAccepted) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceErase op{ctx};
  Tensor a = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor b = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b});
  Tensor pos = Tensor::FromInt32("", {}, {0});

  Sequence out = op(seq, &pos);

  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out.at(0).data, b.data);
}

TEST(BackendKernelClass, SequenceErasePreservesElemTypeOnEmptyResult) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceErase op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Sequence seq("", a.data_type, {a});

  Sequence out = op(seq); // removes the only element

  EXPECT_EQ(out.size(), 0u);
  EXPECT_EQ(out.elem_type, a.data_type);
}

TEST(BackendKernelClass, SequenceEraseRejectsEmptySequenceWithNoPosition) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceErase op{ctx};
  const Sequence empty("", 0, {});

  EXPECT_THROW(op(empty), std::invalid_argument);
}

TEST(BackendKernelClass, SequenceEraseRejectsOutOfRangePosition) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceErase op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Sequence seq("", a.data_type, {a});
  Tensor pos = Tensor::FromInt64("", {}, {5});

  EXPECT_THROW(op(seq, &pos), std::invalid_argument);
}

TEST(BackendKernelClass, SequenceInsertDefaultAppendsToBack) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceInsert op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor x = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b});

  Sequence out = op(seq, x);

  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out.at(0).data, a.data);
  EXPECT_EQ(out.at(1).data, b.data);
  EXPECT_EQ(out.at(2).data, x.data);
}

TEST(BackendKernelClass, SequenceInsertPositionAndNegativePosition) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceInsert op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor c = Tensor::FromFloat("", {2}, {7.0f, 8.0f});
  Tensor x = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b, c});

  Tensor pos = Tensor::FromInt64("", {}, {1});
  Sequence out_pos = op(seq, x, &pos);
  ASSERT_EQ(out_pos.size(), 4u);
  EXPECT_EQ(out_pos.at(0).data, a.data);
  EXPECT_EQ(out_pos.at(1).data, x.data);
  EXPECT_EQ(out_pos.at(2).data, b.data);
  EXPECT_EQ(out_pos.at(3).data, c.data);

  Tensor neg = Tensor::FromInt64("", {}, {-1});
  Sequence out_neg = op(seq, x, &neg);
  ASSERT_EQ(out_neg.size(), 4u);
  EXPECT_EQ(out_neg.at(0).data, a.data);
  EXPECT_EQ(out_neg.at(1).data, b.data);
  EXPECT_EQ(out_neg.at(2).data, x.data);
  EXPECT_EQ(out_neg.at(3).data, c.data);
}

TEST(BackendKernelClass, SequenceInsertRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceInsert op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor x = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a});

  Tensor bad_dtype = Tensor::FromInt32("", {2}, {1, 2});
  EXPECT_THROW(op(seq, bad_dtype), std::invalid_argument);

  Tensor pos_oob = Tensor::FromInt64("", {}, {2});
  EXPECT_THROW(op(seq, x, &pos_oob), std::invalid_argument);
}

// ──────────────────────────────────────────────────────────────────────
// SequenceAt kernel tests.
// ──────────────────────────────────────────────────────────────────────

TEST(BackendKernelClass, SequenceAtReturnsElementAtPosition) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceAt op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor c = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b, c});

  Tensor pos0 = Tensor::FromInt64("", {}, {0});
  Tensor out0 = op(seq, pos0);
  EXPECT_EQ(out0.data_type, a.data_type);
  EXPECT_EQ(out0.shape, a.shape);
  EXPECT_EQ(out0.data, a.data);

  Tensor pos2 = Tensor::FromInt64("", {}, {2});
  Tensor out2 = op(seq, pos2);
  EXPECT_EQ(out2.data, c.data);
}

TEST(BackendKernelClass, SequenceAtNegativePositionCountsFromBack) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceAt op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor c = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b, c});
  Tensor pos = Tensor::FromInt64("", {}, {-2}); // index 1

  Tensor out = op(seq, pos);
  EXPECT_EQ(out.data, b.data);
}

TEST(BackendKernelClass, SequenceAtInt32PositionIsAccepted) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceAt op{ctx};
  Tensor a = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor b = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
  const Sequence seq("", a.data_type, {a, b});
  Tensor pos = Tensor::FromInt32("", {}, {1});

  Tensor out = op(seq, pos);
  EXPECT_EQ(out.data, b.data);
}

TEST(BackendKernelClass, SequenceAtRejectsEmptySequence) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceAt op{ctx};
  const Sequence empty("", 0, {});
  Tensor pos = Tensor::FromInt64("", {}, {0});
  EXPECT_THROW(op(empty, pos), std::invalid_argument);
}

TEST(BackendKernelClass, SequenceAtRejectsOutOfRangePosition) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceAt op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Sequence seq("", a.data_type, {a});
  Tensor pos = Tensor::FromInt64("", {}, {5});
  EXPECT_THROW(op(seq, pos), std::invalid_argument);
}

TEST(BackendKernelClass, SequenceAtRejectsNonScalarPosition) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceAt op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Sequence seq("", a.data_type, {a});
  Tensor pos = Tensor::FromInt64("", {1}, {0});
  EXPECT_THROW(op(seq, pos), std::invalid_argument);
}

// ──────────────────────────────────────────────────────────────────────
// SequenceMap kernel tests.
// ──────────────────────────────────────────────────────────────────────

TEST(BackendKernelClass, SequenceMapBuildsOneSequencePerBodyOutput) {
  const KernelContext ctx{DefaultOpset(17)};
  onnx_kernels::kernel::SequenceMap op{ctx};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor c = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  const Sequence in_seq("", a.data_type, {a, b, c});

  // Single body output: identity-like per-iteration tensors.
  std::vector<std::vector<Tensor>> body_out = {{a, b, c}};
  std::vector<Sequence> outs = op(in_seq, body_out);

  ASSERT_EQ(outs.size(), 1u);
  ASSERT_EQ(outs[0].size(), 3u);
  EXPECT_EQ(outs[0].elem_type, a.data_type);
  EXPECT_EQ(outs[0].at(0).data, a.data);
  EXPECT_EQ(outs[0].at(1).data, b.data);
  EXPECT_EQ(outs[0].at(2).data, c.data);
}

TEST(BackendKernelClass, SequenceMapBuildsMultipleOutputSequences) {
  const KernelContext ctx{DefaultOpset(17)};
  onnx_kernels::kernel::SequenceMap op{ctx};
  Tensor a = Tensor::FromFloat("", {1}, {1.0f});
  Tensor b = Tensor::FromFloat("", {1}, {2.0f});
  Tensor x = Tensor::FromInt64("", {1}, {7});
  Tensor y = Tensor::FromInt64("", {1}, {8});
  const Sequence in_seq("", a.data_type, {a, b});

  // Two body outputs (mixed dtypes), each with one tensor per iteration.
  std::vector<std::vector<Tensor>> body_out = {{a, b}, {x, y}};
  std::vector<Sequence> outs = op(in_seq, body_out);

  ASSERT_EQ(outs.size(), 2u);
  ASSERT_EQ(outs[0].size(), 2u);
  ASSERT_EQ(outs[1].size(), 2u);
  EXPECT_EQ(outs[0].elem_type, a.data_type);
  EXPECT_EQ(outs[1].elem_type, x.data_type);
  EXPECT_EQ(outs[1].at(0).data, x.data);
  EXPECT_EQ(outs[1].at(1).data, y.data);
}

TEST(BackendKernelClass, SequenceMapRejectsRowLengthMismatch) {
  const KernelContext ctx{DefaultOpset(17)};
  onnx_kernels::kernel::SequenceMap op{ctx};
  Tensor a = Tensor::FromFloat("", {1}, {1.0f});
  Tensor b = Tensor::FromFloat("", {1}, {2.0f});
  const Sequence in_seq("", a.data_type, {a, b});

  // Body row has only one tensor — does not match the input sequence length.
  std::vector<std::vector<Tensor>> body_out = {{a}};
  EXPECT_THROW(op(in_seq, body_out), std::invalid_argument);
}

TEST(BackendKernelClass, SequenceMapRejectsMixedDtypeWithinOneOutput) {
  const KernelContext ctx{DefaultOpset(17)};
  onnx_kernels::kernel::SequenceMap op{ctx};
  Tensor a = Tensor::FromFloat("", {1}, {1.0f});
  Tensor x = Tensor::FromInt64("", {1}, {7});
  const Sequence in_seq("", a.data_type, {a, a});

  // Body row mixes FLOAT and INT64 tensors.
  std::vector<std::vector<Tensor>> body_out = {{a, x}};
  EXPECT_THROW(op(in_seq, body_out), std::invalid_argument);
}

TEST(BackendKernelClass, SequenceMapPreservesElemTypeOnEmptyInputSequence) {
  const KernelContext ctx{DefaultOpset(17)};
  onnx_kernels::kernel::SequenceMap op{ctx};
  const Sequence in_seq("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT), {});

  // Zero iterations: each body output row is empty; the resulting output
  // sequence is empty and its elem_type degrades to UNDEFINED (since no
  // sample tensors are available).
  std::vector<std::vector<Tensor>> body_out = {{}};
  std::vector<Sequence> outs = op(in_seq, body_out);

  ASSERT_EQ(outs.size(), 1u);
  EXPECT_EQ(outs[0].size(), 0u);
  EXPECT_EQ(outs[0].elem_type, static_cast<int32_t>(onnx_kernels::DataType::UNDEFINED));
}

// ──────────────────────────────────────────────────────────────────────
// SplitToSequence kernel tests.
// ──────────────────────────────────────────────────────────────────────

TEST(BackendKernelClass, SplitToSequenceScalarSplitProducesEqualChunks) {
  const KernelContext ctx{DefaultOpset(11)};
  SplitToSequence op{ctx};
  // arange(18) reshaped to [3, 6].
  std::vector<float> data(18);
  for (int i = 0; i < 18; ++i)
    data[static_cast<std::size_t>(i)] = static_cast<float>(i);
  Tensor input = Tensor::FromFloat("", {3, 6}, data);
  Tensor split = Tensor::FromInt64("", {}, {2});

  Sequence out = op(input, &split, /*axis=*/1);

  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out.elem_type, input.data_type);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(out.at(i).shape, (std::vector<int64_t>{3, 2}));
  }
  const float expected0[] = {0.f, 1.f, 6.f, 7.f, 12.f, 13.f};
  const float expected1[] = {2.f, 3.f, 8.f, 9.f, 14.f, 15.f};
  const float expected2[] = {4.f, 5.f, 10.f, 11.f, 16.f, 17.f};
  EXPECT_EQ(0, std::memcmp(out.at(0).data.data(), expected0, sizeof(expected0)));
  EXPECT_EQ(0, std::memcmp(out.at(1).data.data(), expected1, sizeof(expected1)));
  EXPECT_EQ(0, std::memcmp(out.at(2).data.data(), expected2, sizeof(expected2)));
}

TEST(BackendKernelClass, SplitToSequenceVectorSplitProducesUnevenChunks) {
  const KernelContext ctx{DefaultOpset(11)};
  SplitToSequence op{ctx};
  std::vector<float> data(18);
  for (int i = 0; i < 18; ++i)
    data[static_cast<std::size_t>(i)] = static_cast<float>(i);
  Tensor input = Tensor::FromFloat("", {3, 6}, data);
  Tensor split = Tensor::FromInt64("", {2}, {1, 2});

  Sequence out = op(input, &split, /*axis=*/0);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out.at(0).shape, (std::vector<int64_t>{1, 6}));
  EXPECT_EQ(out.at(1).shape, (std::vector<int64_t>{2, 6}));
  const float row0[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  const float rows12[] = {6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f};
  EXPECT_EQ(0, std::memcmp(out.at(0).data.data(), row0, sizeof(row0)));
  EXPECT_EQ(0, std::memcmp(out.at(1).data.data(), rows12, sizeof(rows12)));
}

TEST(BackendKernelClass, SplitToSequenceOmittedSplitKeepdimsDefaults) {
  const KernelContext ctx{DefaultOpset(11)};
  SplitToSequence op{ctx};
  std::vector<float> data(18);
  for (int i = 0; i < 18; ++i)
    data[static_cast<std::size_t>(i)] = static_cast<float>(i);
  Tensor input = Tensor::FromFloat("", {3, 6}, data);

  Sequence out = op(input, /*split=*/nullptr, /*axis=*/1, /*keepdims=*/1);

  ASSERT_EQ(out.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(out.at(i).shape, (std::vector<int64_t>{3, 1}));
  }
}

TEST(BackendKernelClass, SplitToSequenceOmittedSplitNoKeepdimsSqueezesAxis) {
  const KernelContext ctx{DefaultOpset(11)};
  SplitToSequence op{ctx};
  std::vector<float> data(18);
  for (int i = 0; i < 18; ++i)
    data[static_cast<std::size_t>(i)] = static_cast<float>(i);
  Tensor input = Tensor::FromFloat("", {3, 6}, data);

  Sequence out = op(input, /*split=*/nullptr, /*axis=*/1, /*keepdims=*/0);

  ASSERT_EQ(out.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(out.at(i).shape, (std::vector<int64_t>{3}));
  }
  // Column i of input (rows are 0..6, 6..12, 12..18 → column i values are i, i+6, i+12).
  const float col1[] = {1.f, 7.f, 13.f};
  EXPECT_EQ(0, std::memcmp(out.at(1).data.data(), col1, sizeof(col1)));
}

TEST(BackendKernelClass, SplitToSequenceNegativeAxisIsAccepted) {
  const KernelContext ctx{DefaultOpset(11)};
  SplitToSequence op{ctx};
  Tensor input = Tensor::FromFloat("", {2, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  Tensor split = Tensor::FromInt64("", {}, {2});

  Sequence out = op(input, &split, /*axis=*/-1);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out.at(0).shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(out.at(1).shape, (std::vector<int64_t>{2, 2}));
}

TEST(BackendKernelClass, SplitToSequenceRejectsScalarInput) {
  const KernelContext ctx{DefaultOpset(11)};
  SplitToSequence op{ctx};
  Tensor input = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW(op(input, /*split=*/nullptr), std::invalid_argument);
}

TEST(BackendKernelClass, SplitToSequenceRejectsOutOfRangeAxis) {
  const KernelContext ctx{DefaultOpset(11)};
  SplitToSequence op{ctx};
  Tensor input = Tensor::FromFloat("", {2}, {1.f, 2.f});
  EXPECT_THROW(op(input, /*split=*/nullptr, /*axis=*/3), std::invalid_argument);
}

TEST(BackendKernelClass, SplitToSequenceRejectsMismatchedSplitSum) {
  const KernelContext ctx{DefaultOpset(11)};
  SplitToSequence op{ctx};
  Tensor input = Tensor::FromFloat("", {3, 6}, std::vector<float>(18, 0.f));
  Tensor split = Tensor::FromInt64("", {2}, {1, 1}); // sums to 2, not 3
  EXPECT_THROW(op(input, &split, /*axis=*/0), std::invalid_argument);
}

} // namespace Test
