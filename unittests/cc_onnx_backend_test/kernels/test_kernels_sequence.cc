// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Sequence;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::SequenceConstruct;
using onnx_backend_test::kernel::SequenceErase;
using onnx_backend_test::kernel::SequenceInsert;
using onnx_backend_test::kernel::SequenceLength;

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
  Tensor bad_out_dtype("", static_cast<int32_t>(TensorProto::DataType::INT32), {2, 2},
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
  Sequence seq("", static_cast<int32_t>(TensorProto::DataType::FLOAT),
               {Tensor::FromFloat("", {2}, {1.0f, 2.0f}), Tensor::FromFloat("", {1}, {3.0f}),
                Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f})});

  Tensor out = op(seq);

  EXPECT_EQ(out.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  EXPECT_TRUE(out.shape.empty());
  ASSERT_EQ(out.data.size(), sizeof(int64_t));
  EXPECT_EQ(*out.AsInt64(), 3);
}

TEST(BackendKernelClass, SequenceLengthHandlesEmptySequence) {
  const KernelContext ctx{DefaultOpset(11)};
  SequenceLength op{ctx};
  Sequence empty("", 0, {});

  Tensor out = op(empty);

  EXPECT_EQ(out.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  EXPECT_TRUE(out.shape.empty());
  ASSERT_EQ(out.data.size(), sizeof(int64_t));
  EXPECT_EQ(*out.AsInt64(), 0);
}

// ──────────────────────────────────────────────────────────────────────
// ConcatFromSequence kernel tests.
// ──────────────────────────────────────────────────────────────────────

using onnx_backend_test::kernel::ConcatFromSequence;

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

} // namespace Test
