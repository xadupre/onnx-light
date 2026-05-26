// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Sequence;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::SequenceConstruct;

namespace Test {

TEST(BackendKernelClass, SequenceConstructStacksInputsAlongNewAxis) {
  SequenceConstruct seq{KernelContext(DefaultOpset(11))};
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
  SequenceConstruct seq{KernelContext(DefaultOpset(11))};
  Tensor a = Tensor::FromInt64("", {4}, {-1, 0, 1, 2});
  Tensor out = seq({a});
  const std::vector<int64_t> expected_shape = {1, 4};
  EXPECT_EQ(out.shape, expected_shape);
  EXPECT_EQ(out.data_type, a.data_type);
  EXPECT_EQ(out.data, a.data);
}

TEST(BackendKernelClass, SequenceConstructEmptyInputsProducesEmptySequence) {
  SequenceConstruct seq{KernelContext(DefaultOpset(11))};
  Tensor out = seq({});
  // Element type cannot be inferred from inputs.
  EXPECT_EQ(out.data_type, 0);
  const std::vector<int64_t> expected_shape = {0};
  EXPECT_EQ(out.shape, expected_shape);
  EXPECT_TRUE(out.data.empty());
}

TEST(BackendKernelClass, SequenceConstructRejectsBadInputsAndMismatchedOutput) {
  SequenceConstruct seq{KernelContext(DefaultOpset(11))};
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
  SequenceConstruct seq{KernelContext(DefaultOpset(11))};
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
  SequenceConstruct seq{KernelContext(DefaultOpset(11))};
  Sequence out = seq.AsSequence({});
  EXPECT_EQ(out.elem_type, 0);
  EXPECT_TRUE(out.empty());
}

TEST(BackendKernelClass, SequenceConstructAsSequenceRejectsDtypeMismatch) {
  SequenceConstruct seq{KernelContext(DefaultOpset(11))};
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor bad = Tensor::FromInt32("", {2}, {1, 2});
  EXPECT_THROW(seq.AsSequence({a, bad}), std::invalid_argument);
}

} // namespace Test
