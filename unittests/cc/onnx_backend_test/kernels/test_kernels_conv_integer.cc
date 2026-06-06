// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_kernels/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::ConvInteger;
using onnx_kernels::kernel::KernelContext;

namespace Test {

// Basic 1x1 ConvInteger with explicit x_zero_point.
// X = [[[[2, 3, 4], [5, 6, 7], [8, 9, 10]]]] (uint8)
// W = [[[[1, 1], [1, 1]]]] (uint8)
// x_zero_point = 1.
// Each 2x2 window sum of (x - 1) gives:
//   Y[0,0] = (2-1)+(3-1)+(5-1)+(6-1) = 12
//   Y[0,1] = (3-1)+(4-1)+(6-1)+(7-1) = 16
//   Y[1,0] = (5-1)+(6-1)+(8-1)+(9-1) = 24
//   Y[1,1] = (6-1)+(7-1)+(9-1)+(10-1) = 28
TEST(BackendKernelClass, ConvIntegerBasicWithoutPaddingMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(10)};
  const ConvInteger ci{ctx};
  Tensor x = Tensor::FromUint8("", {1, 1, 3, 3}, {2, 3, 4, 5, 6, 7, 8, 9, 10});
  Tensor w = Tensor::FromUint8("", {1, 1, 2, 2}, {1, 1, 1, 1});
  Tensor xzp = Tensor::FromUint8("", {}, {1});
  Tensor wzp;
  ConvInteger::Attributes attrs;
  attrs.kernel_shape = {2, 2};
  Tensor y = ci(x, w, xzp, wzp, attrs);
  ASSERT_EQ(y.data_type, 6 /* INT32 */);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 2, 2}));
  const std::vector<int32_t> expected{12, 16, 24, 28};
  ASSERT_EQ(y.data.size(), expected.size() * sizeof(int32_t));
  const int32_t *py = y.AsInt32();
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]) << "index " << i;
  }
}

TEST(BackendKernelClass, ConvIntegerCanRunInPlaceIsFalse) {
  EXPECT_FALSE(ConvInteger::CanRunInPlace());
}

} // namespace Test
