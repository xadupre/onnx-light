// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;
using onnx_kernels::kernel::AveragePool;
using onnx_kernels::kernel::MaxUnpool;

namespace Test {

TEST(PoolingCompatibility, AveragePoolCeilDivisorCountsSampledPadding) {
  for (int64_t opset : {18, 19, 22}) {
    AveragePool pool{KernelContext{DefaultOpset(opset)}};
    Tensor x = Tensor::FromFloat("", {1, 1, 4}, {1, 2, 3, 4});
    Tensor included = pool(x, {3}, {2}, {}, true, true);
    EXPECT_EQ(included.shape, (Shape{1, 1, 2}));
    EXPECT_FLOAT_EQ(included.AsFloat()[0], 2);
    EXPECT_FLOAT_EQ(included.AsFloat()[1], 7.0f / 3);
    Tensor excluded = pool(x, {3}, {2}, {}, true, false);
    EXPECT_FLOAT_EQ(excluded.AsFloat()[1], 3.5f);
    Tensor floor = pool(x, {3}, {2}, {}, false, true);
    EXPECT_EQ(floor.shape, (Shape{1, 1, 1}));
    EXPECT_FLOAT_EQ(floor.AsFloat()[0], 2);
  }
}

TEST(PoolingCompatibility, AveragePoolDilatedCeilFacesEdgesAndCorner) {
  AveragePool pool{KernelContext{DefaultOpset(22)}};
  Tensor x = Tensor::FromFloat("", {1, 1, 32, 32, 32}, std::vector<float>(32 * 32 * 32, 1));
  for (bool include_pad : {false, true}) {
    Tensor y = pool(x, {5, 5, 5}, {3, 3, 3}, {}, true, include_pad, {2, 2, 2});
    ASSERT_EQ(y.shape, (Shape{1, 1, 9, 9, 9}));
    for (int64_t d = 0; d < 9; ++d) {
      for (int64_t h = 0; h < 9; ++h) {
        for (int64_t w = 0; w < 9; ++w) {
          const int64_t samples = (d == 8 ? 4 : 5) * (h == 8 ? 4 : 5) * (w == 8 ? 4 : 5);
          EXPECT_FLOAT_EQ(y.AsFloat()[(d * 9 + h) * 9 + w],
                          include_pad ? static_cast<float>(samples) / 125 : 1);
        }
      }
    }
  }
}

TEST(PoolingCompatibility, AveragePoolDilatedExplicitPaddingAndPreallocatedOutput) {
  AveragePool pool{KernelContext{DefaultOpset(22)}};
  Tensor x = Tensor::FromFloat("", {1, 1, 6}, {1, 2, 3, 4, 5, 6});
  Tensor y = pool(x, {3}, {3}, {1, 1}, true, true, {2});
  ASSERT_EQ(y.shape, (Shape{1, 1, 2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 2);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 8.0f / 3);
  pool(x, {3}, {3}, {1, 1}, true, false, y, {2});
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 3);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 4);
}

TEST(PoolingCompatibility, AveragePoolCeilDoesNotShiftWindowOrigin) {
  AveragePool pool{KernelContext{DefaultOpset(22)}};
  Tensor x = Tensor::FromFloat("", {1, 1, 7}, {1, 2, 3, 4, 5, 6, 7});
  Tensor y = pool(x, {5}, {4}, {}, true, true);
  ASSERT_EQ(y.shape, (Shape{1, 1, 2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 3);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 18.0f / 5);
}

TEST(PoolingCompatibility, MaxUnpoolGenericSpatialRankAndGlobalIndices) {
  MaxUnpool unpool{KernelContext{DefaultOpset(22)}};
  Tensor x1 = Tensor::FromFloat("", {2, 2, 1}, {1, 2, 3, 4});
  Tensor i1 = Tensor::FromInt64("", {2, 2, 1}, {1, 3, 5, 7});
  Tensor y1 = unpool(x1, i1, {2}, {2});
  ASSERT_EQ(y1.shape, (Shape{2, 2, 2}));
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(y1.AsFloat()[i], i % 2 ? static_cast<float>(i / 2 + 1) : 0);
  }
  Tensor x4 = Tensor::FromFloat("", {1, 1, 1, 1, 1, 2}, {1, 2});
  Tensor i4 = Tensor::FromInt64("", {1, 1, 1, 1, 1, 2}, {0, 31});
  Tensor y4 = unpool(x4, i4, {2, 2, 2, 2}, {2, 2, 2, 2});
  ASSERT_EQ(y4.shape, (Shape{1, 1, 2, 2, 2, 4}));
  for (int i = 0; i < 32; ++i) {
    EXPECT_FLOAT_EQ(y4.AsFloat()[i], i == 0 ? 1 : i == 31 ? 2 : 0);
  }
}

TEST(PoolingCompatibility, MaxUnpoolExplicitShapeOverridesInference) {
  MaxUnpool unpool{KernelContext{DefaultOpset(22)}};
  Tensor x = Tensor::FromFloat("", {1, 1, 2}, {3, 4});
  Tensor indices = Tensor::FromInt64("", {1, 1, 2}, {1, 2});
  Tensor shape = Tensor::FromInt64("", {3}, {1, 1, 3});
  Tensor y = unpool(x, indices, shape, {2}, {2});
  ASSERT_EQ(y.shape, (Shape{1, 1, 3}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 0);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 3);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 4);
  // Inference would produce a negative dimension; a supplied shape overrides it.
  y = unpool(x, indices, shape, {2}, {2}, {4, 4});
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 4);
  Tensor repeated = Tensor::FromInt64("", {1, 1, 2}, {2, 2});
  y = unpool(x, repeated, shape, {2}, {2});
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 0);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 4);
}

TEST(PoolingCompatibility, MaxUnpoolValidatesOutputShapeAndIndexBounds) {
  MaxUnpool unpool{KernelContext{DefaultOpset(22)}};
  Tensor x = Tensor::FromFloat("", {1, 1, 2}, {3, 4});
  Tensor indices = Tensor::FromInt64("", {1, 1, 2}, {1, 3});
  for (const std::vector<int64_t> &dims : {std::vector<int64_t>{1, 1},
                                           {1, 1, -1},
                                           {1, 1, 0},
                                           {1, 1, 3},
                                           {1, 2, std::numeric_limits<int64_t>::max()}}) {
    Tensor shape = Tensor::FromInt64("", {static_cast<int64_t>(dims.size())}, dims);
    EXPECT_THROW(unpool(x, indices, shape, {2}, {2}), std::invalid_argument);
  }
  Tensor float_shape = Tensor::FromFloat("", {3}, {1, 1, 4});
  EXPECT_THROW(unpool(x, indices, float_shape, {2}, {2}), std::invalid_argument);
  Tensor matrix_shape = Tensor::FromInt64("", {1, 3}, {1, 1, 4});
  EXPECT_THROW(unpool(x, indices, matrix_shape, {2}, {2}), std::invalid_argument);
  Tensor negative_index = Tensor::FromInt64("", {1, 1, 2}, {-1, 3});
  EXPECT_THROW(unpool(x, negative_index, {2}, {2}), std::invalid_argument);
}

TEST(PoolingCompatibility, MaxUnpoolEmptyExplicitOutput) {
  MaxUnpool unpool{KernelContext{DefaultOpset(22)}};
  Tensor x = Tensor::FromFloat("", {1, 1, 0}, {});
  Tensor indices = Tensor::FromInt64("", {1, 1, 0}, {});
  Tensor shape = Tensor::FromInt64("", {3}, {1, 1, 0});
  Tensor y = unpool(x, indices, shape, {2}, {2});
  EXPECT_EQ(y.shape, (Shape{1, 1, 0}));
  EXPECT_EQ(y.element_count(), 0u);
}

} // namespace Test
