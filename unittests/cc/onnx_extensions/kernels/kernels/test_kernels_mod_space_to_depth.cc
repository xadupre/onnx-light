// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::DataType;
using core::runtime::Shape;
using core::runtime::Tensor;
using onnx_kernels::kernel::KernelContext;

namespace Test {
namespace {

Tensor MakeModTensor(DataType dtype, const Shape &shape, const std::vector<double> &values) {
  if (dtype == DataType::DOUBLE) {
    return Tensor::FromDouble("", shape, values);
  }
  std::vector<float> floats;
  floats.reserve(values.size());
  for (double value : values) {
    floats.push_back(static_cast<float>(value));
  }
  if (dtype == DataType::FLOAT16) {
    return core::runtime::MakeFloat16Tensor("", shape, floats);
  }
  if (dtype == DataType::BFLOAT16) {
    return core::runtime::MakeBfloat16Tensor("", shape, floats);
  }
  return Tensor::FromFloat("", shape, floats);
}

double ModValue(const Tensor &tensor, size_t index) {
  if (tensor.data_type == DataType::DOUBLE) {
    return tensor.AsDouble()[index];
  }
  if (tensor.data_type == DataType::FLOAT) {
    return tensor.AsFloat()[index];
  }
  const uint16_t bits = reinterpret_cast<const uint16_t *>(tensor.bytes())[index];
  return tensor.data_type == DataType::FLOAT16 ? core::runtime::Float16BitsToFloat(bits)
                                               : core::runtime::Bfloat16BitsToFloat(bits);
}

class ModFloorTest : public ::testing::TestWithParam<DataType> {};

TEST_P(ModFloorTest, MixedSignsAndBroadcastWithAllocatedAndPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(28)};
  const onnx_kernels::kernel::Mod kernel{ctx};
  const Tensor x = MakeModTensor(GetParam(), {2, 1}, {-4.5, 4.5});
  const Tensor y = MakeModTensor(GetParam(), {1, 3}, {2.0, -2.0, 8.0});
  const std::vector<double> expected{1.5, -0.5, 3.5, 0.5, -1.5, 4.5};
  Tensor output = MakeModTensor(GetParam(), {2, 3}, std::vector<double>(6, 99.0));
  const Tensor allocated = kernel(x, y);
  kernel(x, y, 0, output);
  ASSERT_EQ(allocated.shape, (Shape{2, 3}));
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(ModValue(allocated, i), expected[i]);
    EXPECT_EQ(ModValue(output, i), expected[i]);
  }
}

TEST_P(ModFloorTest, SignedZerosNaNsAndInfinities) {
  const KernelContext ctx{DefaultOpset(28)};
  const onnx_kernels::kernel::Mod kernel{ctx};
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> a{0.0, -0.0, 0.0, -0.0, -3, 3,  -1,  1,    inf, -inf,
                              1,   1,    nan, 1,    4,  -4, 0.0, -0.0, inf, -inf};
  const std::vector<double> b{-2,  2,    2, -2,  inf, inf, -inf, -inf, 2,   2,
                              0.0, -0.0, 2, nan, -2,  2,   -inf, inf,  inf, -inf};
  const std::vector<double> expected{-0.0, 0.0, 0.0, -0.0, inf,  3,   -1,   -inf, nan, nan,
                                     nan,  nan, nan, nan,  -0.0, 0.0, -0.0, 0.0,  nan, nan};
  const Shape shape{static_cast<int64_t>(a.size())};
  const Tensor x = MakeModTensor(GetParam(), shape, a);
  const Tensor y = MakeModTensor(GetParam(), shape, b);
  Tensor output = MakeModTensor(GetParam(), shape, std::vector<double>(a.size(), 99.0));
  const Tensor allocated = kernel(x, y, 0);
  kernel(x, y, 0, output);
  for (const Tensor *result : {&allocated, static_cast<const Tensor *>(&output)}) {
    for (size_t i = 0; i < expected.size(); ++i) {
      SCOPED_TRACE(i);
      const double actual = ModValue(*result, i);
      if (std::isnan(expected[i])) {
        EXPECT_TRUE(std::isnan(actual));
      } else {
        EXPECT_EQ(actual, expected[i]);
        EXPECT_EQ(std::signbit(actual), std::signbit(expected[i]));
      }
    }
  }
}

TEST_P(ModFloorTest, RejectsInvalidFmod) {
  const KernelContext ctx{DefaultOpset(28)};
  const onnx_kernels::kernel::Mod kernel{ctx};
  const Tensor x = MakeModTensor(GetParam(), {1}, {3.0});
  const Tensor y = MakeModTensor(GetParam(), {1}, {2.0});
  Tensor output = MakeModTensor(GetParam(), {1}, {0.0});
  EXPECT_THROW(kernel(x, y, 2), std::invalid_argument);
  EXPECT_THROW(kernel(x, y, -1, output), std::invalid_argument);
}

TEST_P(ModFloorTest, FmodStillPreservesDividendSign) {
  const KernelContext ctx{DefaultOpset(28)};
  const onnx_kernels::kernel::Mod kernel{ctx};
  const Tensor x = MakeModTensor(GetParam(), {4}, {-4.5, 4.5, -0.0, 0.0});
  const Tensor y = MakeModTensor(GetParam(), {4}, {2, -2, 2, -2});
  const std::vector<double> expected{-0.5, 0.5, -0.0, 0.0};
  Tensor output = MakeModTensor(GetParam(), {4}, std::vector<double>(4, 99.0));
  const Tensor allocated = kernel(x, y, 1);
  kernel(x, y, 1, output);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(ModValue(allocated, i), expected[i]);
    EXPECT_EQ(std::signbit(ModValue(allocated, i)), std::signbit(expected[i]));
    EXPECT_EQ(ModValue(output, i), expected[i]);
    EXPECT_EQ(std::signbit(ModValue(output, i)), std::signbit(expected[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(Weekly, ModFloorTest,
                         ::testing::Values(DataType::FLOAT16, DataType::FLOAT, DataType::DOUBLE,
                                           DataType::BFLOAT16));

TEST(KernelClass, SpaceToDepthModesPreserveSpatialOrdering) {
  const KernelContext ctx{DefaultOpset(28)};
  const onnx_kernels::kernel::SpaceToDepth kernel{ctx};
  for (const int64_t blocksize : {2, 3}) {
    const int64_t n = 2, c = 3, h = blocksize * 2, w = blocksize * 3;
    std::vector<int64_t> values(static_cast<size_t>(n * c * h * w));
    std::iota(values.begin(), values.end(), int64_t{0});
    const Tensor input = Tensor::FromInt64("", {n, c, h, w}, values);
    for (const std::string mode : {"DCR", "CRD"}) {
      SCOPED_TRACE(mode);
      SCOPED_TRACE(blocksize);
      onnx_kernels::kernel::SpaceToDepth::Attributes attrs;
      attrs.blocksize = blocksize;
      attrs.mode = mode;
      const Shape output_shape{n, c * blocksize * blocksize, 2, 3};
      Tensor output = Tensor::FromInt64("", output_shape, std::vector<int64_t>(values.size(), -1));
      const Tensor allocated = kernel(input, attrs);
      kernel(input, attrs, output);
      ASSERT_EQ(allocated.shape, output_shape);
      if (mode == "DCR") {
        onnx_kernels::kernel::SpaceToDepth::Attributes default_attrs;
        default_attrs.blocksize = blocksize;
        const Tensor default_output = kernel(input, default_attrs);
        for (size_t i = 0; i < values.size(); ++i) {
          EXPECT_EQ(default_output.AsInt64()[i], allocated.AsInt64()[i]);
        }
      }
      for (int64_t batch = 0; batch < n; ++batch) {
        for (int64_t channel = 0; channel < c; ++channel) {
          for (int64_t row = 0; row < h; ++row) {
            for (int64_t col = 0; col < w; ++col) {
              const int64_t phase = (row % blocksize) * blocksize + col % blocksize;
              const int64_t out_channel =
                  mode == "DCR" ? phase * c + channel : channel * blocksize * blocksize + phase;
              const int64_t out_index =
                  ((batch * output_shape[1] + out_channel) * 2 + row / blocksize) * 3 +
                  col / blocksize;
              const int64_t expected = ((batch * c + channel) * h + row) * w + col;
              EXPECT_EQ(allocated.AsInt64()[out_index], expected);
              EXPECT_EQ(output.AsInt64()[out_index], expected);
            }
          }
        }
      }
      onnx_kernels::kernel::DepthToSpace inverse{ctx};
      onnx_kernels::kernel::DepthToSpace::Attributes inverse_attrs;
      inverse_attrs.blocksize = blocksize;
      inverse_attrs.mode = mode;
      const Tensor restored = inverse(allocated, inverse_attrs);
      ASSERT_EQ(restored.shape, input.shape);
      for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(restored.AsInt64()[i], values[i]);
      }
    }
  }
}

TEST(KernelClass, SpaceToDepthRejectsInvalidMode) {
  const KernelContext ctx{DefaultOpset(28)};
  const onnx_kernels::kernel::SpaceToDepth kernel{ctx};
  const Tensor input = Tensor::FromFloat("", {1, 1, 2, 2}, {1, 2, 3, 4});
  Tensor output = Tensor::FromFloat("", {1, 4, 1, 1}, {0, 0, 0, 0});
  onnx_kernels::kernel::SpaceToDepth::Attributes attrs;
  attrs.blocksize = 2;
  attrs.mode = "invalid";
  EXPECT_THROW(kernel(input, attrs), std::invalid_argument);
  EXPECT_THROW(kernel(input, attrs, output), std::invalid_argument);
}

} // namespace
} // namespace Test
