// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::Tensor;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::ScatterElements;
using onnx_kernels::kernel::ScatterND;

namespace Test {
namespace {

Tensor Mean(bool nd, const Tensor &data, const Tensor &indices, const Tensor &updates,
            int64_t opset = 29) {
  const KernelContext context{DefaultOpset(opset)};
  if (nd) {
    ScatterND::Attributes attrs;
    attrs.reduction = "mean";
    return ScatterND(context)(data, indices, updates, attrs);
  }
  ScatterElements::Attributes attrs;
  attrs.reduction = "mean";
  return ScatterElements(context)(data, indices, updates, attrs);
}

Tensor Indices(bool nd, const std::vector<int64_t> &values) {
  const int64_t n = static_cast<int64_t>(values.size());
  return Tensor::FromInt64("", nd ? core::runtime::Shape{n, 1} : core::runtime::Shape{n}, values);
}

template <typename T> void CheckIntegerMean() {
  const T low = std::numeric_limits<T>::min();
  const T high = std::numeric_limits<T>::max();
  for (bool nd : {false, true}) {
    const Tensor data = Tensor::From<T>("", {4}, {low, high, high, T{7}});
    const Tensor updates = Tensor::From<T>("", {7}, {low, low, high, high, low, low, T{0}});
    const Tensor result = Mean(nd, data, Indices(nd, {0, -4, 1, 1, 2, 2, 2}), updates);
    EXPECT_EQ(result.As<T>()[0], low);
    EXPECT_EQ(result.As<T>()[1], high);
    if constexpr (std::is_signed_v<T>) {
      // (max + min + min + 0) // 4.
      EXPECT_EQ(result.As<T>()[2], static_cast<T>(low / 4 - 1));
    } else {
      EXPECT_EQ(result.As<T>()[2], static_cast<T>(high / 4));
    }
    EXPECT_EQ(result.As<T>()[3], 7);
    EXPECT_EQ(data.As<T>()[0], low);
  }
}

} // namespace

TEST(ScatterMeanKernel, IntegerExtremaAndFloor) {
  CheckIntegerMean<int8_t>();
  CheckIntegerMean<uint8_t>();
  CheckIntegerMean<int16_t>();
  CheckIntegerMean<uint16_t>();
  CheckIntegerMean<int32_t>();
  CheckIntegerMean<uint32_t>();
  CheckIntegerMean<int64_t>();
  CheckIntegerMean<uint64_t>();
  for (bool nd : {false, true}) {
    const auto out = Mean(nd, Tensor::FromInt64("", {3}, {-2, 8, 9}), Indices(nd, {0, -3, 1}),
                          Tensor::FromInt64("", {3}, {-3, -3, 5}));
    EXPECT_EQ(out.As<int64_t>()[0], -3);
    EXPECT_EQ(out.As<int64_t>()[1], 6);
    EXPECT_EQ(out.As<int64_t>()[2], 9);
    const auto mixed = Mean(nd, Tensor::FromInt64("", {1}, {INT64_MIN}), Indices(nd, {0}),
                            Tensor::FromInt64("", {1}, {INT64_MAX}));
    EXPECT_EQ(mixed.As<int64_t>()[0], -1);
  }
}

TEST(ScatterMeanKernel, FloatOverflowCancellationAndNonfiniteOrder) {
  const double max = std::numeric_limits<double>::max();
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<std::pair<std::array<double, 3>, double>> cases{
      {{max, max, max}, max},      {{-max, -max, -max}, -max},
      {{max, max, -max}, max / 3}, {{1e308, 1e-300, -1e308}, 1e-300 / 3},
      {{max, max, -inf}, -inf},    {{max, -inf, -inf}, -inf},
      {{inf, -inf, max}, nan},     {{nan, max, -inf}, nan}};
  for (bool nd : {false, true}) {
    for (const auto &[values, expected] : cases) {
      std::array<int, 3> order{0, 1, 2};
      do {
        const Tensor data = Tensor::FromDouble("", {2}, {values[order[0]], -0.0});
        const auto output = Mean(nd, data, Indices(nd, {0, -2}),
                                 Tensor::FromDouble("", {2}, {values[order[1]], values[order[2]]}));
        const double actual = output.As<double>()[0];
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(actual));
        } else if (std::isinf(expected)) {
          EXPECT_EQ(actual, expected);
        } else {
          EXPECT_NEAR(actual / expected, 1.0, 1e-14);
        }
        EXPECT_TRUE(std::signbit(output.As<double>()[1]));
      } while (std::next_permutation(order.begin(), order.end()));
    }
    const float max_float = std::numeric_limits<float>::max();
    const auto output = Mean(nd, Tensor::FromFloat("", {1}, {max_float}), Indices(nd, {0, 0}),
                             Tensor::FromFloat("", {2}, {max_float, max_float}));
    EXPECT_EQ(output.As<float>()[0], max_float);
  }
}

TEST(ScatterMeanKernel, EmptyAndUntouchedBits) {
  for (bool nd : {false, true}) {
    Tensor data = Tensor::FromDouble("", {3}, {0, -0.0, 1});
    const uint64_t payload = UINT64_C(0x7ff8000000000123);
    std::memcpy(data.mutable_bytes(), &payload, sizeof(payload));
    const auto empty = Mean(nd, data, Indices(nd, {}), Tensor::FromDouble("", {0}, {}));
    EXPECT_EQ(std::memcmp(empty.bytes(), data.bytes(), data.size_bytes()), 0);
    const auto output = Mean(nd, data, Indices(nd, {2}), Tensor::FromDouble("", {1}, {3}));
    EXPECT_EQ(std::memcmp(output.bytes(), data.bytes(), 2 * sizeof(double)), 0);
    EXPECT_EQ(output.As<double>()[2], 2);
  }
  const auto huge_empty = Mean(false, Tensor::FromFloat("", {0, INT64_MAX, INT64_MAX}, {}),
                               Tensor::FromInt64("", {0, INT64_MAX, INT64_MAX}, {}),
                               Tensor::FromFloat("", {0, INT64_MAX, INT64_MAX}, {}));
  EXPECT_EQ(huge_empty.element_count(), 0);
}

TEST(ScatterMeanKernel, OverflowRepairRetainsSubnormalResiduals) {
  const double max = std::numeric_limits<double>::max();
  const double tiny = std::numeric_limits<double>::denorm_min();
  for (bool nd : {false, true}) {
    for (double residual : {-16 * tiny, -4 * tiny, 4 * tiny, 16 * tiny}) {
      const auto output = Mean(nd, Tensor::FromDouble("", {1}, {max}), Indices(nd, {0, 0, 0, 0}),
                               Tensor::FromDouble("", {4}, {max, -max, -max, residual}));
      EXPECT_EQ(output.As<double>()[0], residual / 5);
    }
  }
}

TEST(ScatterMeanKernel, HalfAndBfloat16) {
  for (bool nd : {false, true}) {
    for (bool half : {false, true}) {
      const auto make = half ? core::runtime::MakeFloat16Tensor : core::runtime::MakeBfloat16Tensor;
      const auto decode =
          half ? core::runtime::Float16BitsToFloat : core::runtime::Bfloat16BitsToFloat;
      Tensor data = make("", {4}, {-2, 8, 9, -0.0}, nullptr);
      reinterpret_cast<uint16_t *>(data.mutable_bytes())[2] = half ? 0x7e35 : 0x7fc5;
      const auto output =
          Mean(nd, data, Indices(nd, {0, -4, 1}), make("", {3}, {-3, -3, 5}, nullptr));
      const auto *bits = reinterpret_cast<const uint16_t *>(output.bytes());
      const auto *original = reinterpret_cast<const uint16_t *>(data.bytes());
      EXPECT_NEAR(decode(bits[0]), -8.0f / 3, 0.02);
      EXPECT_EQ(decode(bits[1]), 6.5);
      EXPECT_EQ(bits[2], original[2]);
      EXPECT_EQ(bits[3], original[3]);
    }
  }
}

TEST(ScatterMeanKernel, ScatterNDSlicesAndEmptyTuples) {
  const Tensor data = Tensor::FromFloat("", {3, 2}, {1, 2, 7, 8, -0.0f, 10});
  const auto out = Mean(true, data, Tensor::FromInt32("", {2, 1}, {0, -3}),
                        Tensor::FromFloat("", {2, 2}, {3, 4, 5, 6}));
  EXPECT_EQ(out.As<float>()[0], 3);
  EXPECT_EQ(out.As<float>()[1], 4);
  EXPECT_EQ(std::memcmp(out.bytes() + 2 * sizeof(float), data.bytes() + 2 * sizeof(float),
                        4 * sizeof(float)),
            0);
  const auto whole =
      Mean(true, Tensor::FromInt64("", {2}, {-2, 8}), Tensor::FromInt64("", {2, 0}, {}),
           Tensor::FromInt64("", {2, 2}, {-3, 5, -3, 5}));
  EXPECT_EQ(whole.As<int64_t>()[0], -3);
  EXPECT_EQ(whole.As<int64_t>()[1], 6);
  const auto single = Mean(true, Tensor::FromFloat("", {2}, {1, 2}), Tensor::FromInt64("", {0}, {}),
                           Tensor::FromFloat("", {2}, {3, 4}));
  EXPECT_EQ(single.As<float>()[0], 2);
  EXPECT_EQ(single.As<float>()[1], 3);
  const auto empty = Mean(true, Tensor::FromFloat("", {0, 2}, {}),
                          Tensor::FromInt64("", {2, 0}, {}), Tensor::FromFloat("", {2, 0, 2}, {}));
  EXPECT_EQ(empty.element_count(), 0);
  const auto huge_empty = Mean(true, Tensor::FromFloat("", {0, INT64_MAX, INT64_MAX}, {}),
                               Tensor::FromInt64("", {1, 0}, {}),
                               Tensor::FromFloat("", {1, 0, INT64_MAX, INT64_MAX}, {}));
  EXPECT_EQ(huge_empty.element_count(), 0);
  EXPECT_THROW(Mean(true, Tensor::FromFloat("", {1, 0}, {}), Tensor::FromInt64("", {1, 1}, {7}),
                    Tensor::FromFloat("", {1, 0}, {})),
               std::invalid_argument);
  EXPECT_THROW(Mean(true, Tensor::FromFloat("", {0}, {}),
                    Tensor::FromInt64("", {INT64_MAX, INT64_MAX, 0}, {}),
                    Tensor::FromFloat("", {INT64_MAX, INT64_MAX, 0}, {})),
               std::invalid_argument);
}

TEST(ScatterMeanKernel, EmptyWholeTensorAvoidsZeroWorkLoop) {
  const KernelContext context{DefaultOpset(29)};
  const ScatterND kernel(context);
  for (const std::string reduction : {"none", "add", "mean"}) {
    ScatterND::Attributes attrs;
    attrs.reduction = reduction;
    const auto zero_work =
        kernel(Tensor::FromFloat("", {0}, {}), Tensor::FromInt64("", {INT64_MAX, 0}, {}),
               Tensor::FromFloat("", {INT64_MAX, 0}, {}), attrs);
    EXPECT_EQ(zero_work.element_count(), 0);
  }
}

TEST(ScatterMeanKernel, RejectsInvalidInputsEvenWithoutUpdates) {
  for (bool nd : {false, true}) {
    for (bool empty : {false, true}) {
      const Tensor indices = Indices(nd, empty ? std::vector<int64_t>{} : std::vector<int64_t>{0});
      const int64_t n = empty ? 0 : 1;
      EXPECT_THROW(Mean(nd, Tensor::FromFloat("", {1}, {1}), indices,
                        Tensor::FromFloat("", {n}, std::vector<float>(n, 2)), 28),
                   std::invalid_argument);
      EXPECT_THROW(Mean(nd, Tensor::FromBool("", {1}, {true}), indices,
                        Tensor::FromBool("", {n}, std::vector<uint8_t>(n, 0))),
                   std::invalid_argument);
      EXPECT_THROW(
          Mean(nd,
               Tensor::FromRawBytes("", core::runtime::DataType::COMPLEX64, {1},
                                    core::runtime::RawByteBuffer(8)),
               indices,
               Tensor::FromRawBytes("", core::runtime::DataType::COMPLEX64, {n},
                                    core::runtime::RawByteBuffer(static_cast<size_t>(n) * 8))),
          std::invalid_argument);
      EXPECT_THROW(Mean(nd, Tensor::MakeString("", {1}, {"old"}), indices,
                        Tensor::MakeString("", {n}, std::vector<std::string>(n, "new"))),
                   std::invalid_argument);
    }
    EXPECT_THROW(Mean(nd, Tensor::FromFloat("", {1}, {1}), Indices(nd, {-2}),
                      Tensor::FromFloat("", {1}, {2})),
                 std::invalid_argument);
    EXPECT_THROW(Mean(nd, Tensor::FromFloat("", {1}, {1}), Indices(nd, {1}),
                      Tensor::FromFloat("", {1}, {2})),
                 std::invalid_argument);
  }
  const KernelContext ctx{DefaultOpset(29)};
  const ScatterElements kernel(ctx);
  ScatterElements::Attributes attrs;
  for (const std::string reduction : {"none", "mean"}) {
    attrs.reduction = reduction;
    EXPECT_THROW(kernel(Tensor::FromFloat("", {1, 1}, {0}), Tensor::FromInt64("", {1, 2}, {0, 0}),
                        Tensor::FromFloat("", {1, 2}, {1, 2}), attrs),
                 std::invalid_argument);
  }
}

TEST(ScatterMeanKernel, InPlaceIncludesOriginalOnlyOnce) {
  const KernelContext context{DefaultOpset(29)};
  for (bool nd : {false, true}) {
    Tensor data = Tensor::FromInt64("", {3}, {-2, 8, 9});
    const Tensor indices = Indices(nd, {0, -3, 1});
    const Tensor updates = Tensor::FromInt64("", {3}, {-3, -3, 5});
    if (nd) {
      ScatterND::Attributes attrs;
      attrs.reduction = "mean";
      const ScatterND kernel(context);
      kernel(data, indices, updates, attrs, data);
    } else {
      ScatterElements::Attributes attrs;
      attrs.reduction = "mean";
      const ScatterElements kernel(context);
      kernel(data, indices, updates, attrs, data);
    }
    EXPECT_EQ(data.As<int64_t>()[0], -3);
    EXPECT_EQ(data.As<int64_t>()[1], 6);
    EXPECT_EQ(data.As<int64_t>()[2], 9);
  }
}

} // namespace Test
