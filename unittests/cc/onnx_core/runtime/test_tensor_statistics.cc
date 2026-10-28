// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/tensor_statistics.h"

#include "onnx_core/runtime/kernels/cast_float8.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include <thread>
#include <type_traits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

TEST(TensorStatistics, BasicAndDefaultHistogramThresholds) {
  for (int64_t count : {9, 10, 500, 501}) {
    std::vector<double> values(static_cast<size_t>(count));
    std::iota(values.begin(), values.end(), 0.0);
    const auto tensor = Tensor::FromDouble("X", {count}, values);
    const auto stats = ComputeTensorStatistics(tensor);
    EXPECT_EQ(stats.dtype, TensorProto::DOUBLE);
    EXPECT_EQ(stats.shape, std::vector<int64_t>({count}));
    EXPECT_EQ(stats.count, count);
    EXPECT_EQ(stats.size_bytes, values.size() * sizeof(double));
    EXPECT_DOUBLE_EQ(stats.min, 0);
    EXPECT_DOUBLE_EQ(stats.max, count - 1);
    EXPECT_DOUBLE_EQ(stats.mean, (count - 1) / 2.0);
    EXPECT_FALSE(stats.distinct_count);
    EXPECT_FALSE(stats.median);
    EXPECT_TRUE(stats.quantiles.empty());
    const size_t bins = count < 10 ? 0 : count > 500 ? 50 : 10;
    EXPECT_EQ(stats.distribution.size(), bins);
    EXPECT_EQ(stats.bin_edges.size(), bins == 0 ? 0 : bins + 1);
    if (bins) {
      EXPECT_EQ(std::accumulate(stats.distribution.begin(), stats.distribution.end(), int64_t{0}),
                count);
    }
  }
}

TEST(TensorStatistics, CustomHistogramAndConstant) {
  const auto tensor = Tensor::FromDouble("X", {10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto stats = ComputeTensorStatistics(tensor, 3);
  EXPECT_EQ(stats.distribution, std::vector<int64_t>({3, 3, 4}));
  EXPECT_EQ(stats.bin_edges, std::vector<double>({0, 3, 6, 9}));
  stats = ComputeTensorStatistics(Tensor::FromFloat("X", {10}, std::vector<float>(10, 2)));
  EXPECT_DOUBLE_EQ(stats.bin_edges.front(), 1.5);
  EXPECT_DOUBLE_EQ(stats.bin_edges.back(), 2.5);
  EXPECT_EQ(stats.distribution[5], 10);
}

TEST(TensorStatistics, HistogramMatchesRoundedBoundaryNeighbors) {
  const std::vector<double> values{0,
                                   1,
                                   std::nextafter(0.3, 0.0),
                                   0.3,
                                   std::nextafter(0.3, 1.0),
                                   std::nextafter(0.6, 0.0),
                                   0.6,
                                   std::nextafter(0.6, 1.0),
                                   std::nextafter(0.9, 0.0),
                                   0.9,
                                   std::nextafter(0.9, 1.0)};
  const auto stats = ComputeTensorStatistics(
      Tensor::FromDouble("X", {static_cast<int64_t>(values.size())}, values), 10);
  EXPECT_EQ(stats.distribution, std::vector<int64_t>({1, 0, 1, 2, 0, 1, 2, 0, 1, 3}));
}

TEST(TensorStatistics, HistogramHandlesCollapsedEdgesAndExtremeWidths) {
  const double largest = std::numeric_limits<double>::max();
  const double smallest = std::numeric_limits<double>::denorm_min();
  const std::vector<std::vector<double>> ranges{{1e16, std::nextafter(1e16, largest)},
                                                {0, smallest},
                                                {-largest, largest},
                                                {largest, largest}};
  for (const auto &range : ranges) {
    SCOPED_TRACE(range[0]);
    SCOPED_TRACE(range[1]);
    std::vector<double> values(12);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = range[i % 2];
    for (int64_t bins : {3, 10, 257}) {
      SCOPED_TRACE(bins);
      const auto stats = ComputeTensorStatistics(Tensor::FromDouble("X", {12}, values), bins);
      std::vector<int64_t> expected(static_cast<size_t>(bins));
      for (double value : values)
        for (size_t bin = 0; bin < expected.size(); ++bin)
          if (value >= stats.bin_edges[bin] &&
              (value < stats.bin_edges[bin + 1] ||
               (bin == expected.size() - 1 && value <= stats.bin_edges.back()))) {
            ++expected[bin];
            break;
          }
      EXPECT_EQ(stats.distribution, expected);
      EXPECT_EQ(std::accumulate(expected.begin(), expected.end(), int64_t{0}), 12);
    }
  }
}

TEST(TensorStatistics, ParallelStatisticsMatchSerialExecutor) {
  const int64_t count = 100003;
  std::vector<double> values(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i)
    values[static_cast<size_t>(i)] = static_cast<double>(i % 4);
  const auto tensor = Tensor::FromDouble("X", {count}, values);
  CpuExecutorRegistry registry(2);
  TensorStatistics serial;
  for (int32_t threads : {1, 3}) {
    SCOPED_TRACE(threads);
    CpuExecutionPolicy policy;
    policy.num_threads = threads;
    policy.affinity_policy = CpuAffinityPolicy::kNone;
    const auto executor = registry.Acquire(policy);
    executor->EnableCounters();
    CpuExecutorScope scope(executor.get());
    const auto stats = ComputeTensorStatistics(tensor, 4, {0, 0.25, 0.5, 1}, true, {1, count});
    EXPECT_DOUBLE_EQ(stats.min, 0);
    EXPECT_DOUBLE_EQ(stats.max, 3);
    EXPECT_NEAR(stats.mean, 150003.0 / count, 1e-15);
    EXPECT_EQ(stats.distribution, std::vector<int64_t>({25001, 25001, 25001, 25000}));
    EXPECT_EQ(stats.rectangle_map, values);
    if (threads == 1) {
      serial = stats;
    } else {
      EXPECT_EQ(stats.mean, serial.mean);
      EXPECT_EQ(stats.bin_edges, serial.bin_edges);
      EXPECT_EQ(stats.quantiles, serial.quantiles);
      EXPECT_EQ(stats.median, serial.median);
      const auto counters = executor->counters();
      EXPECT_GT(counters.dispatches,
                counters.limited_inline_dispatches + counters.nested_inline_dispatches);
    }
    const auto many_bins = ComputeTensorStatistics(tensor, 65537);
    EXPECT_EQ(
        std::accumulate(many_bins.distribution.begin(), many_bins.distribution.end(), int64_t{0}),
        count);
  }
}

TEST(TensorStatistics, MixedAxisRectangleResize) {
  const int64_t count = 100000;
  const auto tensor =
      Tensor::FromDouble("X", {count, 1}, std::vector<double>(static_cast<size_t>(count), 2.5));
  const auto stats = ComputeTensorStatistics(tensor, 0, {}, false, {1, count});
  EXPECT_EQ(stats.rectangle_map.size(), static_cast<size_t>(count));
  for (double value : stats.rectangle_map)
    EXPECT_NEAR(value, 2.5, 1e-12);
}

TEST(TensorStatistics, MixedAxisRectanglePreservesLocalNonfiniteValues) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const int64_t count = 100000;
  std::vector<double> values(static_cast<size_t>(count * 4), 2.5);
  values[1] = inf;
  values[6] = -inf;
  values[11] = nan;
  const auto stats =
      ComputeTensorStatistics(Tensor::FromDouble("X", {count, 4}, values), 0, {}, false, {1, 8});
  ASSERT_EQ(stats.rectangle_map.size(), 8u);
  for (size_t i = 0; i < 8; ++i) {
    if (i < 2)
      EXPECT_DOUBLE_EQ(stats.rectangle_map[i], 2.5);
    else if (i < 4)
      EXPECT_EQ(stats.rectangle_map[i], inf);
    else if (i < 6)
      EXPECT_EQ(stats.rectangle_map[i], -inf);
    else
      EXPECT_TRUE(std::isnan(stats.rectangle_map[i]));
  }
  const auto transposed = ComputeTensorStatistics(
      Tensor::FromDouble("X", {1, count}, std::vector<double>(static_cast<size_t>(count), 2.5)), 0,
      {}, false, {count, 1});
  for (double value : transposed.rectangle_map)
    EXPECT_DOUBLE_EQ(value, 2.5);
}

TEST(TensorStatistics, DirectReductionsMatchMaterializedStatistics) {
  const int64_t count = 100003;
  std::vector<int32_t> values(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i)
    values[static_cast<size_t>(i)] = static_cast<int32_t>(i % 257) - 128;
  const auto tensor = Tensor::FromInt32("X", {count}, values);
  const auto direct = ComputeTensorStatistics(tensor);
  const auto materialized = ComputeTensorStatistics(tensor, 0, {0.25, 0.75}, true);
  EXPECT_EQ(direct.min, -128);
  EXPECT_EQ(direct.max, 128);
  EXPECT_EQ(direct.distinct_count, 257);
  EXPECT_EQ(direct.min, materialized.min);
  EXPECT_EQ(direct.max, materialized.max);
  EXPECT_EQ(direct.mean, materialized.mean);
  EXPECT_EQ(direct.distinct_count, materialized.distinct_count);
  EXPECT_EQ(direct.distribution, materialized.distribution);
  EXPECT_EQ(direct.bin_edges, materialized.bin_edges);
}

TEST(TensorStatistics, ParallelChunkBoundariesPreserveNonfiniteAndPackedValues) {
  const int64_t count = 65539;
  CpuExecutionPolicy policy;
  policy.num_threads = 3;
  policy.affinity_policy = CpuAffinityPolicy::kNone;
  CpuExecutorRegistry registry(1);
  const auto executor = registry.Acquire(policy);
  CpuExecutorScope scope(executor.get());
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double exceptional : {inf, -inf, nan}) {
    std::vector<double> values(static_cast<size_t>(count), 1);
    values[32768] = exceptional;
    values.back() = exceptional;
    const auto stats = ComputeTensorStatistics(Tensor::FromDouble("X", {count}, values), 4, {0.5},
                                               true, {1, count});
    EXPECT_EQ(std::accumulate(stats.distribution.begin(), stats.distribution.end(), int64_t{0}),
              count - 2);
    if (std::isnan(exceptional)) {
      EXPECT_TRUE(std::isnan(stats.min));
      EXPECT_TRUE(std::isnan(stats.max));
      EXPECT_TRUE(std::isnan(stats.mean));
      EXPECT_TRUE(std::isnan(*stats.median));
      EXPECT_TRUE(std::isnan(stats.quantiles[0]));
      EXPECT_TRUE(std::isnan(stats.rectangle_map.back()));
    } else {
      EXPECT_EQ(stats.mean, exceptional);
      EXPECT_EQ(stats.rectangle_map.back(), exceptional);
    }
  }
  std::vector<double> values(static_cast<size_t>(count), 0);
  values[0] = inf;
  values.back() = -inf;
  const auto mixed = ComputeTensorStatistics(Tensor::FromDouble("X", {count}, values));
  EXPECT_TRUE(std::isnan(mixed.mean));
  for (DataType dtype : {DataType::INT2, DataType::UINT2, DataType::INT4, DataType::UINT4}) {
    const auto tensor =
        MakeSubByteTensor(dtype, {count}, std::vector<int8_t>(static_cast<size_t>(count), 1),
                          dtype == DataType::INT2 || dtype == DataType::UINT2 ? 2 : 4);
    const auto stats = ComputeTensorStatistics(tensor);
    EXPECT_EQ(stats.distinct_count, 1);
    EXPECT_EQ(stats.mean, 1);
  }
}

TEST(TensorStatistics, ExactIntegerDistinctAndUnalignedStorage) {
  const int64_t base = int64_t{1} << 60;
  auto stats =
      ComputeTensorStatistics(Tensor::FromInt64("X", {4}, {base, base + 1, base, base + 2}));
  EXPECT_EQ(stats.distinct_count, 3);
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  const auto tensor = Tensor::FromUint64("X", {3}, {maximum, maximum - 1, maximum});
  stats = ComputeTensorStatistics(tensor);
  EXPECT_EQ(stats.distinct_count, 2);
  std::vector<uint8_t> buffer(tensor.size_bytes() + 1);
  std::memcpy(buffer.data() + 1, tensor.bytes(), tensor.size_bytes());
  const auto borrowed =
      Tensor::Borrow("X", tensor.data_type, tensor.shape, buffer.data() + 1, tensor.size_bytes());
  EXPECT_EQ(ComputeTensorStatistics(borrowed).distinct_count, 2);
}

TEST(TensorStatistics, LargeIntegerDistinctCount) {
  std::vector<int32_t> values(100000);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<int32_t>(i % 257);
  EXPECT_EQ(ComputeTensorStatistics(Tensor::FromInt32("X", {100000}, values)).distinct_count, 257);
}

TEST(TensorStatistics, IntegerBooleanAndEmptyDistinctCounts) {
  for (const auto &tensor : {
           Tensor::FromInt8("X", {3}, {0, 1, 1}),
           Tensor::FromUint8("X", {3}, {0, 1, 1}),
           Tensor::FromInt16("X", {3}, {0, 1, 1}),
           Tensor::FromUint16("X", {3}, {0, 1, 1}),
           Tensor::FromInt32("X", {3}, {0, 1, 1}),
           Tensor::FromUint32("X", {3}, {0, 1, 1}),
           Tensor::FromBool("X", {3}, {0, 1, 1}),
       }) {
    const auto stats = ComputeTensorStatistics(tensor);
    EXPECT_EQ(stats.distinct_count, 2);
    EXPECT_DOUBLE_EQ(stats.mean, 2.0 / 3);
  }
  EXPECT_EQ(ComputeTensorStatistics(Tensor::FromInt64("X", {0}, {})).distinct_count, 0);
}

TEST(TensorStatistics, QuantilesAndMedian) {
  const auto stats = ComputeTensorStatistics(Tensor::FromInt32("X", {4}, {30, 0, 20, 10}), 0,
                                             {1, 0, 0.25, 0.5, 0.75}, true);
  EXPECT_EQ(stats.quantiles, std::vector<double>({30, 0, 7.5, 15, 22.5}));
  ASSERT_TRUE(stats.median);
  EXPECT_DOUBLE_EQ(*stats.median, 15);
  const auto scalar =
      ComputeTensorStatistics(Tensor::FromDouble("S", {}, {7}), 0, {0, 0.5, 1}, true);
  EXPECT_EQ(scalar.quantiles, std::vector<double>({7, 7, 7}));
  EXPECT_DOUBLE_EQ(*scalar.median, 7);
}

TEST(TensorStatistics, SelectedQuantilesMatchFullSortAndPreserveInput) {
  const std::vector<double> quantiles{
      1, 0.5, 0, 0.25, 0.75, 0.5, 0.1234567, std::nextafter(1.0, 0.0), 0.001};
  for (int64_t count : {1, 2, 3, 31, 1025, 65539}) {
    SCOPED_TRACE(count);
    std::vector<float> values(static_cast<size_t>(count));
    uint32_t state = 42;
    for (float &value : values) {
      state = state * 1664525u + 1013904223u;
      value = static_cast<float>(static_cast<int32_t>(state % 257) - 128) / 7;
    }
    const auto tensor = Tensor::FromFloat("X", {count}, values);
    const auto stats = ComputeTensorStatistics(tensor, 0, quantiles, true, {1, count});
    EXPECT_EQ(std::memcmp(tensor.bytes(), values.data(), values.size() * sizeof(float)), 0);
    EXPECT_EQ(stats.rectangle_map, std::vector<double>(values.begin(), values.end()));
    std::vector<long double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < quantiles.size(); ++i) {
      const long double position = static_cast<long double>(quantiles[i]) * (count - 1);
      const size_t lower = static_cast<size_t>(position);
      const size_t upper = std::min(lower + 1, sorted.size() - 1);
      EXPECT_DOUBLE_EQ(stats.quantiles[i], static_cast<double>(std::lerp(
                                               sorted[lower], sorted[upper], position - lower)));
    }
    EXPECT_EQ(stats.median, stats.quantiles[1]);
  }
}

TEST(TensorStatistics, SelectionPreservesWideIntegersAndDoublePrecision) {
  const uint64_t base = uint64_t{1} << 63;
  const uint64_t largest = std::numeric_limits<uint64_t>::max();
  const std::vector<uint64_t> values{largest, base + 4097, base + 1, base, base + 8193, base + 2};
  const auto tensor = Tensor::FromUint64("X", {6}, values);
  std::vector<uint8_t> unaligned(tensor.size_bytes() + 1);
  std::memcpy(unaligned.data() + 1, tensor.bytes(), tensor.size_bytes());
  const auto borrowed = Tensor::Borrow("X", tensor.data_type, tensor.shape, unaligned.data() + 1,
                                       tensor.size_bytes());
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const std::vector<double> quantiles{0, 0.2, 0.5, 0.6, 0.9, 1};
  const auto stats = ComputeTensorStatistics(borrowed, 0, quantiles, true);
  for (size_t i = 0; i < quantiles.size(); ++i) {
    const long double position = static_cast<long double>(quantiles[i]) * 5;
    const size_t lower = static_cast<size_t>(position);
    EXPECT_DOUBLE_EQ(stats.quantiles[i],
                     static_cast<double>(
                         std::lerp(static_cast<long double>(sorted[lower]),
                                   static_cast<long double>(sorted[std::min(lower + 1, size_t{5})]),
                                   position - lower)));
  }
  const int64_t small = std::numeric_limits<int64_t>::min();
  const int64_t large = std::numeric_limits<int64_t>::max();
  const auto signed_stats = ComputeTensorStatistics(
      Tensor::FromInt64("X", {4}, {large, small, small + 4097, large - 4097}), 0, {0, 0.5, 1},
      true);
  const double signed_median = static_cast<double>(std::lerp(
      static_cast<long double>(small + 4097), static_cast<long double>(large - 4097), 0.5L));
  EXPECT_EQ(signed_stats.quantiles, std::vector<double>({static_cast<double>(small), signed_median,
                                                         static_cast<double>(large)}));
  const double next = std::nextafter(1.0, 2.0);
  const auto precise =
      ComputeTensorStatistics(Tensor::FromDouble("X", {3}, {next, 1, next}), 0, {0, 0.5, 1}, true);
  EXPECT_EQ(precise.quantiles, std::vector<double>({1, next, next}));
}

TEST(TensorStatistics, LargeFloatHistogramQuantilesMatchFullSort) {
  const int64_t count = 65539;
  const float largest = std::numeric_limits<float>::max();
  const float smallest = std::numeric_limits<float>::denorm_min();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<std::vector<float>> palettes{
      {-largest, largest, -smallest, smallest, -0.0f, 0.0f, -inf, inf},
      {largest, std::nextafter(largest, 0.0f)},
      {-smallest, -0.0f, 0.0f, smallest},
      {-inf, inf},
      {2.5f}};
  const std::vector<double> quantiles{
      0, 0.00001, 0.1, 0.25, 0.5, 0.75, 0.9, std::nextafter(1.0, 0.0), 1};
  for (size_t dataset = 0; dataset <= palettes.size(); ++dataset) {
    SCOPED_TRACE(dataset);
    std::vector<float> values(static_cast<size_t>(count));
    uint32_t state = 42;
    for (size_t i = 0; i < values.size(); ++i) {
      state = state * 1664525u + 1013904223u;
      values[i] = dataset < palettes.size()
                      ? palettes[dataset][state % palettes[dataset].size()]
                      : std::bit_cast<float>((state & 0x807fffffu) | ((state >> 24) % 255) << 23);
    }
    const auto tensor = Tensor::FromFloat("X", {count}, values);
    std::vector<long double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    for (int64_t bins : {1, 3, 50, 257, 65537}) {
      SCOPED_TRACE(bins);
      const auto stats = ComputeTensorStatistics(tensor, bins, quantiles, true);
      for (size_t i = 0; i < quantiles.size(); ++i) {
        const long double position = static_cast<long double>(quantiles[i]) * (count - 1);
        const size_t lower = static_cast<size_t>(position);
        const long double weight = position - lower;
        const long double low = sorted[lower],
                          high = sorted[std::min(lower + 1, sorted.size() - 1)];
        const double expected = static_cast<double>(weight == 0 || low == high ? low
                                                    : std::isfinite(low) && std::isfinite(high)
                                                        ? std::lerp(low, high, weight)
                                                        : (1 - weight) * low + weight * high);
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(stats.quantiles[i]));
        } else {
          EXPECT_DOUBLE_EQ(stats.quantiles[i], expected);
        }
      }
      EXPECT_EQ(stats.median, stats.quantiles[4]);
    }
    EXPECT_EQ(std::memcmp(tensor.bytes(), values.data(), values.size() * sizeof(float)), 0);
  }
}

TEST(TensorStatistics, LargeFloatHistogramCancellationAndManyRanksFallback) {
  const int64_t count = 65539;
  std::vector<float> values(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i)
    values[static_cast<size_t>(i)] = static_cast<float>((i * 197) % count);
  const auto tensor = Tensor::FromFloat("X", {count}, values);
  for (int64_t stop : {0, 1, 2, 3}) {
    int64_t completed_passes = -1;
    const TensorStatisticsProgressCallback cancel = [&](TensorStatisticsPhase phase,
                                                        int64_t completed, int64_t total) {
      if (phase == TensorStatisticsPhase::kQuantiles) {
        EXPECT_EQ(total, 3);
        EXPECT_EQ(completed, completed_passes + 1);
        completed_passes = completed;
        return completed != stop;
      }
      EXPECT_NE(phase, TensorStatisticsPhase::kComplete);
      return true;
    };
    EXPECT_THROW(ComputeTensorStatistics(tensor, 0, {0.25, 0.5, 0.75}, true, {},
                                         kTensorStatisticsMaxOutputBytes, cancel),
                 std::runtime_error);
    EXPECT_EQ(completed_passes, stop);
  }
  std::vector<double> quantiles(130);
  for (size_t i = 0; i < quantiles.size(); ++i)
    quantiles[i] = static_cast<double>(i) / (quantiles.size() - 1);
  const auto stats = ComputeTensorStatistics(tensor, 0, quantiles, true);
  for (size_t i = 0; i < quantiles.size(); ++i)
    EXPECT_DOUBLE_EQ(stats.quantiles[i],
                     static_cast<double>(static_cast<long double>(quantiles[i]) * (count - 1)));
  EXPECT_EQ(std::memcmp(tensor.bytes(), values.data(), values.size() * sizeof(float)), 0);
}

TEST(TensorStatistics, SelectionHandlesRepeatedInfinitiesAndMedianOnly) {
  const double inf = std::numeric_limits<double>::infinity();
  const auto tensor = Tensor::FromDouble("X", {6}, {inf, -inf, 2, inf, -inf, 2});
  const auto stats = ComputeTensorStatistics(tensor, 0, {0, 0.1, 0.2, 0.5, 0.8, 0.9, 1}, true);
  EXPECT_EQ(stats.quantiles, std::vector<double>({-inf, -inf, -inf, 2, inf, inf, inf}));
  EXPECT_EQ(stats.median, 2);
  EXPECT_EQ(ComputeTensorStatistics(tensor, 0, {}, true).median, 2);
  const auto mixed = ComputeTensorStatistics(Tensor::FromDouble("X", {2}, {inf, -inf}), 0,
                                             {0, 0.25, 0.5, 0.75, 1}, true);
  EXPECT_EQ(mixed.quantiles.front(), -inf);
  EXPECT_EQ(mixed.quantiles.back(), inf);
  for (size_t i = 1; i < 4; ++i)
    EXPECT_TRUE(std::isnan(mixed.quantiles[i]));
  EXPECT_TRUE(std::isnan(*mixed.median));
}

TEST(TensorStatistics, ProgressPhasesAndCancellationAcrossOverloads) {
  const auto tensor = Tensor::FromFloat("X", {12}, {9, 3, 6, 2, 5, 8, 1, 4, 7, 0, 10, 11});
  const auto encoded = QuantizeTensor(tensor, MakeQuantizationPlan(QuantizationFormat::kInt8, 12));
  const auto caller = std::this_thread::get_id();
  const auto check = [&](const auto &value, bool expect_decode) {
    const auto compute = [&](const TensorStatisticsProgressCallback &progress) {
      if constexpr (std::is_same_v<std::decay_t<decltype(value)>, EncodedValueProto>)
        return ComputeTensorStatistics(value, 3, {0.25, 0.5, 0.75, 0.5}, true, {2, 2}, {},
                                       kTensorStatisticsMaxOutputBytes, progress);
      else
        return ComputeTensorStatistics(value, 3, {0.25, 0.5, 0.75, 0.5}, true, {2, 2},
                                       kTensorStatisticsMaxOutputBytes, progress);
    };
    std::vector<TensorStatisticsPhase> phases;
    int64_t previous = -1;
    const TensorStatisticsProgressCallback progress = [&](TensorStatisticsPhase phase,
                                                          int64_t completed, int64_t total) {
      EXPECT_EQ(std::this_thread::get_id(), caller);
      EXPECT_GT(total, 0);
      EXPECT_GE(completed, 0);
      EXPECT_LE(completed, total);
      if (phases.empty() || phases.back() != phase) {
        if (phase != TensorStatisticsPhase::kComplete) {
          EXPECT_EQ(completed, 0);
        }
        phases.push_back(phase);
        previous = -1;
      }
      EXPECT_GT(completed, previous);
      previous = completed;
      return true;
    };
    const auto stats = compute(progress);
    const auto plain = ComputeTensorStatistics(value, 3, {0.25, 0.5, 0.75, 0.5}, true, {2, 2});
    EXPECT_EQ(stats.quantiles, plain.quantiles);
    EXPECT_EQ(stats.rectangle_map, plain.rectangle_map);
    EXPECT_EQ(stats.distribution, plain.distribution);
    EXPECT_EQ(stats.mean, plain.mean);
    EXPECT_EQ(phases.front(),
              expect_decode ? TensorStatisticsPhase::kDecode : TensorStatisticsPhase::kSummary);
    EXPECT_EQ(phases.back(), TensorStatisticsPhase::kComplete);
    for (auto cancel_phase : phases) {
      bool cancelled = false;
      const TensorStatisticsProgressCallback cancel = [&](TensorStatisticsPhase phase, int64_t,
                                                          int64_t) {
        EXPECT_FALSE(cancelled);
        cancelled = phase == cancel_phase;
        return !cancelled;
      };
      EXPECT_THROW(compute(cancel), std::runtime_error);
      EXPECT_TRUE(cancelled);
    }
  };
  check(tensor, false);
  check(RuntimeValue(tensor), false);
  check(encoded, true);
  check(encoded.Encoded(), true);
}

TEST(TensorStatistics, CancelsBetweenSelectionsAndPropagatesCallbackErrors) {
  const auto tensor = Tensor::FromFloat("X", {10}, {9, 1, 8, 2, 7, 3, 6, 4, 5, 0});
  int64_t selected = 0;
  const TensorStatisticsProgressCallback cancel = [&](TensorStatisticsPhase phase,
                                                      int64_t completed, int64_t total) {
    if (phase == TensorStatisticsPhase::kQuantiles) {
      EXPECT_EQ(total, 6); // Q2 and median share the same two ranks.
      selected = completed;
      return completed == 0;
    }
    EXPECT_NE(phase, TensorStatisticsPhase::kComplete);
    return true;
  };
  EXPECT_THROW(ComputeTensorStatistics(tensor, 0, {0.25, 0.5, 0.75, 0.5}, true, {},
                                       kTensorStatisticsMaxOutputBytes, cancel),
               std::runtime_error);
  EXPECT_EQ(selected, 1);
  const TensorStatisticsProgressCallback fail = [](TensorStatisticsPhase, int64_t,
                                                   int64_t) -> bool {
    throw std::logic_error("callback failure");
  };
  EXPECT_THROW(
      ComputeTensorStatistics(tensor, 0, {}, false, {}, kTensorStatisticsMaxOutputBytes, fail),
      std::logic_error);
  EXPECT_THROW(
      ComputeTensorStatistics(tensor, -1, {}, false, {}, kTensorStatisticsMaxOutputBytes, fail),
      std::invalid_argument);
  const Tensor malformed;
  EXPECT_THROW(
      ComputeTensorStatistics(malformed, 0, {}, false, {}, kTensorStatisticsMaxOutputBytes, fail),
      std::invalid_argument);
  const auto empty = Tensor::FromFloat("X", {0}, {});
  std::vector<TensorStatisticsPhase> phases;
  ComputeTensorStatistics(empty, 0, {}, false, {}, kTensorStatisticsMaxOutputBytes,
                          [&](TensorStatisticsPhase phase, int64_t completed, int64_t total) {
                            phases.push_back(phase);
                            EXPECT_EQ(completed, 1);
                            EXPECT_EQ(total, 1);
                            return true;
                          });
  EXPECT_EQ(phases, std::vector<TensorStatisticsPhase>{TensorStatisticsPhase::kComplete});
  EXPECT_EQ(ComputeTensorStatistics(tensor, 0, {}, true).median, 4.5);
}

TEST(TensorStatistics, AreaAverageRectangle) {
  const auto tensor = Tensor::FromDouble("X", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto stats = ComputeTensorStatistics(tensor, 0, {}, false, {1, 2});
  EXPECT_EQ(stats.rectangle_shape, std::vector<int64_t>({1, 2}));
  EXPECT_EQ(stats.rectangle_map, std::vector<double>({2.5, 4.5}));
  stats = ComputeTensorStatistics(Tensor::FromDouble("V", {3}, {0, 6, 12}), 0, {}, false, {2, 2});
  ASSERT_EQ(stats.rectangle_map.size(), 4u);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(stats.rectangle_map[i], i % 2 == 0 ? 2 : 10, 1e-12);
  stats = ComputeTensorStatistics(Tensor::FromDouble("S", {}, {7}), 0, {}, false, {2, 3});
  EXPECT_EQ(stats.rectangle_map, std::vector<double>(6, 7));
}

TEST(TensorStatistics, MixedAxisRectangleResizeIsSeparable) {
  std::vector<double> values(100000);
  std::iota(values.begin(), values.end(), 0.0);
  const auto stats = ComputeTensorStatistics(Tensor::FromDouble("X", {100000, 1}, values), 0, {},
                                             false, {1, 100000});
  ASSERT_EQ(stats.rectangle_map.size(), values.size());
  for (double value : stats.rectangle_map)
    EXPECT_DOUBLE_EQ(value, 49999.5);
}

TEST(TensorStatistics, RectangleScalesOnlyLocalContributors) {
  const double largest = std::numeric_limits<double>::max();
  const double smallest = std::numeric_limits<double>::min();
  const auto identity = ComputeTensorStatistics(Tensor::FromDouble("X", {2}, {largest, smallest}),
                                                0, {}, false, {1, 2});
  EXPECT_EQ(identity.rectangle_map, std::vector<double>({largest, smallest}));
  const auto mixed = ComputeTensorStatistics(
      Tensor::FromDouble("X", {2, 2}, {largest, 0, smallest, smallest}), 0, {}, false, {2, 1});
  EXPECT_EQ(mixed.rectangle_map, std::vector<double>({largest / 2, smallest}));
  const auto signed_identity = ComputeTensorStatistics(
      Tensor::FromDouble("X", {4}, {largest, smallest, -largest, -smallest}), 0, {}, false, {1, 4});
  EXPECT_EQ(signed_identity.rectangle_map,
            std::vector<double>({largest, smallest, -largest, -smallest}));
}

TEST(TensorStatistics, EmptyAndInvalidOptions) {
  const auto empty = Tensor::FromFloat("X", {0, 2}, {});
  const auto stats = ComputeTensorStatistics(empty, 0, {0, 0.5, 1}, true, {2, 2});
  EXPECT_EQ(stats.count, 0);
  EXPECT_FALSE(stats.distinct_count);
  EXPECT_TRUE(std::isnan(stats.min));
  EXPECT_TRUE(std::isnan(stats.max));
  EXPECT_TRUE(std::isnan(stats.mean));
  ASSERT_EQ(stats.quantiles.size(), 3u);
  for (double value : stats.quantiles)
    EXPECT_TRUE(std::isnan(value));
  ASSERT_TRUE(stats.median);
  EXPECT_TRUE(std::isnan(*stats.median));
  EXPECT_TRUE(stats.rectangle_shape.empty());
  EXPECT_TRUE(stats.rectangle_map.empty());
  EXPECT_TRUE(stats.distribution.empty());
  EXPECT_TRUE(stats.bin_edges.empty());
  EXPECT_THROW(ComputeTensorStatistics(empty, -1), std::invalid_argument);
  EXPECT_THROW(ComputeTensorStatistics(empty, std::numeric_limits<int64_t>::max()),
               std::invalid_argument);
  for (double q : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::infinity()})
    EXPECT_THROW(ComputeTensorStatistics(empty, 0, {q}), std::invalid_argument);
  for (const std::vector<int64_t> &shape : {std::vector<int64_t>{2},
                                            {0, 1},
                                            {-1, 1},
                                            {1, 2, 3},
                                            {std::numeric_limits<int64_t>::max(), 2},
                                            {std::numeric_limits<int64_t>::max() / 2, 1}})
    EXPECT_THROW(ComputeTensorStatistics(empty, 0, {}, false, shape), std::invalid_argument);
}

TEST(TensorStatistics, OutputAllocationBudgetBoundaries) {
  const auto tensor = Tensor::FromFloat("X", {10}, std::vector<float>(10, 2));
  const auto encoded = QuantizeTensor(tensor, MakeQuantizationPlan(QuantizationFormat::kInt8, 10));
  constexpr int64_t histogram_bins = 2;
  constexpr int64_t rectangle_rows = 1;
  constexpr int64_t rectangle_columns = 2;
  constexpr size_t quantile_count = 1;
  const std::vector<double> quantiles(quantile_count, 0.5);
  const std::vector<int64_t> rectangle_shape{rectangle_rows, rectangle_columns};
  constexpr size_t histogram_counts_bytes = histogram_bins * sizeof(int64_t);
  constexpr size_t histogram_edges_bytes = (histogram_bins + 1) * sizeof(double);
  constexpr size_t rectangle_cells = rectangle_rows * rectangle_columns;
  constexpr size_t rectangle_cells_bytes = rectangle_cells * sizeof(double);
  constexpr size_t quantiles_bytes = quantile_count * sizeof(double);
  constexpr size_t median_bytes = sizeof(double);
  constexpr size_t combined_bytes = histogram_counts_bytes + histogram_edges_bytes +
                                    rectangle_cells_bytes + quantiles_bytes + median_bytes;
  for (const auto &value : {RuntimeValue(tensor), encoded}) {
    const auto stats = ComputeTensorStatistics(value, histogram_bins, quantiles, true,
                                               rectangle_shape, combined_bytes);
    EXPECT_EQ(stats.distribution.size(), histogram_bins);
    EXPECT_EQ(stats.bin_edges.size(), histogram_bins + 1);
    EXPECT_EQ(stats.rectangle_map.size(), rectangle_cells);
    EXPECT_EQ(stats.quantiles.size(), quantile_count);
    EXPECT_TRUE(stats.median.has_value());
    EXPECT_THROW(ComputeTensorStatistics(value, histogram_bins, quantiles, true, rectangle_shape,
                                         combined_bytes - 1),
                 std::invalid_argument);
  }
  EXPECT_NO_THROW(ComputeTensorStatistics(tensor, histogram_bins, quantiles, true, rectangle_shape,
                                          combined_bytes));
  EXPECT_THROW(ComputeTensorStatistics(tensor, histogram_bins, quantiles, true, rectangle_shape,
                                       combined_bytes - 1),
               std::invalid_argument);
  EXPECT_NO_THROW(ComputeTensorStatistics(encoded.Encoded(), histogram_bins, quantiles, true,
                                          rectangle_shape, {}, combined_bytes));
  EXPECT_THROW(ComputeTensorStatistics(encoded.Encoded(), histogram_bins, quantiles, true,
                                       rectangle_shape, {}, combined_bytes - 1),
               std::invalid_argument);

  constexpr size_t automatic_bins = 50;
  constexpr size_t automatic_histogram_bytes =
      automatic_bins * sizeof(int64_t) + (automatic_bins + 1) * sizeof(double);
  EXPECT_NO_THROW(ComputeTensorStatistics(tensor, 0, {}, false, {}, automatic_histogram_bytes));
  EXPECT_THROW(ComputeTensorStatistics(tensor, 0, {}, false, {}, automatic_histogram_bytes - 1),
               std::invalid_argument);
  EXPECT_THROW(ComputeTensorStatistics(tensor, 1, {}, false, {}, 0), std::invalid_argument);
  constexpr int64_t three_bins = 3;
  constexpr size_t three_bin_histogram_bytes =
      three_bins * sizeof(int64_t) + (three_bins + 1) * sizeof(double);
  EXPECT_NO_THROW(
      ComputeTensorStatistics(tensor, three_bins, {}, false, {}, three_bin_histogram_bytes));
  EXPECT_THROW(
      ComputeTensorStatistics(tensor, three_bins, {}, false, {}, three_bin_histogram_bytes - 1),
      std::invalid_argument);
  constexpr int64_t one_bin = 1;
  constexpr int64_t square_side = 2;
  constexpr size_t one_bin_histogram_bytes =
      one_bin * sizeof(int64_t) + (one_bin + 1) * sizeof(double);
  constexpr size_t histogram_rectangle_bytes =
      one_bin_histogram_bytes + square_side * square_side * sizeof(double);
  EXPECT_NO_THROW(ComputeTensorStatistics(tensor, one_bin, {}, false, {square_side, square_side},
                                          histogram_rectangle_bytes));
  EXPECT_THROW(ComputeTensorStatistics(tensor, one_bin, {}, false, {square_side, square_side},
                                       histogram_rectangle_bytes - 1),
               std::invalid_argument);

  const auto empty = Tensor::FromFloat("X", {0}, {});
  const int64_t maximum_bins =
      (kTensorStatisticsMaxOutputBytes - sizeof(double)) / (sizeof(int64_t) + sizeof(double));
  const int64_t maximum_cells =
      (kTensorStatisticsMaxOutputBytes - one_bin_histogram_bytes) / sizeof(double);
  EXPECT_NO_THROW(ComputeTensorStatistics(empty, maximum_bins));
  EXPECT_THROW(ComputeTensorStatistics(empty, maximum_bins + 1), std::invalid_argument);
  constexpr size_t additional_bin_bytes = sizeof(int64_t) + sizeof(double);
  EXPECT_NO_THROW(ComputeTensorStatistics(empty, maximum_bins + 1, {}, false, {},
                                          kTensorStatisticsMaxOutputBytes + additional_bin_bytes));
  EXPECT_NO_THROW(ComputeTensorStatistics(empty, one_bin, {}, false, {1, maximum_cells}));
  EXPECT_THROW(ComputeTensorStatistics(empty, one_bin, {}, false, {1, maximum_cells + 1}),
               std::invalid_argument);
  EXPECT_NO_THROW(ValidateTensorStatisticsOptions(maximum_bins, {}, false, {1, maximum_cells},
                                                  std::numeric_limits<size_t>::max()));
  EXPECT_THROW(ValidateTensorStatisticsOptions(std::numeric_limits<int64_t>::max(), {}, false, {},
                                               std::numeric_limits<size_t>::max()),
               std::invalid_argument);
  EXPECT_THROW(ValidateTensorStatisticsOptions(1, {}, false,
                                               {std::numeric_limits<int64_t>::max(), 2},
                                               std::numeric_limits<size_t>::max()),
               std::invalid_argument);
}

TEST(TensorStatistics, RectangleCellCountValidation) {
  const size_t maximum_budget = std::numeric_limits<size_t>::max();
  const int64_t maximum_dimension = std::numeric_limits<int64_t>::max();
  for (const auto &shape : std::vector<std::vector<int64_t>>{{1},
                                                             {1, 2, 3},
                                                             {0, 1},
                                                             {1, 0},
                                                             {-1, 1},
                                                             {1, -1},
                                                             {-1, -1},
                                                             {maximum_dimension, 2},
                                                             {2, maximum_dimension},
                                                             {maximum_dimension, maximum_dimension},
                                                             {maximum_dimension, 1}}) {
    SCOPED_TRACE(shape.front());
    EXPECT_THROW(ValidateTensorStatisticsOptions(1, {}, false, shape, maximum_budget),
                 std::invalid_argument);
  }
  const int64_t large_cell_count = static_cast<int64_t>(std::vector<double>().max_size() / 2);
  EXPECT_NO_THROW(
      ValidateTensorStatisticsOptions(1, {}, false, {1, large_cell_count}, maximum_budget));
  const size_t exact_budget = sizeof(int64_t) + 2 * sizeof(double) + 6 * sizeof(double);
  for (const auto &shape : std::vector<std::vector<int64_t>>{{2, 3}, {3, 2}}) {
    SCOPED_TRACE(shape.front());
    EXPECT_NO_THROW(ValidateTensorStatisticsOptions(1, {}, false, shape, exact_budget));
    EXPECT_THROW(ValidateTensorStatisticsOptions(1, {}, false, shape, exact_budget - 1),
                 std::invalid_argument);
  }
}

TEST(TensorStatistics, RejectsRectangleOverflowBeforeReadingOrDecodingInput) {
  const auto check = [](const auto &value) {
    try {
      ComputeTensorStatistics(value, 1, {}, false, {std::numeric_limits<int64_t>::max(), 2});
      FAIL() << "Expected rectangle overflow rejection";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find("rectangle size overflows"), std::string::npos);
    }
  };
  const Tensor malformed;
  const EncodedValueProto undecodable;
  RuntimeValue encoded;
  encoded.kind = RuntimeValue::Kind::kEncoded;
  check(malformed);
  check(RuntimeValue(malformed));
  check(encoded);
  check(undecodable);
}

TEST(TensorStatistics, RejectsOutputBudgetBeforeReadingOrDecodingInput) {
  const auto check = [](const auto &value, int64_t bins, const std::vector<int64_t> &shape) {
    try {
      ComputeTensorStatistics(value, bins, {}, false, shape);
      FAIL() << "Expected allocation budget rejection";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find("max_output_bytes"), std::string::npos);
    }
  };
  const auto empty = Tensor::FromFloat("X", {0}, {});
  Tensor malformed;
  EncodedValueProto undecodable;
  RuntimeValue encoded;
  encoded.kind = RuntimeValue::Kind::kEncoded;
  // Exceed the output budget without hitting 32-bit vector capacity limits first.
  constexpr int64_t oversized_count = kTensorStatisticsMaxOutputBytes / sizeof(double) + 1;
  for (const auto &options :
       {std::make_pair(oversized_count, std::vector<int64_t>{}),
        std::make_pair(int64_t{1}, std::vector<int64_t>{1, oversized_count})}) {
    check(empty, options.first, options.second);
    check(malformed, options.first, options.second);
    check(RuntimeValue(empty), options.first, options.second);
    check(RuntimeValue(malformed), options.first, options.second);
    check(encoded, options.first, options.second);
    check(undecodable, options.first, options.second);
  }
}

TEST(TensorStatistics, MalformedAndUnsupportedTensors) {
  auto tensor = Tensor::FromFloat("X", {1}, {2});
  tensor.shape = {2};
  EXPECT_THROW(ComputeTensorStatistics(tensor), std::invalid_argument);
  tensor.shape = {-1};
  EXPECT_THROW(ComputeTensorStatistics(tensor), std::invalid_argument);
  tensor.shape = {std::numeric_limits<int64_t>::max(), 2};
  EXPECT_THROW(ComputeTensorStatistics(tensor), std::invalid_argument);
  tensor.shape = {0, -1};
  EXPECT_THROW(ComputeTensorStatistics(tensor), std::invalid_argument);
  tensor.shape = {0};
  for (int32_t dtype : std::vector<int32_t>{TensorProto::STRING, TensorProto::COMPLEX64,
                                            TensorProto::COMPLEX128, TensorProto::UNDEFINED, 999}) {
    tensor.data_type = dtype;
    EXPECT_THROW(ComputeTensorStatistics(tensor), std::invalid_argument);
  }
  EXPECT_THROW(ComputeTensorStatistics(RuntimeValue()), std::invalid_argument);
  RuntimeValue encoded;
  encoded.kind = RuntimeValue::Kind::kEncoded;
  EXPECT_THROW(ComputeTensorStatistics(encoded), std::invalid_argument);
}

TEST(TensorStatistics, DeterministicNonfiniteValues) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const auto tensor = Tensor::FromDouble("X", {10}, {nan, nan, -inf, inf, 0, -0., 1, 2, 3, 4});
  auto stats = ComputeTensorStatistics(tensor, 2, {0, 0.5, 1}, true, {1, 1});
  EXPECT_TRUE(std::isnan(stats.min));
  EXPECT_TRUE(std::isnan(stats.max));
  EXPECT_TRUE(std::isnan(stats.mean));
  EXPECT_FALSE(stats.distinct_count);
  EXPECT_EQ(stats.distribution, std::vector<int64_t>({3, 3}));
  EXPECT_EQ(stats.bin_edges, std::vector<double>({0, 2, 4}));
  for (double value : stats.quantiles)
    EXPECT_TRUE(std::isnan(value));
  EXPECT_TRUE(std::isnan(*stats.median));
  EXPECT_TRUE(std::isnan(stats.rectangle_map[0]));
  stats =
      ComputeTensorStatistics(Tensor::FromDouble("X", {3}, {-inf, 0, inf}), 0, {0, 0.5, 1}, true);
  EXPECT_EQ(stats.min, -inf);
  EXPECT_EQ(stats.max, inf);
  EXPECT_TRUE(std::isnan(stats.mean));
  EXPECT_EQ(stats.quantiles, std::vector<double>({-inf, 0, inf}));
  EXPECT_DOUBLE_EQ(*stats.median, 0);
  stats = ComputeTensorStatistics(Tensor::FromDouble("X", {10}, std::vector<double>(10, inf)));
  EXPECT_TRUE(stats.distribution.empty());
  EXPECT_EQ(stats.mean, inf);
}

TEST(TensorStatistics, ExtremeFiniteValues) {
  const double largest = std::numeric_limits<double>::max();
  auto stats = ComputeTensorStatistics(
      Tensor::FromDouble("X", {10}, {-largest, largest, 0, 0, 0, 0, 0, 0, 0, 0}), 2, {0.5}, true);
  EXPECT_DOUBLE_EQ(stats.mean, 0);
  EXPECT_EQ(stats.distribution, std::vector<int64_t>({1, 9}));
  EXPECT_EQ(stats.bin_edges, std::vector<double>({-largest, 0, largest}));
  stats = ComputeTensorStatistics(Tensor::FromDouble("X", {10}, std::vector<double>(10, largest)));
  EXPECT_TRUE(std::isfinite(stats.mean));
  EXPECT_EQ(std::accumulate(stats.distribution.begin(), stats.distribution.end(), int64_t{0}), 10);
  for (int64_t count : {3, 100, 501, 65539}) {
    for (double value : {largest, -largest}) {
      stats = ComputeTensorStatistics(
          Tensor::FromDouble("X", {count}, std::vector<double>(static_cast<size_t>(count), value)),
          0, {0.1, 0.5, 0.9}, true, {1, 1});
      EXPECT_DOUBLE_EQ(stats.mean, value);
      ASSERT_EQ(stats.rectangle_map.size(), 1u);
      EXPECT_NEAR(stats.rectangle_map[0] / value, 1, 1e-12);
      for (double edge : stats.bin_edges)
        EXPECT_TRUE(std::isfinite(edge));
      for (double q : stats.quantiles)
        EXPECT_DOUBLE_EQ(q, value);
    }
  }
  stats = ComputeTensorStatistics(Tensor::FromDouble("X", {3}, {-largest, largest, largest}));
  EXPECT_NEAR(stats.mean / largest, 1.0 / 3, 1e-15);
}

TEST(TensorStatistics, PackedAndReducedPrecision) {
  for (const auto dtype : {DataType::INT2, DataType::UINT2, DataType::INT4, DataType::UINT4}) {
    const auto tensor = MakeSubByteTensor(
        dtype, {3}, {0, 1, 1}, dtype == DataType::INT2 || dtype == DataType::UINT2 ? 2 : 4);
    const auto stats = ComputeTensorStatistics(tensor, 0, {0.5}, true);
    EXPECT_EQ(stats.count, 3);
    EXPECT_EQ(stats.size_bytes, dtype == DataType::INT2 || dtype == DataType::UINT2 ? 1u : 2u);
    EXPECT_EQ(stats.distinct_count, 2);
    EXPECT_DOUBLE_EQ(stats.mean, 2.0 / 3);
    EXPECT_DOUBLE_EQ(*stats.median, 1);
  }
  EXPECT_EQ(ComputeTensorStatistics(MakeSubByteTensor(DataType::INT4, {3}, {-8, -1, 7}, 4)).min,
            -8);
  EXPECT_EQ(ComputeTensorStatistics(MakeSubByteTensor(DataType::INT2, {3}, {-2, -1, 1}, 2)).min,
            -2);
  const auto check = [](const Tensor &tensor) {
    auto stats = ComputeTensorStatistics(tensor);
    EXPECT_DOUBLE_EQ(stats.min, 0);
    EXPECT_DOUBLE_EQ(stats.max, 2);
    EXPECT_DOUBLE_EQ(stats.mean, 1);
    EXPECT_FALSE(stats.distinct_count);
  };
  check(MakeFloat16Tensor("X", {3}, {0, 1, 2}));
  check(MakeBfloat16Tensor("X", {3}, {0, 1, 2}));
  check(MakeFloat4E2M1Tensor({3}, {0, 1, 2}));
  check(MakeFloat8Tensor(DataType::FLOAT8E4M3FN, {3}, {0, 1, 2}, FloatToFloat8E4M3FNBits));
  check(MakeFloat8Tensor(DataType::FLOAT8E4M3FNUZ, {3}, {0, 1, 2}, FloatToFloat8E4M3FNUZBits));
  check(MakeFloat8Tensor(DataType::FLOAT8E5M2, {3}, {0, 1, 2}, FloatToFloat8E5M2Bits));
  check(MakeFloat8Tensor(DataType::FLOAT8E5M2FNUZ, {3}, {0, 1, 2}, FloatToFloat8E5M2FNUZBits));
  const auto exponent_only = ComputeTensorStatistics(
      MakeFloat8Tensor(DataType::FLOAT8E8M0, {3}, {0.5, 1, 2},
                       static_cast<uint8_t (*)(float) noexcept>(FloatToFloat8E8M0Bits)));
  EXPECT_DOUBLE_EQ(exponent_only.min, 0.5);
  EXPECT_DOUBLE_EQ(exponent_only.max, 2);
  EXPECT_DOUBLE_EQ(exponent_only.mean, 3.5 / 3);
  for (auto dtype : {DataType::FLOAT6E2M3, DataType::FLOAT6E3M2}) {
    auto tensor = MakeOutputTensor(static_cast<int32_t>(dtype), {3},
                                   PackedByteSize(int32_t(dtype), 3), nullptr);
    std::memset(tensor.mutable_bytes(), 0, tensor.size_bytes());
    for (int64_t i = 0; i < 3; ++i)
      Write6BitElement(tensor.mutable_bytes(), i, FloatToFloat6Bits(static_cast<float>(i), dtype));
    check(tensor);
  }
}

TEST(TensorStatistics, QuantizedAndRuntimeValuesPreserveStorageMetadata) {
  const auto source = Tensor::FromFloat("X", {2, 5}, {-4, -3, -2, -1, 0, 0, 1, 2, 3, 4});
  for (auto format : {QuantizationFormat::kInt4, QuantizationFormat::kInt8}) {
    const auto encoded = QuantizeTensor(source, MakeQuantizationPlan(format, 10, 5));
    const auto reconstructed = DequantizeTensor(encoded);
    const auto expected = ComputeTensorStatistics(reconstructed, 3, {0.25, 0.75}, true, {1, 2});
    for (auto stats : {ComputeTensorStatistics(encoded, 3, {0.25, 0.75}, true, {1, 2}),
                       ComputeTensorStatistics(encoded.Encoded(), 3, {0.25, 0.75}, true, {1, 2})}) {
      EXPECT_EQ(stats.dtype, source.data_type);
      EXPECT_EQ(stats.shape, std::vector<int64_t>({2, 5}));
      EXPECT_EQ(stats.size_bytes, encoded.Encoded().raw_data().size());
      EXPECT_EQ(stats.count, expected.count);
      EXPECT_EQ(stats.min, expected.min);
      EXPECT_EQ(stats.max, expected.max);
      EXPECT_EQ(stats.mean, expected.mean);
      const auto values = reconstructed.AsFloat();
      std::set<float> distinct(values, values + reconstructed.element_count());
      EXPECT_EQ(stats.distinct_count, static_cast<int64_t>(distinct.size()));
      EXPECT_EQ(stats.distribution, expected.distribution);
      EXPECT_EQ(stats.bin_edges, expected.bin_edges);
      EXPECT_EQ(stats.quantiles, expected.quantiles);
      EXPECT_EQ(stats.median, expected.median);
      EXPECT_EQ(stats.rectangle_map, expected.rectangle_map);
    }
  }
  EXPECT_EQ(ComputeTensorStatistics(RuntimeValue(source)).count, 10);
}

TEST(TensorStatistics, EncodedCatalogueLastOverload) {
  auto encoded = QuantizeTensor(Tensor::FromFloat("X", {3}, {-1, 0, 1}),
                                MakeQuantizationPlan(QuantizationFormat::kInt8, 3))
                     .Encoded();
  const auto expected = ComputeTensorStatistics(encoded, 0, {0.5}, true, {1, 1});
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(17);
  encoded.mutable_struct_type()->Clear();
  encoded.mutable_struct_type()->set_type_ref(17);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  EXPECT_THROW(ComputeTensorStatistics(encoded), std::invalid_argument);
  const auto stats = ComputeTensorStatistics(encoded, 0, {0.5}, true, {1, 1}, catalogue);
  EXPECT_EQ(stats.dtype, expected.dtype);
  EXPECT_EQ(stats.shape, expected.shape);
  EXPECT_EQ(stats.size_bytes, expected.size_bytes);
  EXPECT_EQ(stats.min, expected.min);
  EXPECT_EQ(stats.max, expected.max);
  EXPECT_EQ(stats.mean, expected.mean);
  EXPECT_EQ(stats.distinct_count, expected.distinct_count);
  EXPECT_EQ(stats.quantiles, expected.quantiles);
  EXPECT_EQ(stats.median, expected.median);
  EXPECT_EQ(stats.rectangle_map, expected.rectangle_map);
}

TEST(TensorStatistics, EncodedStatisticsRequireLogicalTensorType) {
  EncodedValueProto encoded;
  EXPECT_THROW(ComputeTensorStatistics(encoded), std::invalid_argument);
  encoded.mutable_logical_type()->mutable_sequence_type();
  EXPECT_THROW(ComputeTensorStatistics(encoded), std::invalid_argument);
}

TEST(TensorStatistics, EncodedValuesRequireLogicalTensorMetadata) {
  auto encoded = QuantizeTensor(Tensor::FromFloat("X", {3}, {-1, 0, 1}),
                                MakeQuantizationPlan(QuantizationFormat::kInt8, 3))
                     .Encoded();
  encoded.clear_logical_type();
  EXPECT_THROW(ComputeTensorStatistics(encoded), std::invalid_argument);
  encoded.mutable_logical_type()->mutable_sequence_type();
  EXPECT_THROW(ComputeTensorStatistics(encoded), std::invalid_argument);
}
