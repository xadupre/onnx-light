// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/tensor_statistics.h"

#include "onnx_core/runtime/kernels/cast_float8.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/cast_sub_byte.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/quantization.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_lib/common/safe_math.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

void ValidateTensorStatisticsOptions(int64_t bins, const std::vector<double> &quantiles,
                                     bool median, const std::vector<int64_t> &rectangle,
                                     size_t max_output_bytes) {
  EXT_ENFORCE_INVALID(bins >= 0, "Tensor statistics bins must be non-negative.");
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(bins) < std::vector<double>().max_size() &&
                          static_cast<uint64_t>(bins) <= std::vector<int64_t>().max_size(),
                      "Tensor statistics histogram allocation overflows.");
  for (double q : quantiles)
    EXT_ENFORCE_INVALID(std::isfinite(q) && q >= 0 && q <= 1,
                        "Tensor statistics quantiles must be finite and in [0,1].");
  int64_t rectangle_cells = 0;
  if (!rectangle.empty()) {
    EXT_ENFORCE_INVALID(rectangle.size() == 2 && rectangle[0] > 0 && rectangle[1] > 0,
                        "Tensor statistics rectangle must contain positive [height,width].");
    EXT_ENFORCE_INVALID(!checked_mul_overflow(rectangle[0], rectangle[1], &rectangle_cells),
                        "Tensor statistics rectangle size overflows.");
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(rectangle_cells) <= std::vector<double>().max_size(),
                        "Tensor statistics rectangle allocation overflows.");
  }
  size_t remaining = max_output_bytes;
  const auto charge = [&](uint64_t count, size_t item_size, const char *option) {
    EXT_ENFORCE_INVALID(count <= remaining / item_size, "Tensor statistics ", option,
                        " exceeds max_output_bytes allocation budget (", max_output_bytes,
                        " bytes).");
    remaining -= static_cast<size_t>(count) * item_size;
  };
  const uint64_t histogram_bins = bins == 0 ? 50 : static_cast<uint64_t>(bins);
  charge(histogram_bins, sizeof(int64_t), "bins");
  charge(histogram_bins + 1, sizeof(double), "bins");
  if (!rectangle.empty())
    charge(static_cast<uint64_t>(rectangle_cells), sizeof(double), "rectangle_shape");
  charge(quantiles.size(), sizeof(double), "quantiles");
  charge(median ? 1 : 0, sizeof(double), "median");
}

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void ReportProgress(const TensorStatisticsProgressCallback &progress, TensorStatisticsPhase phase,
                    int64_t completed, int64_t total = 1) {
  if (progress && !progress(phase, completed, total))
    throw std::runtime_error("Tensor statistics cancelled.");
}

template <typename F>
void StatisticsPhase(const TensorStatisticsProgressCallback &progress, TensorStatisticsPhase phase,
                     F &&compute) {
  ReportProgress(progress, phase, 0);
  compute();
  ReportProgress(progress, phase, 1);
}

template <typename T> T Load(const uint8_t *data, int64_t index) {
  T result;
  std::memcpy(&result, data + static_cast<size_t>(index) * sizeof(T), sizeof(T));
  return result;
}

long double ReadNumber(const Tensor &tensor, int64_t index) {
  const auto *data = tensor.bytes();
  switch (static_cast<DataType>(tensor.data_type)) {
  case DataType::FLOAT:
    return Load<float>(data, index);
  case DataType::DOUBLE:
    return Load<double>(data, index);
  case DataType::INT8:
    return Load<int8_t>(data, index);
  case DataType::UINT8:
    return data[index];
  case DataType::BOOL:
    return data[index] != 0;
  case DataType::INT16:
    return Load<int16_t>(data, index);
  case DataType::UINT16:
    return Load<uint16_t>(data, index);
  case DataType::INT32:
    return Load<int32_t>(data, index);
  case DataType::UINT32:
    return Load<uint32_t>(data, index);
  case DataType::INT64:
    return Load<int64_t>(data, index);
  case DataType::UINT64:
    return Load<uint64_t>(data, index);
  case DataType::FLOAT16:
    return Float16BitsToFloat(Load<uint16_t>(data, index));
  case DataType::BFLOAT16:
    return Bfloat16BitsToFloat(Load<uint16_t>(data, index));
  case DataType::FLOAT8E4M3FN:
    return Float8E4M3FNBitsToFloat(data[index]);
  case DataType::FLOAT8E4M3FNUZ:
    return Float8E4M3FNUZBitsToFloat(data[index]);
  case DataType::FLOAT8E5M2:
    return Float8E5M2BitsToFloat(data[index]);
  case DataType::FLOAT8E5M2FNUZ:
    return Float8E5M2FNUZBitsToFloat(data[index]);
  case DataType::FLOAT8E8M0:
    return Float8E8M0BitsToFloat(data[index]);
  case DataType::INT4:
    return Int4NibbleToInt8(Read4BitElement(data, index));
  case DataType::UINT4:
    return Read4BitElement(data, index);
  case DataType::INT2:
    return Int2BitsToInt8(Read2BitElement(data, index));
  case DataType::UINT2:
    return Read2BitElement(data, index);
  case DataType::FLOAT4E2M1:
    return Float4E2M1NibbleToFloat(Read4BitElement(data, index));
  case DataType::FLOAT6E2M3:
  case DataType::FLOAT6E3M2:
    return Float6BitsToFloat(Read6BitElement(data, index), static_cast<DataType>(tensor.data_type));
  default:
    EXT_THROW_INVALID("Tensor statistics requires a real numeric tensor.");
  }
}

template <typename T> int64_t DistinctCountFromRaw(const Tensor &tensor, int64_t count) {
  std::unordered_set<T> distinct;
  distinct.reserve(static_cast<size_t>(std::min<int64_t>(count, 256)));
  for (int64_t i = 0; i < count; ++i)
    distinct.insert(Load<T>(tensor.bytes(), i));
  return static_cast<int64_t>(distinct.size());
}

int64_t DistinctCount(const Tensor &tensor, bool has_nan) {
  const int64_t count = tensor.element_count();
  // Integer keys must not pass through floating-point conversion, including on platforms where
  // long double has only double precision. memcpy also permits unaligned borrowed tensor storage.
  if (tensor.data_type == TensorProto::INT64)
    return DistinctCountFromRaw<int64_t>(tensor, count);
  if (tensor.data_type == TensorProto::UINT64)
    return DistinctCountFromRaw<uint64_t>(tensor, count);
  std::unordered_set<long double> distinct;
  distinct.reserve(static_cast<size_t>(std::min<int64_t>(count, 256)));
  for (int64_t i = 0; i < count; ++i) {
    const long double value = ReadNumber(tensor, i);
    if (!std::isnan(value))
      distinct.insert(value);
  }
  return static_cast<int64_t>(distinct.size()) + (has_nan ? 1 : 0);
}

template <typename Reader> double Quantile(size_t count, double q, Reader &&read) {
  const long double position = static_cast<long double>(q) * (count - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, count - 1);
  const long double weight = position - lower;
  const long double low = read(lower);
  if (weight == 0)
    return static_cast<double>(low);
  const long double high = read(upper);
  if (low == high)
    return static_cast<double>(low);
  if (std::isfinite(low) && std::isfinite(high))
    return static_cast<double>(std::lerp(low, high, weight));
  return static_cast<double>((1 - weight) * low + weight * high);
}

template <typename T>
void SelectRanks(std::vector<T> &values, const std::vector<size_t> &ranks, size_t first,
                 size_t last, size_t begin, size_t end,
                 const TensorStatisticsProgressCallback &progress, int64_t &completed) {
  if (first == last)
    return;
  const size_t middle = first + (last - first) / 2;
  const size_t rank = ranks[middle];
  auto start = values.begin();
  std::nth_element(start + begin, start + rank, start + end);
  ReportProgress(progress, TensorStatisticsPhase::kQuantiles, ++completed,
                 static_cast<int64_t>(ranks.size()));
  SelectRanks(values, ranks, first, middle, begin, rank, progress, completed);
  SelectRanks(values, ranks, middle + 1, last, rank + 1, end, progress, completed);
}

std::vector<size_t> QuantileRanks(size_t count, const std::vector<double> &quantiles, bool median) {
  std::vector<size_t> ranks;
  EXT_ENFORCE_INVALID(quantiles.size() <= (ranks.max_size() - 2) / 2,
                      "Tensor statistics quantile ranks allocation overflows.");
  ranks.reserve(2 * quantiles.size() + (median ? 2 : 0));
  const auto add_ranks = [&](double q) {
    const long double position = static_cast<long double>(q) * (count - 1);
    const size_t lower = static_cast<size_t>(position);
    ranks.push_back(lower);
    if (position != lower)
      ranks.push_back(std::min(lower + 1, count - 1));
  };
  for (double q : quantiles)
    add_ranks(q);
  if (median)
    add_ranks(0.5);
  std::sort(ranks.begin(), ranks.end());
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
  return ranks;
}

template <typename Reader>
void SetQuantiles(TensorStatistics &result, const std::vector<double> &quantiles, bool median,
                  Reader &&read) {
  for (size_t i = 0; i < quantiles.size(); ++i)
    result.quantiles[i] = Quantile(static_cast<size_t>(result.count), quantiles[i], read);
  if (median)
    result.median = Quantile(static_cast<size_t>(result.count), 0.5, read);
}

template <typename T, typename Reader>
void ComputeQuantiles(TensorStatistics &result, const std::vector<double> &quantiles, bool median,
                      const std::vector<size_t> &ranks,
                      const TensorStatisticsProgressCallback &progress, Reader &&read) {
  ReportProgress(progress, TensorStatisticsPhase::kQuantiles, 0,
                 static_cast<int64_t>(ranks.size()));
  std::vector<T> values;
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(result.count) <= values.max_size(),
                      "Tensor statistics numeric allocation overflows.");
  values.resize(static_cast<size_t>(result.count));
  ParallelFor(result.count, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i)
      values[static_cast<size_t>(i)] = read(i);
  });
  int64_t completed = 0;
  SelectRanks(values, ranks, 0, ranks.size(), 0, values.size(), progress, completed);
  SetQuantiles(result, quantiles, median, [&](size_t i) { return values[i]; });
}

uint32_t FloatOrderKey(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

float FloatFromOrderKey(uint32_t key) {
  return std::bit_cast<float>(key & 0x80000000u ? key ^ 0x80000000u : ~key);
}

uint32_t FloatLowerKey(double edge) {
  float value = static_cast<float>(edge);
  if (static_cast<double>(value) < edge)
    value = std::nextafter(value, std::numeric_limits<float>::infinity());
  // Both signs of zero belong to the same exported histogram bin.
  return FloatOrderKey(value == 0 ? -0.0f : value);
}

void HistogramQuantiles(TensorStatistics &result, const Tensor &tensor,
                        const std::vector<double> &quantiles, bool median,
                        const std::vector<size_t> &ranks, int64_t negative_infinities,
                        const TensorStatisticsProgressCallback &progress) {
  struct Range {
    uint32_t low, high;
    bool operator==(const Range &) const = default;
  };
  std::vector<Range> ranges(ranks.size());
  std::vector<int64_t> offsets(ranks.size(), -1);
  std::vector<float> selected(ranks.size());
  const int64_t finite_count =
      std::accumulate(result.distribution.begin(), result.distribution.end(), int64_t{0});
  bool has_finite_rank = false;
  size_t initial_bin = 0;
  int64_t preceding = 0;
  for (size_t i = 0; i < ranks.size(); ++i) {
    const int64_t rank = static_cast<int64_t>(ranks[i]);
    if (rank < negative_infinities) {
      selected[i] = -std::numeric_limits<float>::infinity();
    } else if (rank - negative_infinities >= finite_count) {
      selected[i] = std::numeric_limits<float>::infinity();
    } else {
      while (rank - negative_infinities - preceding >= result.distribution[initial_bin])
        preceding += result.distribution[initial_bin++];
      ranges[i] = {FloatLowerKey(result.bin_edges[initial_bin]),
                   initial_bin + 1 == result.distribution.size()
                       ? FloatOrderKey(std::numeric_limits<float>::max())
                       : FloatLowerKey(result.bin_edges[initial_bin + 1]) - 1};
      offsets[i] = rank - negative_infinities - preceding;
      has_finite_rank = true;
    }
  }
  const int64_t passes = has_finite_rank ? 3 : 1;
  ReportProgress(progress, TensorStatisticsPhase::kQuantiles, 0, passes);
  int64_t completed = 0;
  if (has_finite_rank) {
    // Refines only bins containing requested ranks. No tensor-sized selection copy is needed.
    for (int shift : {20, 10, 0}) {
      const int bits = shift == 20 ? 12 : 10;
      const size_t bins = size_t{1} << bits;
      std::vector<Range> groups;
      for (size_t i = 0; i < ranges.size(); ++i)
        if (offsets[i] >= 0 && (groups.empty() || groups.back() != ranges[i]))
          groups.push_back(ranges[i]);
      const size_t cells = groups.size() * bins;
      const int64_t blocks = std::min(
          int64_t{16}, std::min(1 + (result.count - 1) / kParallelForGrainSize,
                                std::max(int64_t{1}, static_cast<int64_t>(1048576 / cells))));
      std::vector<std::vector<int64_t>> histograms(static_cast<size_t>(blocks));
      for (auto &histogram : histograms)
        histogram.resize(cells);
      ParallelFor(blocks, 2, [&](int64_t begin, int64_t end) {
        for (int64_t block = begin; block < end; ++block) {
          auto &histogram = histograms[static_cast<size_t>(block)];
          const int64_t first = block * (result.count / blocks);
          const int64_t last = block == blocks - 1 ? result.count : first + result.count / blocks;
          for (int64_t i = first; i < last; ++i) {
            const uint32_t key = FloatOrderKey(Load<float>(tensor.bytes(), i));
            const auto group = std::lower_bound(
                groups.begin(), groups.end(), key,
                [](const Range &range, uint32_t value) { return range.high < value; });
            if (group != groups.end() && key >= group->low)
              ++histogram[static_cast<size_t>(group - groups.begin()) * bins +
                          ((key >> shift) & (bins - 1))];
          }
        }
      });
      auto &histogram = histograms[0];
      for (size_t block = 1; block < histograms.size(); ++block)
        for (size_t cell = 0; cell < cells; ++cell)
          histogram[cell] += histograms[block][cell];
      for (size_t i = 0; i < ranks.size(); ++i) {
        if (offsets[i] < 0)
          continue;
        const auto group =
            std::lower_bound(groups.begin(), groups.end(), ranges[i].low,
                             [](const Range &range, uint32_t key) { return range.high < key; });
        EXT_ENFORCE(group != groups.end(), "Tensor statistics histogram selection lost a range.");
        const size_t base = static_cast<size_t>(group - groups.begin()) * bins;
        size_t bin = 0;
        while (bin < bins && offsets[i] >= histogram[base + bin])
          offsets[i] -= histogram[base + bin++];
        EXT_ENFORCE(bin < bins, "Tensor statistics histogram selection rank is out of range.");
        const uint64_t prefix = (static_cast<uint64_t>(ranges[i].low) >> (shift + bits))
                                << (shift + bits);
        const uint32_t low = static_cast<uint32_t>(prefix | (bin << shift));
        ranges[i] = {std::max(ranges[i].low, low),
                     std::min(ranges[i].high, low | ((uint32_t{1} << shift) - 1))};
      }
      ReportProgress(progress, TensorStatisticsPhase::kQuantiles, ++completed, passes);
    }
    for (size_t i = 0; i < ranks.size(); ++i)
      if (offsets[i] >= 0)
        selected[i] = FloatFromOrderKey(ranges[i].low);
  } else {
    ReportProgress(progress, TensorStatisticsPhase::kQuantiles, 1, passes);
  }
  SetQuantiles(result, quantiles, median, [&](size_t rank) {
    return selected[static_cast<size_t>(std::lower_bound(ranks.begin(), ranks.end(), rank) -
                                        ranks.begin())];
  });
}

void ComputeQuantiles(TensorStatistics &result, const Tensor &tensor,
                      const std::vector<double> &quantiles, bool median,
                      int64_t negative_infinities,
                      const TensorStatisticsProgressCallback &progress) {
  const auto ranks = QuantileRanks(static_cast<size_t>(result.count), quantiles, median);
  if (tensor.data_type == TensorProto::FLOAT && result.min == result.max) {
    StatisticsPhase(progress, TensorStatisticsPhase::kQuantiles, [&] {
      SetQuantiles(result, quantiles, median, [&](size_t) { return result.min; });
    });
    return;
  }
  if (tensor.data_type == TensorProto::FLOAT && result.count >= kParallelForGrainSize &&
      ranks.size() <= 128) {
    HistogramQuantiles(result, tensor, quantiles, median, ranks, negative_infinities, progress);
    return;
  }
  switch (static_cast<DataType>(tensor.data_type)) {
  case DataType::FLOAT:
    ComputeQuantiles<float>(result, quantiles, median, ranks, progress,
                            [&](int64_t i) { return Load<float>(tensor.bytes(), i); });
    break;
  case DataType::INT64:
    ComputeQuantiles<int64_t>(result, quantiles, median, ranks, progress,
                              [&](int64_t i) { return Load<int64_t>(tensor.bytes(), i); });
    break;
  case DataType::UINT64:
    ComputeQuantiles<uint64_t>(result, quantiles, median, ranks, progress,
                               [&](int64_t i) { return Load<uint64_t>(tensor.bytes(), i); });
    break;
  default:
    // DOUBLE exactly represents every other supported real type, including INT32/UINT32.
    ComputeQuantiles<double>(result, quantiles, median, ranks, progress,
                             [&](int64_t i) { return static_cast<double>(ReadNumber(tensor, i)); });
  }
}

double Mean(const Tensor &tensor, long double scale) {
  const int64_t count = tensor.element_count();
  const int64_t blocks = 1 + (count - 1) / kParallelForGrainSize;
  std::vector<long double> partials(static_cast<size_t>(blocks));
  ParallelFor(blocks, 2, [&](int64_t begin, int64_t end) {
    for (int64_t block = begin; block < end; ++block) {
      const int64_t first = block * kParallelForGrainSize;
      const int64_t last = first + std::min(kParallelForGrainSize, count - first);
      long double total = 0;
      for (int64_t i = first; i < last; ++i)
        total += ReadNumber(tensor, i) / scale;
      partials[static_cast<size_t>(block)] = total;
    }
  });
  long double total = 0;
  for (long double partial : partials)
    total += partial;
  if (!std::isfinite(total))
    return static_cast<double>(total);
  // Accumulates bounded terms, then clamps rounding excursions before restoring the scale.
  // This does not rely on long double having a larger exponent range than double.
  return static_cast<double>(std::clamp(total / count, -1.0L, 1.0L) * scale);
}

void Histogram(TensorStatistics &result, const Tensor &tensor, int64_t bins, long double low,
               long double high) {
  result.distribution.clear();
  result.bin_edges.clear();
  if (result.count < 10)
    return;
  if (low > high)
    return;
  if (bins == 0)
    bins = result.count > 500 ? 50 : 10;
  if (low == high) {
    low -= 0.5L;
    high += 0.5L;
  }
  result.distribution.assign(static_cast<size_t>(bins), 0);
  result.bin_edges.reserve(static_cast<size_t>(bins) + 1);
  for (int64_t i = 0; i <= bins; ++i) {
    const long double weight = static_cast<long double>(i) / bins;
    result.bin_edges.push_back(static_cast<double>(std::lerp(low, high, weight)));
  }
  const double bin_width = (result.bin_edges.back() - result.bin_edges.front()) / bins;
  const bool arithmetic_bins = std::isfinite(bin_width) && bin_width > 0;
  // Caps private histogram storage independently of the tensor size and worker count.
  const int64_t blocks = std::min(1 + (result.count - 1) / kParallelForGrainSize,
                                  std::max(int64_t{1}, int64_t{65536} / bins));
  std::vector<std::vector<int64_t>> partials(static_cast<size_t>(blocks - 1));
  for (auto &counts : partials)
    counts.resize(static_cast<size_t>(bins), 0);
  ParallelFor(blocks, 2, [&](int64_t begin, int64_t end) {
    for (int64_t block = begin; block < end; ++block) {
      auto &counts = block == 0 ? result.distribution : partials[static_cast<size_t>(block - 1)];
      const int64_t first = block * (result.count / blocks);
      const int64_t last = block == blocks - 1 ? result.count : first + result.count / blocks;
      for (int64_t i = first; i < last; ++i) {
        const long double value = ReadNumber(tensor, i);
        if (!std::isfinite(value))
          continue;
        const double rounded = static_cast<double>(value);
        size_t index = counts.size() - 1;
        if (rounded < result.bin_edges.back()) {
          bool found = false;
          if (arithmetic_bins) {
            const double estimate = (rounded - result.bin_edges.front()) / bin_width;
            if (std::isfinite(estimate) && estimate >= 0 &&
                estimate < static_cast<double>(counts.size())) {
              index = static_cast<size_t>(estimate);
              found = index < counts.size() && rounded >= result.bin_edges[index] &&
                      rounded < result.bin_edges[index + 1];
            }
          }
          // Rounded or repeated edges can disagree with the arithmetic estimate.
          if (!found) {
            auto edge = std::upper_bound(result.bin_edges.begin(), result.bin_edges.end(), rounded);
            index =
                edge == result.bin_edges.begin() ? 0 : size_t(edge - result.bin_edges.begin() - 1);
            index = std::min(index, counts.size() - 1);
          }
        }
        ++counts[index];
      }
    }
  });
  for (const auto &counts : partials)
    for (size_t i = 0; i < counts.size(); ++i)
      result.distribution[i] += counts[i];
}

struct RectangleAccumulator {
  long double scale = 0;
  long double total = 0;
  long double correction = 0;
  bool has_nan = false;
  bool positive_infinity = false;
  bool negative_infinity = false;
};

void AccumulateRectangleTotal(RectangleAccumulator &result, long double term) {
  // Compensates rounding even when long double has only double precision.
  const long double adjusted = term - result.correction;
  const long double sum = result.total + adjusted;
  result.correction = (sum - result.total) - adjusted;
  result.total = sum;
}

void AccumulateRectangleValue(RectangleAccumulator &result, const RectangleAccumulator &value,
                              long double weight) {
  result.has_nan |= value.has_nan;
  result.positive_infinity |= value.positive_infinity;
  result.negative_infinity |= value.negative_infinity;
  if (value.scale > 0)
    AccumulateRectangleTotal(result, value.total * (value.scale / result.scale) * weight);
}

void AccumulateRectangleValue(RectangleAccumulator &result, long double value, long double weight) {
  if (std::isnan(value)) {
    result.has_nan = true;
  } else if (std::isinf(value)) {
    result.positive_infinity |= value > 0;
    result.negative_infinity |= value < 0;
  } else {
    if (result.scale > 0)
      AccumulateRectangleTotal(result, (value / result.scale) * weight);
  }
}

long double RectangleScale(long double value) { return std::isfinite(value) ? std::abs(value) : 0; }

long double RectangleScale(const RectangleAccumulator &value) { return value.scale; }

double RectangleValue(const RectangleAccumulator &value) {
  if (value.has_nan || (value.positive_infinity && value.negative_infinity))
    return kNaN;
  if (value.positive_infinity)
    return std::numeric_limits<double>::infinity();
  if (value.negative_infinity)
    return -std::numeric_limits<double>::infinity();
  return static_cast<double>(std::clamp(value.total, -1.0L, 1.0L) * value.scale);
}

template <typename Values>
std::vector<RectangleAccumulator> ReduceRectangle(const Values &values, int64_t rows, int64_t cols,
                                                  int64_t output_rows, int64_t output_cols,
                                                  bool reduce_rows) {
  std::vector<RectangleAccumulator> reduced(static_cast<size_t>(output_rows * output_cols));
  const int64_t input_axis = reduce_rows ? rows : cols;
  const int64_t output_axis = reduce_rows ? output_rows : output_cols;
  ParallelFor(output_rows * output_cols, [&](int64_t begin, int64_t finish) {
    for (int64_t index = begin; index < finish; ++index) {
      const int64_t y = index / output_cols, x = index % output_cols;
      const int64_t output_coordinate = reduce_rows ? y : x;
      const long double start =
          static_cast<long double>(output_coordinate) * input_axis / output_axis;
      const long double end =
          static_cast<long double>(output_coordinate + 1) * input_axis / output_axis;
      const int64_t input_end = std::min(input_axis, static_cast<int64_t>(std::ceil(end)));
      auto &cell = reduced[static_cast<size_t>(index)];
      // Fixes each cell's scale before summing: repeated rescaling also accumulates rounding.
      for (int64_t i = static_cast<int64_t>(start); i < input_end; ++i) {
        const size_t input_index = static_cast<size_t>(reduce_rows ? i * cols + x : y * cols + i);
        cell.scale = std::max(cell.scale, RectangleScale(values[input_index]));
      }
      for (int64_t i = static_cast<int64_t>(start); i < input_end; ++i) {
        const long double weight = (std::min(end, static_cast<long double>(i + 1)) -
                                    std::max(start, static_cast<long double>(i))) /
                                   (end - start);
        const size_t input_index = static_cast<size_t>(reduce_rows ? i * cols + x : y * cols + i);
        AccumulateRectangleValue(cell, values[input_index], weight);
      }
    }
  });
  return reduced;
}

struct TensorNumbers {
  const Tensor &tensor;
  long double operator[](size_t index) const {
    return ReadNumber(tensor, static_cast<int64_t>(index));
  }
};

void Rectangle(TensorStatistics &result, const Tensor &tensor, const std::vector<int64_t> &shape) {
  if (shape.empty())
    return;
  const TensorNumbers values{tensor};
  const int64_t rows = tensor.shape.size() >= 2 ? tensor.shape[0] : 1;
  const int64_t cols = result.count / rows;
  result.rectangle_shape = shape;
  result.rectangle_map.resize(static_cast<size_t>(shape[0] * shape[1]));
  if (rows > shape[0] || cols > shape[1]) {
    std::vector<RectangleAccumulator> reduced;
    if (rows > shape[0]) {
      reduced = ReduceRectangle(values, rows, cols, shape[0], cols, true);
      reduced = ReduceRectangle(reduced, shape[0], cols, shape[0], shape[1], false);
    } else {
      reduced = ReduceRectangle(values, rows, cols, rows, shape[1], false);
      reduced = ReduceRectangle(reduced, rows, shape[1], shape[0], shape[1], true);
    }
    for (size_t i = 0; i < reduced.size(); ++i)
      result.rectangle_map[i] = RectangleValue(reduced[i]);
    return;
  }
  ParallelFor(shape[0] * shape[1], [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t y = index / shape[1], x = index % shape[1];
      const long double top = static_cast<long double>(y) * rows / shape[0];
      const long double bottom = static_cast<long double>(y + 1) * rows / shape[0];
      const long double left = static_cast<long double>(x) * cols / shape[1];
      const long double right = static_cast<long double>(x + 1) * cols / shape[1];
      long double total = 0;
      const int64_t row_end = std::min(rows, static_cast<int64_t>(std::ceil(bottom)));
      const int64_t col_end = std::min(cols, static_cast<int64_t>(std::ceil(right)));
      // Uses only this cell's contributors: an unrelated large value must not cause small
      // representable values to underflow during normalization of an identity or local resize.
      long double scale = 0;
      for (int64_t row = static_cast<int64_t>(top); row < row_end; ++row)
        for (int64_t col = static_cast<int64_t>(left); col < col_end; ++col) {
          const long double value = values[static_cast<size_t>(row * cols + col)];
          if (std::isfinite(value))
            scale = std::max(scale, std::abs(value));
        }
      if (scale == 0)
        scale = 1;
      for (int64_t row = static_cast<int64_t>(top); row < row_end; ++row)
        for (int64_t col = static_cast<int64_t>(left); col < col_end; ++col) {
          const long double height = std::min(bottom, static_cast<long double>(row + 1)) -
                                     std::max(top, static_cast<long double>(row));
          const long double width = std::min(right, static_cast<long double>(col + 1)) -
                                    std::max(left, static_cast<long double>(col));
          if (height > 0 && width > 0)
            total += (values[static_cast<size_t>(row * cols + col)] / scale) *
                     (height / (bottom - top)) * (width / (right - left));
        }
      result.rectangle_map[static_cast<size_t>(y * shape[1] + x)] = static_cast<double>(
          std::isfinite(total) ? std::clamp(total, -1.0L, 1.0L) * scale : total);
    }
  });
}

struct EncodedTensorMetadata {
  int32_t dtype;
  size_t size_bytes;
};

EncodedTensorMetadata EncodedMetadata(const EncodedValueProto &value) {
  EXT_ENFORCE_INVALID(value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "Tensor statistics requires an encoded logical tensor type.");
  return {value.logical_type().tensor_type().elem_type(), value.raw_data().size()};
}

TensorStatistics ComputeStatistics(const Tensor &tensor, int64_t bins,
                                   const std::vector<double> &quantiles, bool median,
                                   const std::vector<int64_t> &rectangle_shape,
                                   bool report_distinct,
                                   const TensorStatisticsProgressCallback &progress) {
  const auto type = static_cast<DataType>(tensor.data_type);
  EXT_ENFORCE_INVALID(type != DataType::STRING && type != DataType::COMPLEX64 &&
                          type != DataType::COMPLEX128 && type != DataType::UNDEFINED,
                      "Tensor statistics requires a real numeric tensor.");
  TensorStatistics result;
  result.dtype = tensor.data_type;
  result.shape = tensor.shape;
  result.size_bytes = tensor.size_bytes();
  result.count = tensor.shape.product(0, tensor.shape.size(), "tensor statistics");
  const size_t required = PackedByteSize(tensor.data_type, result.count);
  EXT_ENFORCE_INVALID(tensor.size_bytes() >= required && (required == 0 || tensor.bytes()),
                      "Tensor statistics tensor buffer is truncated or null.");
  result.quantiles.assign(quantiles.size(), kNaN);
  if (median)
    result.median = kNaN;
  if (report_distinct)
    result.distinct_count = 0;
  if (result.count == 0) {
    ReportProgress(progress, TensorStatisticsPhase::kComplete, 1);
    return result;
  }
  struct Summary {
    long double low = std::numeric_limits<long double>::infinity(), high = -low;
    long double finite_low = low, finite_high = high;
    long double scale = 0;
    bool has_nan = false;
    int64_t negative_infinities = 0;
  };
  const int64_t blocks = 1 + (result.count - 1) / kParallelForGrainSize;
  std::vector<Summary> summaries(static_cast<size_t>(blocks));
  StatisticsPhase(progress, TensorStatisticsPhase::kSummary, [&] {
    ParallelFor(blocks, 2, [&](int64_t begin, int64_t end) {
      for (int64_t block = begin; block < end; ++block) {
        Summary &summary = summaries[static_cast<size_t>(block)];
        const int64_t first = block * kParallelForGrainSize;
        const int64_t last = first + std::min(kParallelForGrainSize, result.count - first);
        for (int64_t i = first; i < last; ++i) {
          const long double value = ReadNumber(tensor, i);
          summary.has_nan |= std::isnan(value);
          summary.low = std::min(summary.low, value);
          summary.high = std::max(summary.high, value);
          if (std::isfinite(value)) {
            summary.scale = std::max(summary.scale, std::abs(value));
            summary.finite_low = std::min(summary.finite_low, value);
            summary.finite_high = std::max(summary.finite_high, value);
          } else if (value < 0) {
            ++summary.negative_infinities;
          }
        }
      }
    });
  });
  bool has_nan = false;
  int64_t negative_infinities = 0;
  long double scale = 0;
  long double low = std::numeric_limits<long double>::infinity(), high = -low;
  long double finite_low = low, finite_high = high;
  for (const auto &summary : summaries) {
    has_nan |= summary.has_nan;
    low = std::min(low, summary.low);
    high = std::max(high, summary.high);
    scale = std::max(scale, summary.scale);
    finite_low = std::min(finite_low, summary.finite_low);
    finite_high = std::max(finite_high, summary.finite_high);
    negative_infinities += summary.negative_infinities;
  }
  if (scale == 0)
    scale = 1;
  if (!has_nan) {
    result.min = static_cast<double>(low);
    result.max = static_cast<double>(high);
    StatisticsPhase(progress, TensorStatisticsPhase::kMean,
                    [&] { result.mean = Mean(tensor, scale); });
  }
  if (report_distinct)
    StatisticsPhase(progress, TensorStatisticsPhase::kDistinct,
                    [&] { result.distinct_count = DistinctCount(tensor, has_nan); });
  if (result.count >= 10 && finite_low <= finite_high)
    StatisticsPhase(progress, TensorStatisticsPhase::kHistogram,
                    [&] { Histogram(result, tensor, bins, finite_low, finite_high); });
  if (!rectangle_shape.empty())
    StatisticsPhase(progress, TensorStatisticsPhase::kRectangle,
                    [&] { Rectangle(result, tensor, rectangle_shape); });
  if ((!quantiles.empty() || median) && !has_nan)
    ComputeQuantiles(result, tensor, quantiles, median, negative_infinities, progress);
  ReportProgress(progress, TensorStatisticsPhase::kComplete, 1);
  return result;
}

} // namespace

// Keeps the callback-free ABI for already compiled SDK consumers.
TensorStatistics ComputeTensorStatistics(const Tensor &tensor, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         size_t max_output_bytes) {
  return ComputeTensorStatistics(tensor, bins, quantiles, median, rectangle_shape, max_output_bytes,
                                 {});
}

TensorStatistics ComputeTensorStatistics(const Tensor &tensor, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         size_t max_output_bytes,
                                         const TensorStatisticsProgressCallback &progress) {
  ValidateTensorStatisticsOptions(bins, quantiles, median, rectangle_shape, max_output_bytes);
  const auto type = static_cast<TensorProto::DataType>(tensor.data_type);
  const bool report_distinct =
      symbolic::IsIntegerTensorType(symbolic::DataTypeToTensorType(type)) ||
      type == TensorProto::BOOL || type == TensorProto::INT2 || type == TensorProto::UINT2 ||
      type == TensorProto::INT4 || type == TensorProto::UINT4;
  return ComputeStatistics(tensor, bins, quantiles, median, rectangle_shape, report_distinct,
                           progress);
}

// Keeps the callback-free ABI for already compiled SDK consumers.
TensorStatistics ComputeTensorStatistics(const RuntimeValue &value, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         size_t max_output_bytes) {
  return ComputeTensorStatistics(value, bins, quantiles, median, rectangle_shape, max_output_bytes,
                                 {});
}

TensorStatistics ComputeTensorStatistics(const RuntimeValue &value, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         size_t max_output_bytes,
                                         const TensorStatisticsProgressCallback &progress) {
  if (value.kind == RuntimeValue::Kind::kTensor)
    return ComputeTensorStatistics(value.tensor, bins, quantiles, median, rectangle_shape,
                                   max_output_bytes, progress);
  ValidateTensorStatisticsOptions(bins, quantiles, median, rectangle_shape, max_output_bytes);
  EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kEncoded && value.encoded,
                      "Tensor statistics requires a tensor or encoded tensor RuntimeValue.");
  const auto metadata = EncodedMetadata(value.Encoded());
  Tensor decoded;
  StatisticsPhase(progress, TensorStatisticsPhase::kDecode,
                  [&] { decoded = DequantizeTensor(value); });
  auto result =
      ComputeStatistics(decoded, bins, quantiles, median, rectangle_shape, true, progress);
  result.dtype = metadata.dtype;
  result.size_bytes = metadata.size_bytes;
  return result;
}

// Keeps the callback-free ABI for already compiled SDK consumers.
TensorStatistics ComputeTensorStatistics(const EncodedValueProto &value, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         const StructTypeCatalogue &catalogue,
                                         size_t max_output_bytes) {
  return ComputeTensorStatistics(value, bins, quantiles, median, rectangle_shape, catalogue,
                                 max_output_bytes, {});
}

TensorStatistics ComputeTensorStatistics(const EncodedValueProto &value, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         const StructTypeCatalogue &catalogue,
                                         size_t max_output_bytes,
                                         const TensorStatisticsProgressCallback &progress) {
  ValidateTensorStatisticsOptions(bins, quantiles, median, rectangle_shape, max_output_bytes);
  const auto metadata = EncodedMetadata(value);
  Tensor decoded;
  StatisticsPhase(progress, TensorStatisticsPhase::kDecode,
                  [&] { decoded = DequantizeTensor(value, catalogue); });
  auto result =
      ComputeStatistics(decoded, bins, quantiles, median, rectangle_shape, true, progress);
  result.dtype = metadata.dtype;
  result.size_bytes = metadata.size_bytes;
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
