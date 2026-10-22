// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/tensor_statistics.h"

#include "onnx_core/runtime/kernels/cast_float8.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/cast_sub_byte.h"
#include "onnx_core/runtime/quantization.h"
#include <cmath>
#include <cstring>
#include <set>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void ValidateOptions(int64_t bins, const std::vector<double> &quantiles,
                     const std::vector<int64_t> &rectangle) {
  EXT_ENFORCE_INVALID(bins >= 0, "Tensor statistics bins must be non-negative.");
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(bins) < std::vector<double>().max_size() &&
                          static_cast<uint64_t>(bins) <= std::vector<int64_t>().max_size(),
                      "Tensor statistics histogram allocation overflows.");
  for (double q : quantiles)
    EXT_ENFORCE_INVALID(std::isfinite(q) && q >= 0 && q <= 1,
                        "Tensor statistics quantiles must be finite and in [0,1].");
  if (!rectangle.empty()) {
    EXT_ENFORCE_INVALID(rectangle.size() == 2 && rectangle[0] > 0 && rectangle[1] > 0,
                        "Tensor statistics rectangle must contain positive [height,width].");
    EXT_ENFORCE_INVALID(rectangle[0] <= std::numeric_limits<int64_t>::max() / rectangle[1],
                        "Tensor statistics rectangle size overflows.");
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(rectangle[0] * rectangle[1]) <=
                            std::vector<double>().max_size(),
                        "Tensor statistics rectangle allocation overflows.");
  }
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

int64_t DistinctCount(const Tensor &tensor, const std::vector<long double> &values, bool has_nan) {
  // Integer keys must not pass through floating-point conversion, including on platforms where
  // long double has only double precision. memcpy also permits unaligned borrowed tensor storage.
  if (tensor.data_type == TensorProto::INT64) {
    std::set<int64_t> distinct;
    for (size_t i = 0; i < values.size(); ++i)
      distinct.insert(Load<int64_t>(tensor.bytes(), static_cast<int64_t>(i)));
    return static_cast<int64_t>(distinct.size());
  }
  if (tensor.data_type == TensorProto::UINT64) {
    std::set<uint64_t> distinct;
    for (size_t i = 0; i < values.size(); ++i)
      distinct.insert(Load<uint64_t>(tensor.bytes(), static_cast<int64_t>(i)));
    return static_cast<int64_t>(distinct.size());
  }
  std::set<long double> distinct;
  for (long double value : values)
    if (!std::isnan(value))
      distinct.insert(value);
  return static_cast<int64_t>(distinct.size()) + (has_nan ? 1 : 0);
}

double Quantile(const std::vector<long double> &sorted, double q) {
  const long double position = static_cast<long double>(q) * (sorted.size() - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, sorted.size() - 1);
  const long double weight = position - lower;
  if (weight == 0 || sorted[lower] == sorted[upper])
    return static_cast<double>(sorted[lower]);
  if (std::isfinite(sorted[lower]) && std::isfinite(sorted[upper]))
    return static_cast<double>(std::lerp(sorted[lower], sorted[upper], weight));
  return static_cast<double>((1 - weight) * sorted[lower] + weight * sorted[upper]);
}

long double FiniteScale(const std::vector<long double> &values) {
  long double scale = 0;
  for (long double value : values)
    if (std::isfinite(value))
      scale = std::max(scale, std::abs(value));
  return scale == 0 ? 1 : scale;
}

double Mean(const std::vector<long double> &values, long double scale) {
  long double total = 0;
  bool positive_infinity = false, negative_infinity = false;
  for (long double value : values) {
    if (std::isinf(value)) {
      positive_infinity |= value > 0;
      negative_infinity |= value < 0;
    } else {
      total += value / scale;
    }
  }
  if (positive_infinity && negative_infinity)
    return kNaN;
  if (positive_infinity || negative_infinity)
    return positive_infinity ? std::numeric_limits<double>::infinity()
                             : -std::numeric_limits<double>::infinity();
  // Accumulates bounded terms, then clamps rounding excursions before restoring the scale.
  // This does not rely on long double having a larger exponent range than double.
  return static_cast<double>(std::clamp(total / values.size(), -1.0L, 1.0L) * scale);
}

void Histogram(TensorStatistics &result, const std::vector<long double> &values, int64_t bins) {
  if (result.count < 10)
    return;
  long double low = std::numeric_limits<long double>::infinity(), high = -low;
  for (long double value : values)
    if (std::isfinite(value)) {
      low = std::min(low, value);
      high = std::max(high, value);
    }
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
  for (long double value : values) {
    if (!std::isfinite(value))
      continue;
    // Search the exported edges so rounded edges and histogram counts agree exactly.
    const double rounded = static_cast<double>(value);
    auto edge = std::upper_bound(result.bin_edges.begin(), result.bin_edges.end(), rounded);
    size_t index =
        edge == result.bin_edges.begin() ? 0 : size_t(edge - result.bin_edges.begin() - 1);
    ++result.distribution[std::min(index, result.distribution.size() - 1)];
  }
}

void Rectangle(TensorStatistics &result, const Tensor &tensor,
               const std::vector<long double> &values, const std::vector<int64_t> &shape) {
  if (shape.empty())
    return;
  const int64_t rows = tensor.shape.size() >= 2 ? tensor.shape[0] : 1;
  const int64_t cols = result.count / rows;
  result.rectangle_shape = shape;
  result.rectangle_map.resize(static_cast<size_t>(shape[0] * shape[1]));
  for (int64_t y = 0; y < shape[0]; ++y) {
    const long double top = static_cast<long double>(y) * rows / shape[0];
    const long double bottom = static_cast<long double>(y + 1) * rows / shape[0];
    for (int64_t x = 0; x < shape[1]; ++x) {
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
  }
}

void EncodedMetadata(TensorStatistics &result, const EncodedValueProto &value) {
  // DequantizeTensor validates logical dtype and shape before we reach this point.
  result.dtype = value.logical_type().tensor_type().elem_type();
  result.size_bytes = value.raw_data().size();
}

bool IsInteger(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::INT8:
  case DataType::INT16:
  case DataType::INT32:
  case DataType::INT64:
  case DataType::UINT8:
  case DataType::UINT16:
  case DataType::UINT32:
  case DataType::UINT64:
  case DataType::INT2:
  case DataType::UINT2:
  case DataType::INT4:
  case DataType::UINT4:
  case DataType::BOOL:
    return true;
  default:
    return false;
  }
}

TensorStatistics ComputeStatistics(const Tensor &tensor, int64_t bins,
                                   const std::vector<double> &quantiles, bool median,
                                   const std::vector<int64_t> &rectangle_shape,
                                   bool report_distinct) {
  ValidateOptions(bins, quantiles, rectangle_shape);
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
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(result.count) <= std::vector<long double>().max_size(),
                      "Tensor statistics numeric allocation overflows.");
  result.quantiles.assign(quantiles.size(), kNaN);
  if (median)
    result.median = kNaN;
  if (report_distinct)
    result.distinct_count = 0;
  if (result.count == 0)
    return result;
  std::vector<long double> values;
  values.reserve(static_cast<size_t>(result.count));
  bool has_nan = false;
  long double low = std::numeric_limits<long double>::infinity(), high = -low;
  for (int64_t i = 0; i < result.count; ++i) {
    const long double value = ReadNumber(tensor, i);
    values.push_back(value);
    has_nan |= std::isnan(value);
    low = std::min(low, value);
    high = std::max(high, value);
  }
  const long double scale = FiniteScale(values);
  if (!has_nan) {
    result.min = static_cast<double>(low);
    result.max = static_cast<double>(high);
    result.mean = Mean(values, scale);
  }
  if (report_distinct)
    result.distinct_count = DistinctCount(tensor, values, has_nan);
  Histogram(result, values, bins);
  Rectangle(result, tensor, values, rectangle_shape);
  if ((!quantiles.empty() || median) && !has_nan) {
    std::sort(values.begin(), values.end());
    for (size_t i = 0; i < quantiles.size(); ++i)
      result.quantiles[i] = Quantile(values, quantiles[i]);
    if (median)
      result.median = Quantile(values, 0.5);
  }
  return result;
}

} // namespace

TensorStatistics ComputeTensorStatistics(const Tensor &tensor, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape) {
  return ComputeStatistics(tensor, bins, quantiles, median, rectangle_shape,
                           IsInteger(tensor.data_type));
}

TensorStatistics ComputeTensorStatistics(const RuntimeValue &value, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape) {
  ValidateOptions(bins, quantiles, rectangle_shape);
  if (value.kind == RuntimeValue::Kind::kTensor)
    return ComputeTensorStatistics(value.tensor, bins, quantiles, median, rectangle_shape);
  EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kEncoded && value.encoded,
                      "Tensor statistics requires a tensor or encoded tensor RuntimeValue.");
  auto result =
      ComputeStatistics(DequantizeTensor(value), bins, quantiles, median, rectangle_shape, true);
  EncodedMetadata(result, value.Encoded());
  return result;
}

TensorStatistics ComputeTensorStatistics(const EncodedValueProto &value, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         const StructTypeCatalogue &catalogue) {
  ValidateOptions(bins, quantiles, rectangle_shape);
  auto result = ComputeStatistics(DequantizeTensor(value, catalogue), bins, quantiles, median,
                                  rectangle_shape, true);
  EncodedMetadata(result, value);
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
