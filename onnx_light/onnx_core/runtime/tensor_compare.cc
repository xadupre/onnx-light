// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tensor_compare.h"

#include "onnx_core/runtime/cast_helper.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

// Reads element ``i`` of a numeric tensor as ``double``. ``dtype`` must be one
// of the types handled below; ``NumericDecodable`` gates the callers so the
// default case is unreachable in practice.
double ReadElementAsDouble(int32_t dtype, const uint8_t *ptr, int64_t i) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
    return static_cast<double>(reinterpret_cast<const float *>(ptr)[i]);
  case DataType::DOUBLE:
    return reinterpret_cast<const double *>(ptr)[i];
  case DataType::FLOAT16:
    return static_cast<double>(Float16BitsToFloat(reinterpret_cast<const uint16_t *>(ptr)[i]));
  case DataType::BFLOAT16:
    return static_cast<double>(Bfloat16BitsToFloat(reinterpret_cast<const uint16_t *>(ptr)[i]));
  case DataType::INT8:
    return static_cast<double>(reinterpret_cast<const int8_t *>(ptr)[i]);
  case DataType::INT16:
    return static_cast<double>(reinterpret_cast<const int16_t *>(ptr)[i]);
  case DataType::INT32:
    return static_cast<double>(reinterpret_cast<const int32_t *>(ptr)[i]);
  case DataType::INT64:
    return static_cast<double>(reinterpret_cast<const int64_t *>(ptr)[i]);
  case DataType::UINT8:
  case DataType::BOOL:
    return static_cast<double>(ptr[i]);
  case DataType::UINT16:
    return static_cast<double>(reinterpret_cast<const uint16_t *>(ptr)[i]);
  case DataType::UINT32:
    return static_cast<double>(reinterpret_cast<const uint32_t *>(ptr)[i]);
  case DataType::UINT64:
    return static_cast<double>(reinterpret_cast<const uint64_t *>(ptr)[i]);
  default:
    return 0.0;
  }
}

// Returns whether ``dtype`` can be decoded to ``double`` element-wise by
// :cpp:func:`ReadElementAsDouble` (i.e. is comparable with a tolerance).
bool NumericDecodable(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
  case DataType::DOUBLE:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::INT8:
  case DataType::INT16:
  case DataType::INT32:
  case DataType::INT64:
  case DataType::UINT8:
  case DataType::BOOL:
  case DataType::UINT16:
  case DataType::UINT32:
  case DataType::UINT64:
    return true;
  default:
    return false;
  }
}

std::string ShapeToString(const Shape &shape) {
  std::ostringstream oss;
  oss << '(';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << ')';
  return oss.str();
}

TensorComparison Mismatch(std::string message) {
  return TensorComparison{false, std::move(message)};
}

} // namespace

TensorComparison CompareTensors(const Tensor &actual, const Tensor &expected, double rtol,
                                double atol, bool equal_nan) {
  if (actual.data_type != expected.data_type) {
    std::ostringstream oss;
    oss << "data_type mismatch: actual=" << actual.data_type << " expected=" << expected.data_type;
    return Mismatch(oss.str());
  }
  if (actual.shape != expected.shape) {
    std::ostringstream oss;
    oss << "shape mismatch: actual=" << ShapeToString(actual.shape)
        << " expected=" << ShapeToString(expected.shape);
    return Mismatch(oss.str());
  }

  if (static_cast<DataType>(actual.data_type) == DataType::STRING) {
    if (actual.string_data.size() != expected.string_data.size()) {
      std::ostringstream oss;
      oss << "string element count mismatch: actual=" << actual.string_data.size()
          << " expected=" << expected.string_data.size();
      return Mismatch(oss.str());
    }
    for (size_t i = 0; i < actual.string_data.size(); ++i) {
      if (actual.string_data[i] != expected.string_data[i]) {
        std::ostringstream oss;
        oss << "string mismatch at index " << i << ": actual='" << actual.string_data[i]
            << "' expected='" << expected.string_data[i] << "'";
        return Mismatch(oss.str());
      }
    }
    return TensorComparison{true, ""};
  }

  if (!NumericDecodable(actual.data_type)) {
    // Types that cannot be decoded to ``double`` (FLOAT8*, packed 4-bit / 2-bit)
    // are compared byte-for-byte.
    if (actual.size_bytes() != expected.size_bytes()) {
      std::ostringstream oss;
      oss << "byte size mismatch: actual=" << actual.size_bytes()
          << " expected=" << expected.size_bytes();
      return Mismatch(oss.str());
    }
    if (actual.size_bytes() != 0 &&
        std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()) != 0) {
      return Mismatch("raw byte content differs for a non-tolerance-comparable data_type");
    }
    return TensorComparison{true, ""};
  }

  const int64_t count = expected.element_count();
  const uint8_t *ap = actual.bytes();
  const uint8_t *ep = expected.bytes();
  for (int64_t i = 0; i < count; ++i) {
    const double a = ReadElementAsDouble(actual.data_type, ap, i);
    const double b = ReadElementAsDouble(expected.data_type, ep, i);
    if (std::isnan(a) || std::isnan(b)) {
      if (equal_nan && std::isnan(a) && std::isnan(b)) {
        continue;
      }
      std::ostringstream oss;
      if (std::isnan(a) && std::isnan(b)) {
        oss << "NaN mismatch at index " << i << ": both values are NaN but equal_nan is false";
      } else {
        oss << "NaN mismatch at index " << i << ": actual=" << a << " expected=" << b;
      }
      return Mismatch(oss.str());
    }
    if (std::isinf(a) || std::isinf(b)) {
      if (a == b) {
        continue;
      }
      std::ostringstream oss;
      oss << "infinity mismatch at index " << i << ": actual=" << a << " expected=" << b;
      return Mismatch(oss.str());
    }
    const double diff = std::fabs(a - b);
    if (diff > atol + rtol * std::fabs(b)) {
      std::ostringstream oss;
      oss << "value mismatch at index " << i << ": actual=" << a << " expected=" << b
          << " (|diff|=" << diff << " > atol=" << atol << " + rtol=" << rtol
          << " * |expected|=" << std::fabs(b) << ")";
      return Mismatch(oss.str());
    }
  }
  return TensorComparison{true, ""};
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
