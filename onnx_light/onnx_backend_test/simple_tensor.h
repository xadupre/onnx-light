// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

/**
 * Tensor — minimal runtime tensor used by backend test cases.
 *
 * This struct is intentionally distinct from ``TensorProto``: it carries no
 * protobuf wire dependency, owns its bytes in row-major little-endian layout,
 * and is meant to be consumed directly by a runtime exercising a single
 * backend test node case.
 */
struct Tensor {
  /// Optional name of the tensor (input/output name in the test model).
  std::string name;
  /// Element data type stored as a ``TensorProto::DataType`` integer value.
  int32_t data_type = 0;
  /// Tensor shape; an empty shape denotes a scalar (element_count == 1).
  std::vector<int64_t> shape;
  /// Raw element bytes in row-major little-endian layout.
  std::vector<uint8_t> data;

  Tensor() = default;
  Tensor(std::string n, int32_t dt, std::vector<int64_t> s, std::vector<uint8_t> d)
      : name(std::move(n)), data_type(dt), shape(std::move(s)), data(std::move(d)) {}

  /// Returns the product of all shape dimensions; 1 for an empty shape.
  int64_t element_count() const;

  /// Returns the size in bytes of one element of ``data_type``.
  /// Throws ``std::invalid_argument`` for unsupported types.
  size_t element_size() const;

  /// Typed factories that construct a tensor of the given shape and copy the
  /// provided values into ``data``. They throw ``std::invalid_argument`` if
  /// any dimension in ``shape`` is negative or if ``values.size()`` does not
  /// match ``prod(shape)``.
  ///
  /// The templated ``From<T>`` factory is the generic version. The non-template
  /// ``FromFloat``/``FromDouble``/``FromInt32``/``FromInt64`` are thin wrappers
  /// kept for source compatibility.
  template <typename T>
  static Tensor From(const std::string &name, const std::vector<int64_t> &shape,
                     const std::vector<T> &values);

  static Tensor FromFloat(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<float> &values);
  static Tensor FromDouble(const std::string &name, const std::vector<int64_t> &shape,
                           const std::vector<double> &values);
  static Tensor FromInt32(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<int32_t> &values);
  static Tensor FromInt64(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<int64_t> &values);

  /// Typed views over the underlying ``data`` buffer. They throw if the
  /// requested type does not match ``data_type``.
  ///
  /// The templated ``As<T>()`` accessor is the generic version. The non-template
  /// ``AsFloat``/``AsDouble``/``AsInt32``/``AsInt64`` are thin wrappers kept for
  /// source compatibility.
  template <typename T> const T *As() const;
  template <typename T> T *As();

  const float *AsFloat() const;
  float *AsFloat();
  const double *AsDouble() const;
  double *AsDouble();
  const int32_t *AsInt32() const;
  int32_t *AsInt32();
  const int64_t *AsInt64() const;
  int64_t *AsInt64();
};

/// Trait mapping a C++ element type to its ``TensorProto::DataType`` value.
/// Specialize to support additional element types in ``Tensor::From``/``As``.
template <typename T> struct TensorElementType; // primary template intentionally undefined

#define ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(CPP_TYPE, ENUM_VALUE)                               \
  template <> struct TensorElementType<CPP_TYPE> {                                                 \
    static constexpr int32_t value = ENUM_VALUE;                                                   \
  }

ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(float, TensorProto::DataType::FLOAT);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(double, TensorProto::DataType::DOUBLE);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int32_t, TensorProto::DataType::INT32);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int64_t, TensorProto::DataType::INT64);

#undef ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE

template <typename T>
Tensor Tensor::From(const std::string &name, const std::vector<int64_t> &shape,
                    const std::vector<T> &values) {
  int64_t expected = 1;
  for (int64_t d : shape) {
    if (d < 0) {
      throw std::invalid_argument("Tensor shape dimensions must be non-negative.");
    }
    expected *= d;
  }
  if (static_cast<int64_t>(values.size()) != expected) {
    throw std::invalid_argument("Tensor values size does not match the product of shape.");
  }
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return Tensor(name, TensorElementType<T>::value, shape, std::move(bytes));
}

template <typename T> const T *Tensor::As() const {
  if (data_type != TensorElementType<T>::value) {
    throw std::invalid_argument("Tensor data_type does not match the requested view type.");
  }
  return reinterpret_cast<const T *>(data.data());
}

template <typename T> T *Tensor::As() {
  if (data_type != TensorElementType<T>::value) {
    throw std::invalid_argument("Tensor data_type does not match the requested view type.");
  }
  return reinterpret_cast<T *>(data.data());
}

/// Returns the size in bytes of one element of ``dtype``
/// (a ``TensorProto::DataType`` integer). Throws ``std::invalid_argument``
/// for unsupported types.
size_t ElementSize(int32_t dtype);

/// Fills ``vi`` with the type/shape information described by ``tensor``.
/// ``vi.name`` is set to ``tensor.name``.
void FillValueInfo(const Tensor &tensor, ValueInfoProto &vi);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
