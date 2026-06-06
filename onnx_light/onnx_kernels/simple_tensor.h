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
namespace onnx_kernels {

/**
 * DataType — alias for ``TensorProto::DataType``.
 *
 * Provides a short, namespace-local name for the upstream ONNX
 * ``TensorProto::DataType`` enumeration so backend test code can write
 * ``DataType::INT64`` instead of fully qualifying every reference.
 */
using DataType = TensorProto::DataType;

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
  /// Element data type stored as a ``DataType`` integer value.
  int32_t data_type = 0;
  /// Tensor shape; an empty shape denotes a scalar (element_count == 1).
  std::vector<int64_t> shape;
  /// Raw element bytes in row-major little-endian layout.
  ///
  /// Unused (empty) when ``data_type`` is ``DataType::STRING``;
  /// in that case the element values are stored in ``string_data`` instead
  /// since UTF-8 strings are variable length and do not have a fixed byte
  /// stride compatible with this raw buffer.
  std::vector<uint8_t> data;

  /// String element values in row-major layout. Populated only when
  /// ``data_type`` is ``DataType::STRING``; empty for all other
  /// element types.
  std::vector<std::string> string_data;

  Tensor() = default;
  Tensor(std::string n, int32_t dt, std::vector<int64_t> s, std::vector<uint8_t> d)
      : name(std::move(n)), data_type(dt), shape(std::move(s)), data(std::move(d)) {}
  /// Constructs a ``STRING`` tensor whose elements live in ``string_data``.
  /// Distinct from the bytes-based constructor so brace-enclosed
  /// ``{ ... }`` initializer lists at call sites are unambiguous.
  static Tensor MakeString(std::string n, std::vector<int64_t> s, std::vector<std::string> sd) {
    Tensor t;
    t.name = std::move(n);
    t.data_type = static_cast<int32_t>(DataType::STRING);
    t.shape = std::move(s);
    t.string_data = std::move(sd);
    return t;
  }

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
  static Tensor FromInt8(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<int8_t> &values);
  static Tensor FromUint8(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<uint8_t> &values);
  static Tensor FromInt16(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<int16_t> &values);
  static Tensor FromUint16(const std::string &name, const std::vector<int64_t> &shape,
                           const std::vector<uint16_t> &values);
  static Tensor FromUint32(const std::string &name, const std::vector<int64_t> &shape,
                           const std::vector<uint32_t> &values);
  static Tensor FromUint64(const std::string &name, const std::vector<int64_t> &shape,
                           const std::vector<uint64_t> &values);
  /// Constructs a ``BOOL`` tensor; element values are stored as one byte each
  /// (0 == false, non-zero == true). Provided as a ``uint8_t`` vector so the
  /// usual ``std::vector<bool>`` packing pitfalls are avoided.
  static Tensor FromBool(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<uint8_t> &values);
  /// Constructs a ``STRING`` tensor whose elements are the provided UTF-8
  /// strings (stored in ``string_data``). Throws ``std::invalid_argument`` if
  /// any dimension in ``shape`` is negative or if ``values.size()`` does not
  /// match ``prod(shape)``.
  static Tensor FromStrings(const std::string &name, const std::vector<int64_t> &shape,
                            const std::vector<std::string> &values);

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
  const int8_t *AsInt8() const;
  int8_t *AsInt8();
  const uint8_t *AsUint8() const;
  uint8_t *AsUint8();
  const int16_t *AsInt16() const;
  int16_t *AsInt16();
  const uint16_t *AsUint16() const;
  uint16_t *AsUint16();
  const uint32_t *AsUint32() const;
  uint32_t *AsUint32();
  const uint64_t *AsUint64() const;
  uint64_t *AsUint64();
  /// Typed view over ``data`` for ``BOOL`` element type, stored one byte per
  /// element. The byte value is 0 for false and non-zero (canonically 1) for
  /// true.
  const uint8_t *AsBool() const;
  uint8_t *AsBool();

  /// Typed view over the underlying ``string_data`` buffer. Throws
  /// ``std::invalid_argument`` if ``data_type`` is not
  /// ``DataType::STRING``.
  const std::vector<std::string> &AsStrings() const;
  std::vector<std::string> &AsStrings();
};

/// Trait mapping a C++ element type to its ``DataType`` value.
/// Specialize to support additional element types in ``Tensor::From``/``As``.
template <typename T> struct TensorElementType; // primary template intentionally undefined

#define ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(CPP_TYPE, ENUM_VALUE)                               \
  template <> struct TensorElementType<CPP_TYPE> {                                                 \
    static constexpr int32_t value = ENUM_VALUE;                                                   \
  }

ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(float, DataType::FLOAT);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(double, DataType::DOUBLE);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int16_t, DataType::INT16);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int32_t, DataType::INT32);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int64_t, DataType::INT64);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int8_t, DataType::INT8);
// Note: ``uint8_t`` aliases both ``UINT8`` and ``BOOL`` element storage; the
// trait maps it to ``UINT8`` and ``BOOL`` accessors go through ``AsBool``
// which uses the same byte layout but validates ``data_type == BOOL``.
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(uint8_t, DataType::UINT8);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(uint16_t, DataType::UINT16);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(uint32_t, DataType::UINT32);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(uint64_t, DataType::UINT64);

#undef ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE

template <typename T>
Tensor Tensor::From(const std::string &name, const std::vector<int64_t> &shape,
                    const std::vector<T> &values) {
  int64_t expected = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, "Tensor shape dimensions must be non-negative.");
    expected *= d;
  }
  EXT_ENFORCE_INVALID(static_cast<int64_t>(values.size()) == expected,
                      "Tensor values size does not match the product of shape.");
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return Tensor(name, TensorElementType<T>::value, shape, std::move(bytes));
}

template <typename T> const T *Tensor::As() const {
  EXT_ENFORCE_INVALID(data_type == TensorElementType<T>::value,
                      "Tensor data_type does not match the requested view type.");
  return reinterpret_cast<const T *>(data.data());
}

template <typename T> T *Tensor::As() {
  EXT_ENFORCE_INVALID(data_type == TensorElementType<T>::value,
                      "Tensor data_type does not match the requested view type.");
  return reinterpret_cast<T *>(data.data());
}

/// Returns the size in bytes of one element of ``dtype``
/// (a ``DataType`` integer). Throws ``std::invalid_argument``
/// for unsupported types. Sub-byte packed dtypes (INT4/UINT4/INT2/UINT2)
/// are not supported by this helper because they do not have a whole-byte
/// per-element size; use ``PackedByteSize`` instead.
size_t ElementSize(int32_t dtype);

/// Returns the storage size in bytes for ``element_count`` elements of
/// ``dtype``. Whole-byte dtypes return ``element_count * ElementSize(dtype)``;
/// sub-byte packed dtypes pack 2 (INT4/UINT4) or 4 (INT2/UINT2) elements per
/// byte and ``element_count`` is rounded up to fill the trailing byte. Throws
/// ``std::invalid_argument`` for unsupported types.
size_t PackedByteSize(int32_t dtype, int64_t element_count);

/// Fills ``vi`` with the type/shape information described by ``tensor``.
/// ``vi.name`` is set to ``tensor.name``.
void FillValueInfo(const Tensor &tensor, ValueInfoProto &vi);

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
