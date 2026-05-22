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
  /// provided values into ``data``. They throw if ``values.size()`` does not
  /// match ``prod(shape)``.
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
  const float *AsFloat() const;
  float *AsFloat();
  const double *AsDouble() const;
  double *AsDouble();
  const int32_t *AsInt32() const;
  int32_t *AsInt32();
  const int64_t *AsInt64() const;
  int64_t *AsInt64();
};

/// Returns the size in bytes of one element of ``dtype``
/// (a ``TensorProto::DataType`` integer). Throws for unsupported types.
size_t ElementSize(int32_t dtype);

/// Fills ``vi`` with the type/shape information described by ``tensor``.
/// ``vi.name`` is set to ``tensor.name``.
void FillValueInfo(const Tensor &tensor, ValueInfoProto &vi);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
