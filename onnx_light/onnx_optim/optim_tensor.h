// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "onnx_light_helpers.h"
#include "onnx_op/light_op_schema.h"
#include "onnx_proto/onnx.h"

/**
 * @file optim_tensor.h
 * @brief Lightweight, non-owning tensor description used by ONNX
 *        graph optimisation passes.
 *
 * The ``onnx_optim`` namespace exposes three small value types:
 *
 *   - :cpp:class:`OptimDim` — a single shape dimension, either a
 *     concrete ``int64_t`` or a symbolic string expression.
 *   - :cpp:class:`OptimShape` — an ordered, bounded-rank collection of
 *     :cpp:class:`OptimDim`.
 *   - :cpp:class:`OptimTensor` — a non-owning view over a contiguous
 *     buffer carrying a :cpp:type:`TensorType`, an
 *     :cpp:class:`OptimShape`, and an optional shape annotation when
 *     the tensor itself represents a shape (e.g. the ``shape`` input of
 *     ``Reshape``).
 *
 * These types are intentionally header-only friendly and never
 * allocate the tensor data they describe; callers are responsible for
 * the lifetime of the underlying buffer.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {

/// Reuse the TensorType enumeration defined by ``onnx_op`` so that
/// ``onnx_optim`` is fully aligned with the rest of the operator stack.
using TensorType = ONNX_LIGHT_NAMESPACE::onnx_op::TensorType;

/**
 * Maps a ``TensorProto::DataType`` to the matching :cpp:type:`TensorType`
 * enumerator used by :cpp:class:`OptimTensor`. Returns
 * :cpp:enumerator:`TensorType::kUndefined` for any data type that is not
 * representable in the ``onnx_optim`` stack (e.g. ``UNDEFINED``).
 */
TensorType DataTypeToTensorType(TensorProto::DataType dtype);

// Returns ``true`` when ``t`` is an integer scalar/element type for
// which ``ValueAsShape`` is meaningful (i.e. the tensor's content can
// legitimately be interpreted as shape dimensions).
bool IsIntegerTensorType(TensorType t);

/**
 * A single shape dimension that is either a concrete non-negative integer or
 * a symbolic expression represented as a string. ``OptimDim`` is used by
 * ``OptimShape`` to describe both fully-known and partially-symbolic shapes.
 */
class OptimDim {
public:
  /// Default constructs a zero-valued integer dimension.
  OptimDim() : value_(static_cast<int64_t>(0)) {}

  /// Constructs an integer-valued dimension.
  OptimDim(int64_t value) : value_(value) {}

  /// Constructs a symbolic dimension expressed as a string expression.
  OptimDim(std::string expr) : value_(std::move(expr)) {}

  /// Convenience overload for C string literals.
  OptimDim(const char *expr) : value_(std::string(expr)) {}

  /// Returns ``true`` when the dimension holds a concrete integer value.
  bool IsInt() const noexcept { return std::holds_alternative<int64_t>(value_); }

  /// Returns ``true`` when the dimension holds a symbolic expression string.
  bool IsExpr() const noexcept { return std::holds_alternative<std::string>(value_); }

  /// Returns the integer value. Throws ``std::bad_variant_access`` if the
  /// dimension is not an integer.
  int64_t AsInt() const { return std::get<int64_t>(value_); }

  /// Returns the symbolic expression. Throws ``std::bad_variant_access`` if
  /// the dimension is not a string expression.
  const std::string &AsExpr() const { return std::get<std::string>(value_); }

  /// Equality compares both the alternative and the underlying value.
  bool operator==(const OptimDim &other) const noexcept { return value_ == other.value_; }
  bool operator!=(const OptimDim &other) const noexcept { return !(*this == other); }

private:
  std::variant<int64_t, std::string> value_;
};

/// Maximum number of dimensions an ``OptimShape`` can describe inline. Shapes
/// in practice rarely exceed this rank, so storing the dimensions in a small
/// container keeps the structure compact and cache-friendly.
inline constexpr std::size_t kMaxOptimRank = 8;

/**
 * A short, value-typed shape composed of ``OptimDim`` entries. The rank is
 * bounded by ``kMaxOptimRank`` so that the structure fits comfortably on the
 * stack. Symbolic and concrete dimensions can be mixed freely.
 */
class OptimShape {
public:
  using value_type = OptimDim;

  OptimShape() = default;

  /// Constructs a shape from an initializer list of dimensions.
  OptimShape(std::initializer_list<OptimDim> dims);

  /// Constructs a shape from any iterable container of ``OptimDim``.
  explicit OptimShape(const std::vector<OptimDim> &dims);

  /// Number of dimensions.
  std::size_t Rank() const noexcept { return dims_.size(); }

  /// ``true`` when the shape contains no dimensions (scalar / rank-0).
  bool Empty() const noexcept { return dims_.empty(); }

  /// Access a dimension by index. Throws ``std::out_of_range`` if invalid.
  const OptimDim &operator[](std::size_t i) const { return dims_.at(i); }
  OptimDim &operator[](std::size_t i) { return dims_.at(i); }

  /// Appends a new dimension. Throws ``std::length_error`` when the maximum
  /// rank ``kMaxOptimRank`` would be exceeded.
  void PushBack(OptimDim dim);

  /// Returns ``true`` when every dimension is a concrete integer.
  bool IsFullyKnown() const noexcept;

  /// Computes the product of all integer dimensions. Returns ``1`` for a
  /// rank-0 (empty) shape, matching the standard scalar-element-count
  /// semantic. Throws ``std::runtime_error`` if any dimension is symbolic.
  int64_t NumElements() const;

  /// Equality compares the dimensions element-wise.
  bool operator==(const OptimShape &other) const noexcept { return dims_ == other.dims_; }
  bool operator!=(const OptimShape &other) const noexcept { return !(*this == other); }

  /// Read-only access to the underlying dimensions.
  const std::vector<OptimDim> &Dims() const noexcept { return dims_; }

private:
  std::vector<OptimDim> dims_;
};

OptimShape ShapeFromTensorProtoDims(const TensorProto &tensor_proto);

/**
 * A non-owning view over a contiguous tensor buffer. ``OptimTensor`` never
 * allocates: the caller is responsible for the lifetime of the underlying
 * memory referenced by ``data``. The shape may contain symbolic dimensions
 * which is useful for representing intermediate values during optimisation
 * passes where the concrete shape is not yet known.
 */
class OptimTensor {
public:
  /// Constructs an empty (null) tensor.
  OptimTensor() = default;

  /**
   * Constructs an ``OptimTensor`` referencing an external buffer.
   *
   * @param data Pointer to the first element of the externally-owned buffer.
   *             May be ``nullptr`` only when ``shape`` is empty or all
   *             integer dimensions are zero.
   * @param dtype Element type of the data referenced by ``data``.
   * @param shape Shape describing the layout of the data.
   */
  OptimTensor(void *data, TensorType dtype, OptimShape shape)
      : data_(data), dtype_(dtype), shape_(std::move(shape)) {}

  /// Pointer to the externally-owned buffer. The view itself is ``const``
  /// but the buffer it references is not, mirroring ``std::span`` semantics.
  void *Data() const noexcept { return data_; }

  /// Element type of the referenced buffer.
  TensorType Dtype() const noexcept { return dtype_; }

  /// Shape of the tensor (may contain symbolic dimensions).
  const OptimShape &Shape() const noexcept { return shape_; }
  OptimShape &Shape() noexcept { return shape_; }

  /// ``true`` when the tensor has no associated data pointer.
  bool IsNull() const noexcept { return data_ == nullptr; }

  /**
   * Tags the tensor as carrying a shape value (e.g. the ``shape`` input of a
   * ``Reshape`` node) and stores that shape. An empty ``OptimShape`` is
   * permitted: it denotes a rank-0 / scalar shape value.
   */
  void SetValueAsShape(OptimShape shape) { value_as_shape_ = std::move(shape); }

  /// Clears the value-as-shape annotation so that ``HasValueAsShape`` becomes
  /// ``false`` again.
  void ClearValueAsShape() noexcept { value_as_shape_.reset(); }

  /// ``true`` when the tensor's value is interpreted as a shape. An empty
  /// stored shape still returns ``true`` — use ``ValueAsShape().Empty()`` to
  /// distinguish the empty case.
  bool HasValueAsShape() const noexcept { return value_as_shape_.has_value(); }

  /// Returns the shape value carried by this tensor. Throws
  /// ``std::bad_optional_access`` if ``HasValueAsShape()`` is ``false``.
  const OptimShape &ValueAsShape() const { return value_as_shape_.value(); }
  OptimShape &ValueAsShape() { return value_as_shape_.value(); }

  /// Equality compares the data pointer, dtype, shape, and the optional
  /// value-as-shape annotation. Because :cpp:class:`OptimTensor` is a
  /// non-owning view, two tensors are considered equal only when they refer
  /// to the same external buffer.
  bool operator==(const OptimTensor &other) const noexcept {
    return data_ == other.data_ && dtype_ == other.dtype_ && shape_ == other.shape_ &&
           value_as_shape_ == other.value_as_shape_;
  }
  bool operator!=(const OptimTensor &other) const noexcept { return !(*this == other); }

private:
  void *data_ = nullptr;
  TensorType dtype_ = TensorType::kUndefined;
  OptimShape shape_{};
  std::optional<OptimShape> value_as_shape_{};
};

} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
